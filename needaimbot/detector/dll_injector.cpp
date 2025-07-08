#include <windows.h>
#include <tlhelp32.h>
#include <iostream>
#include <string>

class DllInjector {
private:
    HANDLE m_process;
    DWORD m_process_id;
    
public:
    DllInjector() : m_process(nullptr), m_process_id(0) {}
    
    ~DllInjector() {
        if (m_process) {
            CloseHandle(m_process);
        }
    }
    
    bool OpenProcess(DWORD process_id) {
        m_process_id = process_id;
        m_process = ::OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | 
                                  PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
                                  FALSE, process_id);
        return m_process != nullptr;
    }
    
    bool InjectDll(const std::wstring& dll_path) {
        if (!m_process) return false;
        
        // Check if DLL file exists
        if (GetFileAttributesW(dll_path.c_str()) == INVALID_FILE_ATTRIBUTES) {
            std::wcout << L"DLL file not found: " << dll_path << std::endl;
            return false;
        }
        
        // Get LoadLibraryW address
        HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        if (!kernel32) return false;
        
        FARPROC loadLibraryW = GetProcAddress(kernel32, "LoadLibraryW");
        if (!loadLibraryW) return false;
        
        // Calculate memory size needed
        size_t dll_path_size = (dll_path.length() + 1) * sizeof(wchar_t);
        
        // Allocate memory in target process
        LPVOID remote_memory = VirtualAllocEx(m_process, nullptr, dll_path_size, 
                                             MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!remote_memory) {
            std::wcout << L"Failed to allocate memory in target process" << std::endl;
            return false;
        }
        
        // Write DLL path to target process
        SIZE_T bytes_written = 0;
        if (!WriteProcessMemory(m_process, remote_memory, dll_path.c_str(), 
                               dll_path_size, &bytes_written)) {
            std::wcout << L"Failed to write DLL path to target process" << std::endl;
            VirtualFreeEx(m_process, remote_memory, 0, MEM_RELEASE);
            return false;
        }
        
        // Create remote thread to load DLL
        HANDLE remote_thread = CreateRemoteThread(m_process, nullptr, 0,
                                                 (LPTHREAD_START_ROUTINE)loadLibraryW,
                                                 remote_memory, 0, nullptr);
        if (!remote_thread) {
            std::wcout << L"Failed to create remote thread" << std::endl;
            VirtualFreeEx(m_process, remote_memory, 0, MEM_RELEASE);
            return false;
        }
        
        // Wait for thread to complete (reduced timeout for faster response)
        DWORD wait_result = WaitForSingleObject(remote_thread, 1000);
        if (wait_result != WAIT_OBJECT_0) {
            std::wcout << L"Remote thread timed out or failed" << std::endl;
            TerminateThread(remote_thread, 0);
            CloseHandle(remote_thread);
            VirtualFreeEx(m_process, remote_memory, 0, MEM_RELEASE);
            return false;
        }
        
        // Get thread exit code (LoadLibraryW return value)
        DWORD exit_code = 0;
        GetExitCodeThread(remote_thread, &exit_code);
        
        // Cleanup
        CloseHandle(remote_thread);
        VirtualFreeEx(m_process, remote_memory, 0, MEM_RELEASE);
        
        if (exit_code == 0) {
            std::wcout << L"LoadLibraryW failed in target process" << std::endl;
            return false;
        }
        
        std::wcout << L"DLL injected successfully, module handle: 0x" << std::hex << exit_code << std::endl;
        return true;
    }
    
    static DWORD FindProcessByName(const std::wstring& process_name) {
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return 0;
        
        PROCESSENTRY32W pe32;
        pe32.dwSize = sizeof(PROCESSENTRY32W);
        
        DWORD process_id = 0;
        if (Process32FirstW(snapshot, &pe32)) {
            do {
                if (process_name == pe32.szExeFile) {
                    process_id = pe32.th32ProcessID;
                    break;
                }
            } while (Process32NextW(snapshot, &pe32));
        }
        
        CloseHandle(snapshot);
        return process_id;
    }
    
    static DWORD FindProcessByWindowTitle(const std::wstring& window_title) {
        HWND hwnd = FindWindowW(nullptr, window_title.c_str());
        if (!hwnd) return 0;
        
        DWORD process_id = 0;
        GetWindowThreadProcessId(hwnd, &process_id);
        return process_id;
    }
};

// Function to inject DLL into target process
bool InjectDllIntoProcess(DWORD process_id, const std::wstring& dll_path) {
    DllInjector injector;
    if (!injector.OpenProcess(process_id)) {
        std::wcout << L"Failed to open process " << process_id << std::endl;
        return false;
    }
    
    return injector.InjectDll(dll_path);
}

// Function to inject DLL into process by window title
bool InjectDllIntoWindow(const std::wstring& window_title, const std::wstring& dll_path) {
    DWORD process_id = DllInjector::FindProcessByWindowTitle(window_title);
    if (process_id == 0) {
        std::wcout << L"Window not found: " << window_title << std::endl;
        return false;
    }
    
    std::wcout << L"Found process ID " << process_id << L" for window: " << window_title << std::endl;
    return InjectDllIntoProcess(process_id, dll_path);
}

// Main function for standalone injector
int wmain(int argc, wchar_t* argv[]) {
    if (argc != 3) {
        std::wcout << L"Usage: " << argv[0] << L" <dll_path> <process_id>" << std::endl;
        std::wcout << L"   or: " << argv[0] << L" <dll_path> \"<window_title>\"" << std::endl;
        return 1;
    }
    
    std::wstring dll_path = argv[1];
    std::wstring target = argv[2];
    
    // Check if target is a number (process ID) or string (window title)
    bool is_process_id = true;
    for (wchar_t c : target) {
        if (!iswdigit(c)) {
            is_process_id = false;
            break;
        }
    }
    
    bool success = false;
    if (is_process_id) {
        DWORD process_id = std::wcstoul(target.c_str(), nullptr, 10);
        success = InjectDllIntoProcess(process_id, dll_path);
    } else {
        success = InjectDllIntoWindow(target, dll_path);
    }
    
    if (success) {
        std::wcout << L"DLL injection successful!" << std::endl;
        return 0;
    } else {
        std::wcout << L"DLL injection failed!" << std::endl;
        return 1;
    }
}