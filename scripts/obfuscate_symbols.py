#!/usr/bin/env python3
"""
심볼 + 문자열 난독화 스크립트
모든 사용자 정의 식별자와 민감한 문자열을 랜덤 문자열로 치환
"""

import os
import re
import random
import string
import shutil
import json
from pathlib import Path
from typing import Dict, Set, List

# 난독화 설정
CONFIG = {
    "source_dir": "needaimbot",
    "output_dir": "needaimbot_obf",
    "extensions": [".cpp", ".h", ".cu", ".hpp", ".cuh"],
    "min_identifier_length": 3,  # 이 길이 이상만 난독화
    "obfuscated_length": 8,      # 난독화된 이름 길이
    "seed": None,                # None이면 랜덤, 숫자면 고정 시드
}

# =============================================================================
# 민감한 키워드 (문자열 리터럴 내에서도 치환)
# =============================================================================
SENSITIVE_KEYWORDS = {
    # AI/ML 관련
    "YOLO": "GraphProcessor",
    "yolo": "graphprocessor", 
    "detection": "analysis",
    "Detection": "Analysis",
    "DETECTION": "ANALYSIS",
    "detect": "analyze",
    "Detect": "Analyze",
    "TensorRT": "ComputeEngine",
    "tensorrt": "computeengine",
    "inference": "processing",
    "Inference": "Processing",
    "neural": "compute",
    "Neural": "Compute",
    "model": "module",
    "Model": "Module",
    "onnx": "data",
    "ONNX": "DATA",
    
    # 에이밍 관련
    "aim": "focus",
    "Aim": "Focus",
    "AIM": "FOCUS",
    "aimbot": "assistant",
    "Aimbot": "Assistant",
    "target": "point",
    "Target": "Point",
    "TARGET": "POINT",
    "headshot": "priority",
    "Headshot": "Priority",
    "recoil": "stability",
    "Recoil": "Stability",
    "tracking": "following",
    "Tracking": "Following",
    
    # 마우스/입력 관련
    "mouse": "input",
    "Mouse": "Input",
    "MOUSE": "INPUT",
    "kmbox": "device",
    "KMBox": "Device",
    "KMBOX": "DEVICE",
    "ghub": "driver",
    "Ghub": "Driver",
    "GHUB": "DRIVER",
    "logitech": "peripheral",
    "Logitech": "Peripheral",
    "arduino": "controller",
    "Arduino": "Controller",
    "ARDUINO": "CONTROLLER",
    "serial": "comm",
    "Serial": "Comm",
    "SERIAL": "COMM",
    
    # 화면 캡처 관련
    "capture": "acquire",
    "Capture": "Acquire",
    "CAPTURE": "ACQUIRE",
    "screenshot": "snapshot",
    "Screenshot": "Snapshot",
    "screen": "display",
    "Screen": "Display",
    "SCREEN": "DISPLAY",
    "Desktop Duplication": "Display Mirror",
    "duplication": "mirror",
    "Duplication": "Mirror",
    
    # 오버레이 관련
    "overlay": "layer",
    "Overlay": "Layer",
    "OVERLAY": "LAYER",
    "esp": "visual",
    "ESP": "VISUAL",
    "wallhack": "enhancement",
    "cheat": "tool",
    "Cheat": "Tool",
    "hack": "mod",
    "Hack": "Mod",
    
    # 게임 관련
    "game": "app",
    "Game": "App",
    "GAME": "APP",
    "player": "entity",
    "Player": "Entity",
    "PLAYER": "ENTITY",
    "enemy": "object",
    "Enemy": "Object",
    "ENEMY": "OBJECT",
    "hitbox": "region",
    "Hitbox": "Region",
    "bbox": "rect",
    "BBox": "Rect",
    "bounding": "enclosing",
    "Bounding": "Enclosing",
    
    # 프로젝트 특정
    "needaimbot": "nvcontainer",
    "NeedAimbot": "NVContainer",
    "NEEDAIMBOT": "NVCONTAINER",
}

# 난독화하지 않을 키워드/식별자
RESERVED_KEYWORDS = {
    # C++ 키워드
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
    "bool", "break", "case", "catch", "char", "char8_t", "char16_t", "char32_t",
    "class", "compl", "concept", "const", "consteval", "constexpr", "constinit",
    "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype",
    "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
    "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected",
    "public", "register", "reinterpret_cast", "requires", "return", "short",
    "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
    "switch", "template", "this", "thread_local", "throw", "true", "try",
    "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual",
    "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
    
    # 전처리기 지시문 (절대 난독화 금지!)
    "pragma", "include", "define", "undef", "ifdef", "ifndef", "endif", "elif",
    "error", "warning", "line", "once",
    
    # CUDA 키워드
    "__global__", "__device__", "__host__", "__shared__", "__constant__",
    "__restrict__", "__launch_bounds__", "dim3", "blockIdx", "blockDim",
    "threadIdx", "gridDim", "warpSize",
    
    # Windows/MSVC
    "__declspec", "__cdecl", "__stdcall", "__fastcall", "__thiscall",
    "__forceinline", "__inline", "WINAPI", "CALLBACK", "APIENTRY",
    "DWORD", "WORD", "BYTE", "BOOL", "HANDLE", "HWND", "HINSTANCE",
    "LPVOID", "LPCSTR", "LPWSTR", "LPCWSTR", "HRESULT", "UINT", "INT",
    "LONG", "ULONG", "SHORT", "USHORT", "CHAR", "UCHAR", "WCHAR",
    "TRUE", "FALSE", "NULL", "nullptr",
    
    # DirectX
    "ID3D11Device", "ID3D11DeviceContext", "ID3D11Texture2D", "ID3D11Buffer",
    "ID3D11ShaderResourceView", "ID3D11RenderTargetView", "IDXGISwapChain",
    "D3D11_TEXTURE2D_DESC", "D3D11_BUFFER_DESC", "D3D11_MAPPED_SUBRESOURCE",
    "DXGI_FORMAT", "DXGI_SWAP_CHAIN_DESC",
    
    # TensorRT/CUDA
    "nvinfer1", "IRuntime", "ICudaEngine", "IExecutionContext", "IBuilder",
    "INetworkDefinition", "IBuilderConfig", "IHostMemory", "Dims", "Dims2",
    "Dims3", "Dims4", "DataType", "cudaStream_t", "cudaError_t", "cudaMalloc",
    "cudaFree", "cudaMemcpy", "cudaMemcpyAsync", "cudaDeviceSynchronize",
    "cudaStreamCreate", "cudaStreamDestroy", "cudaGetLastError",
    
    # STL
    "std", "string", "vector", "map", "unordered_map", "set", "unordered_set",
    "array", "list", "deque", "queue", "stack", "pair", "tuple", "optional",
    "variant", "any", "shared_ptr", "unique_ptr", "weak_ptr", "make_shared",
    "make_unique", "move", "forward", "swap", "begin", "end", "size", "empty",
    "push_back", "emplace_back", "insert", "erase", "find", "count", "clear",
    "reserve", "resize", "data", "c_str", "substr", "length", "append",
    "cout", "cin", "cerr", "endl", "flush", "ifstream", "ofstream", "fstream",
    "stringstream", "ostringstream", "istringstream", "getline", "stoi", "stof",
    "stod", "to_string", "printf", "sprintf", "snprintf", "fprintf", "scanf",
    "sscanf", "fscanf", "fopen", "fclose", "fread", "fwrite", "fseek", "ftell",
    "memcpy", "memset", "memmove", "memcmp", "strlen", "strcpy", "strncpy",
    "strcmp", "strncmp", "strcat", "strncat", "strstr", "strchr", "strrchr",
    "malloc", "calloc", "realloc", "free", "new", "delete",
    "thread", "mutex", "lock_guard", "unique_lock", "condition_variable",
    "atomic", "future", "promise", "async", "chrono", "duration", "time_point",
    "system_clock", "steady_clock", "high_resolution_clock", "sleep_for",
    "sleep_until", "yield", "get_id", "join", "detach", "joinable",
    "exception", "runtime_error", "logic_error", "invalid_argument",
    "out_of_range", "overflow_error", "underflow_error", "what", "throw",
    "try", "catch", "noexcept",
    "min", "max", "abs", "sqrt", "pow", "exp", "log", "log10", "sin", "cos",
    "tan", "asin", "acos", "atan", "atan2", "ceil", "floor", "round", "fmod",
    "isnan", "isinf", "isfinite", "numeric_limits", "INT_MAX", "INT_MIN",
    "UINT_MAX", "FLOAT_MAX", "DOUBLE_MAX", "FLT_MAX", "DBL_MAX",
    "sort", "stable_sort", "partial_sort", "nth_element", "binary_search",
    "lower_bound", "upper_bound", "equal_range", "merge", "unique", "reverse",
    "rotate", "shuffle", "random_shuffle", "next_permutation", "prev_permutation",
    "copy", "copy_if", "copy_n", "fill", "fill_n", "transform", "replace",
    "replace_if", "remove", "remove_if", "accumulate", "inner_product",
    "adjacent_difference", "partial_sum", "iota", "reduce", "transform_reduce",
    "for_each", "all_of", "any_of", "none_of", "count_if", "find_if",
    "find_if_not", "search", "search_n", "mismatch", "equal",
    
    # ImGui
    "ImGui", "ImVec2", "ImVec4", "ImColor", "ImDrawList", "ImFont", "ImFontAtlas",
    "ImGuiIO", "ImGuiStyle", "ImGuiContext", "ImTextureID", "ImGuiWindowFlags",
    "ImGuiInputTextFlags", "ImGuiTreeNodeFlags", "ImGuiSelectableFlags",
    "ImGuiColorEditFlags", "ImGuiSliderFlags", "ImGuiKey", "ImGuiMouseButton",
    "Begin", "End", "BeginChild", "EndChild", "Text", "TextColored", "Button",
    "Checkbox", "RadioButton", "SliderFloat", "SliderInt", "InputText",
    "InputFloat", "InputInt", "ColorEdit3", "ColorEdit4", "TreeNode", "TreePop",
    "Selectable", "ListBox", "Combo", "SetNextWindowPos", "SetNextWindowSize",
    "GetWindowPos", "GetWindowSize", "GetIO", "GetStyle", "GetDrawList",
    "SameLine", "NewLine", "Separator", "Spacing", "Indent", "Unindent",
    "PushStyleColor", "PopStyleColor", "PushStyleVar", "PopStyleVar",
    "PushItemWidth", "PopItemWidth", "SetCursorPos", "GetCursorPos",
    "IsItemHovered", "IsItemClicked", "IsItemActive", "IsWindowFocused",
    "SetKeyboardFocusHere", "OpenPopup", "BeginPopup", "EndPopup",
    "BeginPopupModal", "EndPopup", "CloseCurrentPopup", "BeginMenu",
    "EndMenu", "MenuItem", "BeginMenuBar", "EndMenuBar", "BeginMainMenuBar",
    "EndMainMenuBar", "Columns", "NextColumn", "GetColumnWidth", "SetColumnWidth",
    "BeginTable", "EndTable", "TableNextRow", "TableNextColumn", "TableSetColumnIndex",
    "Render", "GetDrawData", "CreateContext", "DestroyContext", "GetCurrentContext",
    "SetCurrentContext", "StyleColorsDark", "StyleColorsLight", "StyleColorsClassic",
    
    # 프로젝트 외부 인터페이스 (유지 필요)
    "main", "WinMain", "wWinMain", "DllMain", "WndProc", "WindowProc",
}

# 난독화하지 않을 패턴 (정규식)
EXCLUDE_PATTERNS = [
    r"^_+",           # 언더스코어로 시작
    r"^[A-Z_]+$",     # 전체 대문자 (매크로)
    r"^[a-z]$",       # 단일 문자
    r"^\d",           # 숫자로 시작
    r"^gl[A-Z]",      # OpenGL 함수
    r"^cu[A-Z]",      # CUDA 함수
    r"^nv[A-Z]",      # NVIDIA 함수
    r"^D3D",          # DirectX
    r"^DXGI",         # DXGI
    r"^ID3D",         # DirectX 인터페이스
    r"^IDXGI",        # DXGI 인터페이스
    r"^Im[A-Z]",      # ImGui
]

class SymbolObfuscator:
    def __init__(self, config: dict):
        self.config = config
        self.symbol_map: Dict[str, str] = {}
        self.used_names: Set[str] = set()
        self.string_replacements: Dict[str, str] = SENSITIVE_KEYWORDS.copy()
        
        if config["seed"] is not None:
            random.seed(config["seed"])
    
    def generate_random_name(self) -> str:
        """랜덤 식별자 생성"""
        while True:
            # 첫 글자는 문자 또는 언더스코어
            first = random.choice(string.ascii_letters + "_")
            # 나머지는 문자, 숫자, 언더스코어
            rest = ''.join(random.choices(
                string.ascii_letters + string.digits + "_",
                k=self.config["obfuscated_length"] - 1
            ))
            name = first + rest
            if name not in self.used_names and name not in RESERVED_KEYWORDS:
                self.used_names.add(name)
                return name
    
    def should_obfuscate(self, identifier: str) -> bool:
        """이 식별자를 난독화해야 하는지 확인"""
        if len(identifier) < self.config["min_identifier_length"]:
            return False
        if identifier in RESERVED_KEYWORDS:
            return False
        for pattern in EXCLUDE_PATTERNS:
            if re.match(pattern, identifier):
                return False
        return True
    
    def extract_identifiers(self, content: str) -> Set[str]:
        """소스 코드에서 식별자 추출"""
        # 문자열 리터럴 제거
        content = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', content)
        content = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", content)
        # 주석 제거
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # 전처리기 지시문 제거
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        
        # 식별자 추출
        identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', content))
        return {id for id in identifiers if self.should_obfuscate(id)}
    
    def collect_all_identifiers(self, source_dir: Path) -> Set[str]:
        """모든 소스 파일에서 식별자 수집"""
        all_identifiers = set()
        for ext in self.config["extensions"]:
            for file_path in source_dir.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    all_identifiers.update(self.extract_identifiers(content))
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
        return all_identifiers
    
    def build_symbol_map(self, identifiers: Set[str]):
        """식별자 -> 난독화된 이름 매핑 생성"""
        for identifier in sorted(identifiers):  # 정렬하여 일관성 유지
            if identifier not in self.symbol_map:
                self.symbol_map[identifier] = self.generate_random_name()
    
    def obfuscate_content(self, content: str) -> str:
        """소스 코드 내용 난독화 (최적화 버전)"""
        # =================================================================
        # Step 0: 전처리기 지시문 보존 (#pragma, #include, #define 등)
        # =================================================================
        preprocessor_lines = []
        def save_preprocessor(match):
            preprocessor_lines.append(match.group(0))
            return f"\x00PP{len(preprocessor_lines)-1}\x00"
        
        # 전처리기 지시문 전체 라인 보존
        content = re.sub(r'^[ \t]*#[^\n]*$', save_preprocessor, content, flags=re.MULTILINE)
        
        # =================================================================
        # Step 1: 문자열 리터럴 내 민감한 키워드 치환
        # =================================================================
        def replace_in_string(match):
            string_content = match.group(0)
            quote_char = string_content[0]
            inner = string_content[1:-1]
            
            for original, replacement in self._sorted_string_replacements:
                inner = inner.replace(original, replacement)
            
            return quote_char + inner + quote_char
        
        content = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', replace_in_string, content)
        
        # =================================================================
        # Step 2: 문자열 리터럴 보존
        # =================================================================
        strings = []
        def save_string(match):
            strings.append(match.group(0))
            return f"\x00STR{len(strings)-1}\x00"
        
        content = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', save_string, content)
        content = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", save_string, content)
        
        # =================================================================
        # Step 3: 주석 보존
        # =================================================================
        comments = []
        def save_comment(match):
            comments.append(match.group(0))
            return f"\x00CMT{len(comments)-1}\x00"
        
        content = re.sub(r'//[^\n]*', save_comment, content)
        content = re.sub(r'/\*.*?\*/', save_comment, content, flags=re.DOTALL)
        
        # =================================================================
        # Step 4: 단일 정규식으로 모든 식별자 한번에 치환 (핵심 최적화)
        # =================================================================
        def replace_identifier(match):
            word = match.group(0)
            return self.symbol_map.get(word, word)
        
        content = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', replace_identifier, content)
        
        # =================================================================
        # Step 5: 보존된 내용 복원 (역순)
        # =================================================================
        # 주석 복원
        for i, c in enumerate(comments):
            content = content.replace(f"\x00CMT{i}\x00", c)
        
        # 문자열 리터럴 복원
        for i, s in enumerate(strings):
            content = content.replace(f"\x00STR{i}\x00", s)
        
        # 전처리기 지시문 복원
        for i, p in enumerate(preprocessor_lines):
            content = content.replace(f"\x00PP{i}\x00", p)
        
        return content
    
    def _prepare_replacements(self):
        """치환 목록 미리 정렬 (한번만 실행)"""
        self._sorted_string_replacements = sorted(
            self.string_replacements.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
    
    def obfuscate_directory(self):
        """전체 디렉토리 난독화"""
        source_dir = Path(self.config["source_dir"])
        output_dir = Path(self.config["output_dir"])
        
        # 출력 디렉토리 생성
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        # 전체 복사 (모든 파일 포함)
        shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
        
        # 복사 확인
        copied_files = list(output_dir.rglob("*"))
        print(f"Copied {len(copied_files)} files/folders to {output_dir}")
        
        print(f"Collecting identifiers from {source_dir}...")
        identifiers = self.collect_all_identifiers(source_dir)
        print(f"Found {len(identifiers)} unique identifiers to obfuscate")
        
        self.build_symbol_map(identifiers)
        self._prepare_replacements()  # 치환 목록 미리 준비
        
        # =================================================================
        # Step 1: 파일 내용 난독화
        # =================================================================
        obfuscated_count = 0
        for ext in self.config["extensions"]:
            for file_path in output_dir.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    obfuscated = self.obfuscate_content(content)
                    file_path.write_text(obfuscated, encoding='utf-8')
                    obfuscated_count += 1
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
        
        print(f"Obfuscated {obfuscated_count} files")
        
        # =================================================================
        # Step 2: 파일명/폴더명 난독화 (비활성화 - include 경로 호환성)
        # =================================================================
        # print("Obfuscating file and folder names...")
        # self._obfuscate_filenames(output_dir)
        self.file_rename_map = {}  # 빈 맵
        
        # =================================================================
        # Step 3: #include 경로 업데이트 (비활성화)
        # =================================================================
        # print("Updating include paths...")
        # self._update_include_paths(output_dir)
        
        # 심볼 맵 저장 (디버깅용, 배포 시 삭제)
        map_file = output_dir / "_symbol_map.json"
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump({
                "symbols": self.symbol_map,
                "strings": self.string_replacements,
                "files": self.file_rename_map
            }, f, indent=2, ensure_ascii=False)
        print(f"Symbol map saved to {map_file}")
        
        return self.symbol_map
    
    def _obfuscate_filenames(self, output_dir: Path):
        """파일명과 폴더명에서 민감한 키워드 치환"""
        self.file_rename_map = {}
        
        # 파일명 치환 매핑 생성 (needaimbot은 제외 - CMake 호환성)
        filename_replacements = {
            "mouse": "input",
            "capture": "acquire",
            "detection": "analysis",
            "overlay": "layer",
            "keyboard": "hid",
            "target": "point",
            "aim": "focus",
            # "needaimbot" 제외 - 메인 파일명 유지 필요
        }
        
        # 모든 파일/폴더 수집 (깊은 것부터 처리)
        all_paths = sorted(output_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        
        for path in all_paths:
            if path.name.startswith("_"):  # _symbol_map.json 등 제외
                continue
                
            new_name = path.name
            for old, new in filename_replacements.items():
                new_name = new_name.replace(old, new)
            
            if new_name != path.name:
                new_path = path.parent / new_name
                if not new_path.exists():
                    path.rename(new_path)
                    self.file_rename_map[str(path.relative_to(output_dir))] = str(new_path.relative_to(output_dir))
    
    def _update_include_paths(self, output_dir: Path):
        """#include 경로 업데이트"""
        include_replacements = {
            "mouse/": "input/",
            "capture/": "acquire/",
            "detection/": "analysis/",
            "overlay/": "layer/",
            "keyboard/": "hid/",
            "mouse.h": "input.h",
            "capture.h": "acquire.h",
            "overlay.h": "layer.h",
            "keyboard_listener": "hid_listener",
            "dda_capture": "dda_acquire",
        }
        
        for ext in self.config["extensions"]:
            for file_path in output_dir.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    for old, new in include_replacements.items():
                        content = content.replace(f'#include "{old}', f'#include "{new}')
                        content = content.replace(f'#include <{old}', f'#include <{new}')
                        content = content.replace(f'"{old}', f'"{new}')
                    
                    file_path.write_text(content, encoding='utf-8')
                except Exception as e:
                    print(f"Warning: Could not update includes in {file_path}: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="C++ Symbol Obfuscator")
    parser.add_argument("--source", default="needaimbot", help="Source directory")
    parser.add_argument("--output", default="needaimbot_obf", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--length", type=int, default=8, help="Obfuscated name length")
    parser.add_argument("--cmake", action="store_true", help="Generate obfuscated CMakeLists.txt")
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config["source_dir"] = args.source
    config["output_dir"] = args.output
    config["seed"] = args.seed
    config["obfuscated_length"] = args.length
    
    obfuscator = SymbolObfuscator(config)
    obfuscator.obfuscate_directory()
    
    # CMakeLists.txt 생성
    if args.cmake:
        generate_obfuscated_cmake(args.output)
    
    print("\nObfuscation complete!")


def generate_obfuscated_cmake(output_dir: str):
    """난독화된 소스용 CMakeLists.txt 생성"""
    cmake_path = Path("CMakeLists.txt")
    if not cmake_path.exists():
        print("Warning: CMakeLists.txt not found")
        return
    
    content = cmake_path.read_text(encoding='utf-8')
    
    # 경로 치환
    replacements = {
        "needaimbot/": f"{output_dir}/",
        "needaimbot\\\\": f"{output_dir}\\\\",
        "/needaimbot": f"/{output_dir}",
        # 파일명 치환
        "mouse/": "input/",
        "capture/": "acquire/",
        "overlay/": "layer/",
        "keyboard/": "hid/",
        "dda_capture": "dda_acquire",
        "keyboard_listener": "hid_listener",
        "mouse.cpp": "input.cpp",
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    output_cmake = Path("CMakeLists_obf.txt")
    output_cmake.write_text(content, encoding='utf-8')
    print(f"Generated {output_cmake}")


if __name__ == "__main__":
    main()
