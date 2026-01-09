#include "gui_app.h"
#include "engine_exporter.h"
#include "config.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

GuiApp::GuiApp() {
}

GuiApp::~GuiApp() {
    shutdown();
}

bool GuiApp::initialize() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // GL 3.3 + GLSL 330
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    m_window = glfwCreateWindow(800, 600, "EngineExport - ONNX to TensorRT Converter", NULL, NULL);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Custom styling
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 2.0f;
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Initialize plugins
    initializePlugins();
    
    addLog("EngineExport GUI initialized successfully");
    return true;
}

void GuiApp::run() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Process log messages from export thread
        processLogQueue();

        // Render main window
        renderMainWindow();
        
        // Render plugin selection window if opened
        if (m_showPluginWindow) {
            renderPluginSelection();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(m_window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(m_window);
    }
}

void GuiApp::shutdown() {
    // Wait for export thread to finish
    if (m_exportThread && m_exportThread->joinable()) {
        m_exportThread->join();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

void GuiApp::renderMainWindow() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | 
                                   ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
    
    ImGui::Begin("EngineExport", nullptr, window_flags);

    // Header
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 10));
    ImGui::Text("EngineExport - ONNX to TensorRT Engine Converter");
    ImGui::Separator();
    ImGui::PopStyleVar();

    // File selection section
    if (ImGui::CollapsingHeader("File Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderFileSelection();
    }

    ImGui::Spacing();

    // Export options section
    if (ImGui::CollapsingHeader("Export Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderExportOptions();
    }

    ImGui::Spacing();

    // Export button and progress
    renderExportButton();
    renderProgressBar();

    ImGui::Spacing();

    // Log window
    if (ImGui::CollapsingHeader("Log Output", ImGuiTreeNodeFlags_DefaultOpen)) {
        renderLogWindow();
    }

    ImGui::End();
}

void GuiApp::renderFileSelection() {
    // Input ONNX file
    ImGui::Text("Input ONNX File:");
    ImGui::PushItemWidth(-100);
    ImGui::InputText("##InputPath", m_inputPath, sizeof(m_inputPath));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Browse##Input")) {
        std::string path;
        if (openFileDialog(path, "ONNX Files\0*.onnx\0All Files\0*.*\0")) {
            strncpy_s(m_inputPath, path.c_str(), sizeof(m_inputPath) - 1);
            std::string outputPath = generateOutputPath();
            strncpy_s(m_outputPath, outputPath.c_str(), sizeof(m_outputPath) - 1);
        }
    }

    ImGui::Spacing();

    // Output engine file
    ImGui::Text("Output Engine File:");
    ImGui::PushItemWidth(-100);
    ImGui::InputText("##OutputPath", m_outputPath, sizeof(m_outputPath));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Browse##Output")) {
        std::string path;
        if (saveFileDialog(path, "Engine Files\0*.engine\0All Files\0*.*\0")) {
            strncpy_s(m_outputPath, path.c_str(), sizeof(m_outputPath) - 1);
        }
    }
}

void GuiApp::renderExportOptions() {
    // Resolution selection
    ImGui::Text("Input Resolution:");
    ImGui::SameLine();
    helpMarker("The input resolution for the model (e.g., 640 for 640x640)");
    
    const char* resolution_items[] = { "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544", "576", "608", "640", "672", "704", "736", "768", "800", "832", "864", "896", "928", "960", "992", "1024", "1280" };
    const int resolution_values[] = { 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1280 };
    int current_resolution_index = 16; // default to 640
    for (int i = 0; i < IM_ARRAYSIZE(resolution_values); i++) {
        if (m_resolution == resolution_values[i]) {
            current_resolution_index = i;
            break;
        }
    }

    if (ImGui::Combo("##Resolution", &current_resolution_index, resolution_items, IM_ARRAYSIZE(resolution_items))) {
        m_resolution = resolution_values[current_resolution_index];
        if (strlen(m_inputPath) > 0) {
            std::string outputPath = generateOutputPath();
            strncpy_s(m_outputPath, outputPath.c_str(), sizeof(m_outputPath) - 1);
        }
    }

    ImGui::Spacing();

    // Precision options
    ImGui::Text("Precision Options:");
    ImGui::Checkbox("Enable FP16", &m_enableFp16);
    ImGui::SameLine();
    helpMarker("Enable FP16 precision for better performance on supported GPUs");
    
    ImGui::Checkbox("Enable FP8", &m_enableFp8);
    ImGui::SameLine();
    helpMarker("Enable FP8 precision (experimental, requires Ada Lovelace or newer)");
    
    ImGui::Checkbox("Enable INT8", &m_enableInt8);
    ImGui::SameLine();
    helpMarker("Enable INT8 quantization (fastest but may reduce accuracy)");

    // QAT (Quantization Aware Training) toggle
    // If ONNX has QuantizeLinear/DequantizeLinear (Q/DQ), you can enable INT8 without calibration data
    ImGui::Indent();
    ImGui::Checkbox("Assume QAT (ONNX has Q/DQ)", &m_assumeQat);
    ImGui::SameLine();
    helpMarker("If your ONNX contains QuantizeLinear/DequantizeLinear nodes, enable this to build INT8 without a dataset.");
    ImGui::Unindent();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Advanced Optimization Settings
    if (ImGui::CollapsingHeader("Advanced Optimization Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        
        // Performance Flags
        ImGui::Text("Performance Optimization:");
        
        ImGui::Checkbox("TF32", &m_enableTf32);
        ImGui::SameLine();
        helpMarker("Use TF32 for faster computation on Ampere GPUs (RTX 30xx+)");
        
        ImGui::Checkbox("Sparse Weights", &m_enableSparseWeights);
        ImGui::SameLine();
        helpMarker("Enable sparse weight optimization for Ampere GPUs");
        
        ImGui::Checkbox("Direct I/O", &m_enableDirectIO);
        ImGui::SameLine();
        helpMarker("Minimize memory copies for faster execution");
        
        ImGui::Checkbox("REFIT Engine", &m_enableRefit);
        ImGui::SameLine();
        helpMarker("Create refittable engine (allows weight updates without rebuild)");
        
        ImGui::Checkbox("Disable Timing Cache", &m_disableTimingCache);
        ImGui::SameLine();
        helpMarker("Disable timing cache for faster build (may affect kernel selection)");
        
        ImGui::Text("Optimization Level:");
        ImGui::SliderInt("##OptLevel", &m_optimizationLevel, 1, 5, "Level %d");
        helpMarker("Higher levels = more aggressive optimization (5 = maximum)");
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Tactic Sources
        ImGui::Text("Tactic Sources (Kernel Selection):");
        
        ImGui::Checkbox("CUBLAS", &m_useCublas);
        ImGui::SameLine(150);
        ImGui::Checkbox("CUBLAS LT", &m_useCublasLt);
        
        ImGui::Checkbox("CUDNN", &m_useCudnn);
        ImGui::SameLine(150);
        ImGui::Checkbox("Edge Mask Conv", &m_useEdgeMaskConv);
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Other Settings
        ImGui::Text("Other Settings:");
        
        ImGui::Checkbox("GPU Fallback", &m_enableGpuFallback);
        ImGui::SameLine();
        helpMarker("Allow GPU fallback for unsupported operations");
        
        ImGui::Checkbox("Precision Constraints", &m_enablePrecisionConstraints);
        ImGui::SameLine();
        helpMarker("Prefer precision constraints (may reduce speed)");
        
        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Workspace size
    ImGui::Text("Workspace Size:");
    ImGui::SliderInt("MB##Workspace", &m_workspaceMb, 256, 4096, "%d MB");
    helpMarker("Amount of GPU memory to use for TensorRT workspace");

    ImGui::Spacing();
    
    // NMS Settings
    ImGui::Text("NMS Settings:");
    ImGui::Checkbox("Fix NMS Output Size", &m_fixNmsOutput);
    ImGui::SameLine();
    helpMarker("Fix NMS output to constant size for CUDA Graph compatibility");
    
    if (m_fixNmsOutput) {
        ImGui::Indent();
        ImGui::Text("Max Detections:");
        ImGui::SameLine();
        ImGui::InputInt("##MaxDetections", &m_nmsMaxDetections, 50, 100);
        if (m_nmsMaxDetections < 100) m_nmsMaxDetections = 100;
        if (m_nmsMaxDetections > 1000) m_nmsMaxDetections = 1000;
        helpMarker("Maximum number of detections (100-1000, default: 300)");
        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Verbose output
    ImGui::Checkbox("Verbose Output", &m_verbose);
    ImGui::SameLine();
    helpMarker("Enable detailed logging during conversion");
    
    ImGui::Spacing();
    
    // Plugin selection button
    if (ImGui::Button("Select Plugins", ImVec2(150, 30))) {
        m_showPluginWindow = true;
    }
    ImGui::SameLine();
    helpMarker("Configure TensorRT plugins for your model");
    
    // Show selected plugin count
    int enabledBuiltInCount = 0;
    for (const auto& plugin : m_availablePlugins) {
        if (plugin.enabled) enabledBuiltInCount++;
    }
    
    int enabledCustomCount = 0;
    for (const auto& plugin : m_customPlugins) {
        if (plugin.enabled) enabledCustomCount++;
    }
    
    int totalEnabled = enabledBuiltInCount + enabledCustomCount;
    
    ImGui::SameLine();
    if (totalEnabled > 0) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "(%d plugins selected)", totalEnabled);
        if (ImGui::IsItemHovered() && (enabledBuiltInCount > 0 || enabledCustomCount > 0)) {
            ImGui::BeginTooltip();
            ImGui::Text("Built-in: %d", enabledBuiltInCount);
            ImGui::Text("Custom: %d", enabledCustomCount);
            ImGui::EndTooltip();
        }
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no plugins selected)");
    }
}

void GuiApp::renderExportButton() {
    bool canExport = m_exportStatus == ExportStatus::IDLE && validateInputs();
    
    if (!canExport) {
        ImGui::BeginDisabled();
    }
    
    if (ImGui::Button("Start Export", ImVec2(150, 40))) {
        startExport();
    }
    
    if (!canExport) {
        ImGui::EndDisabled();
    }
    
    if (m_exportStatus == ExportStatus::RUNNING) {
        ImGui::SameLine();
        ImGui::Text("Converting...");
    } else if (m_exportStatus == ExportStatus::COMPLETED) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Conversion Completed!");
    } else if (m_exportStatus == ExportStatus::FAILED) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Conversion Failed");
    }
}

void GuiApp::renderProgressBar() {
    if (m_exportStatus == ExportStatus::RUNNING) {
        ImGui::ProgressBar(m_exportProgress.load(), ImVec2(-1.0f, 0.0f));
    }
}

void GuiApp::renderLogWindow() {
    static std::vector<LogEntry> displayLogs;
    
    // Process new log entries
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        while (!m_logQueue.empty()) {
            displayLogs.push_back(m_logQueue.front());
            m_logQueue.pop();
        }
    }
    
    // Limit log entries to prevent memory growth
    if (displayLogs.size() > 1000) {
        displayLogs.erase(displayLogs.begin(), displayLogs.begin() + 500);
    }
    
    ImGui::BeginChild("LogWindow", ImVec2(0, 200), true);
    
    for (const auto& log : displayLogs) {
        ImVec4 color = log.isError ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        ImGui::TextColored(color, "%s", log.message.c_str());
    }
    
    // Auto-scroll to bottom
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
    }
    
    ImGui::EndChild();
}

#ifdef _WIN32
bool GuiApp::openFileDialog(std::string& path, const char* filter) {
    OPENFILENAME ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = glfwGetWin32Window(m_window);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = filter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileName(&ofn)) {
        path = szFile;
        return true;
    }
    return false;
}

bool GuiApp::saveFileDialog(std::string& path, const char* filter) {
    OPENFILENAME ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = glfwGetWin32Window(m_window);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = filter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetSaveFileName(&ofn)) {
        path = szFile;
        return true;
    }
    return false;
}
#else
bool GuiApp::openFileDialog(std::string& path, const char* filter) {
    // For non-Windows platforms, would need to implement native dialogs
    // or use a cross-platform library
    return false;
}

bool GuiApp::saveFileDialog(std::string& path, const char* filter) {
    return false;
}
#endif

void GuiApp::startExport() {
    if (m_exportThread && m_exportThread->joinable()) {
        m_exportThread->join();
    }
    
    m_exportStatus = ExportStatus::RUNNING;
    m_exportProgress = 0.0f;
    m_exportError.clear();
    
    m_exportThread = std::make_unique<std::thread>(&GuiApp::exportThreadFunc, this);
}

void GuiApp::exportThreadFunc() {
    try {
        // Create config from GUI settings
        ExportConfig config;
        config.input_onnx_path = std::string(m_inputPath);
        config.output_engine_path = std::string(m_outputPath);
        config.input_resolution = m_resolution;
        config.enable_fp16 = m_enableFp16;
        config.enable_fp8 = m_enableFp8;
        config.enable_int8 = m_enableInt8;
        config.assume_qat_quantized = m_assumeQat;
        config.workspace_mb = m_workspaceMb;
        config.verbose = m_verbose;
        config.fix_nms_output = m_fixNmsOutput;
        config.nms_max_detections = m_nmsMaxDetections;
        
        // Advanced optimization settings
        config.enable_tf32 = m_enableTf32;
        config.enable_sparse_weights = m_enableSparseWeights;
        config.enable_direct_io = m_enableDirectIO;
        config.enable_refit = m_enableRefit;
        config.disable_timing_cache = m_disableTimingCache;
        config.optimization_level = m_optimizationLevel;
        
        // Tactic sources
        config.use_cublas = m_useCublas;
        config.use_cublas_lt = m_useCublasLt;
        config.use_cudnn = m_useCudnn;
        config.use_edge_mask_conv = m_useEdgeMaskConv;
        
        // Other settings
        config.enable_gpu_fallback = m_enableGpuFallback;
        config.enable_precision_constraints = m_enablePrecisionConstraints;
        
        // Add selected plugins to config
        config.selected_plugins.clear();
        int enabledPluginCount = 0;
        
        // Add built-in plugins
        for (const auto& plugin : m_availablePlugins) {
            if (plugin.enabled) {
                config.selected_plugins.insert(plugin.name);
                enabledPluginCount++;
            }
        }
        
        // Add custom plugins (use library path as identifier)
        for (const auto& plugin : m_customPlugins) {
            if (plugin.enabled) {
                config.selected_plugins.insert(plugin.libraryPath);
                enabledPluginCount++;
            }
        }
        
        if (enabledPluginCount > 0) {
            addLog("Using " + std::to_string(enabledPluginCount) + " selected plugins");
            for (const auto& plugin : m_availablePlugins) {
                if (plugin.enabled) {
                    addLog("  - Built-in: " + plugin.name);
                }
            }
            for (const auto& plugin : m_customPlugins) {
                if (plugin.enabled) {
                    addLog("  - Custom: " + plugin.name + " (" + plugin.libraryPath + ")");
                }
            }
        }
        
        addLog("Starting engine export...");
        addLog("Input: " + config.input_onnx_path);
        addLog("Output: " + config.get_output_path());
        
        m_exportProgress = 0.1f;
        
        // Create and run exporter
        EngineExporter exporter(config);
        
        m_exportProgress = 0.2f;
        
        bool success = exporter.exportEngine();
        
        if (success) {
            m_exportProgress = 1.0f;
            m_exportStatus = ExportStatus::COMPLETED;
            addLog("Engine export completed successfully!");
        } else {
            m_exportStatus = ExportStatus::FAILED;
            addLog("Engine export failed", true);
        }
        
    } catch (const std::exception& e) {
        m_exportStatus = ExportStatus::FAILED;
        addLog("Exception during export: " + std::string(e.what()), true);
    } catch (...) {
        m_exportStatus = ExportStatus::FAILED;
        addLog("Unknown error during export", true);
    }
}

void GuiApp::addLog(const std::string& message, bool isError) {
    std::lock_guard<std::mutex> lock(m_logMutex);
    m_logQueue.emplace(message, isError);
}

void GuiApp::processLogQueue() {
    // This is called from the main thread to update the UI
    // The actual log processing happens in renderLogWindow()
}

bool GuiApp::validateInputs() {
    if (strlen(m_inputPath) == 0) return false;
    if (!fileExists(std::string(m_inputPath))) return false;
    if (strlen(m_outputPath) == 0) return false;
    return true;
}

std::string GuiApp::generateOutputPath() {
    if (strlen(m_inputPath) == 0) return "";
    
    std::filesystem::path inputPath(m_inputPath);
    std::string baseName = inputPath.stem().string();
    
    // Add resolution and precision suffixes
    baseName += "_" + std::to_string(m_resolution);
    if (m_enableFp16) baseName += "_fp16";
    if (m_enableFp8) baseName += "_fp8";
    
    std::filesystem::path outputPath = inputPath.parent_path() / (baseName + ".engine");
    return outputPath.string();
}

void GuiApp::helpMarker(const char* desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

bool GuiApp::fileExists(const std::string& path) {
    return std::filesystem::exists(path);
}

void GuiApp::initializePlugins() {
    m_availablePlugins = PluginManager::getAvailablePlugins();
    addLog("Loaded " + std::to_string(m_availablePlugins.size()) + " available plugins");
}

void GuiApp::renderPluginSelection() {
    ImGui::SetNextWindowSize(ImVec2(700, 600), ImGuiCond_FirstUseEver);
    
    if (ImGui::Begin("Plugin Selection", &m_showPluginWindow)) {
        ImGui::Text("Select TensorRT plugins to enable for your model:");
        ImGui::Separator();
        ImGui::Spacing();
        
        // Create tabs for built-in and custom plugins
        if (ImGui::BeginTabBar("PluginTabs")) {
            // Built-in plugins tab
            if (ImGui::BeginTabItem("Built-in Plugins")) {
                if (ImGui::BeginChild("BuiltInPluginList", ImVec2(0, -80), true)) {
                    for (auto& plugin : m_availablePlugins) {
                        ImGui::PushID(&plugin);
                        
                        bool changed = ImGui::Checkbox(plugin.name.c_str(), &plugin.enabled);
                        if (changed) {
                            updateSelectedPlugins();
                            std::string status = plugin.enabled ? "enabled" : "disabled";
                            addLog("Plugin " + plugin.name + " " + status);
                        }
                        
                        ImGui::SameLine();
                        ImGui::TextDisabled("(?)");
                        if (ImGui::IsItemHovered()) {
                            ImGui::BeginTooltip();
                            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                            ImGui::TextUnformatted(plugin.description.c_str());
                            ImGui::PopTextWrapPos();
                            ImGui::EndTooltip();
                        }
                        
                        ImGui::PopID();
                    }
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            
            // Custom plugins tab
            if (ImGui::BeginTabItem("Custom Plugins")) {
                if (ImGui::BeginChild("CustomPluginList", ImVec2(0, -80), true)) {
                    if (m_customPlugins.empty()) {
                        ImGui::TextDisabled("No custom plugins added yet.");
                        ImGui::Spacing();
                        ImGui::Text("Click 'Add Plugin' to add a custom TensorRT plugin library.");
                    } else {
                        for (size_t i = 0; i < m_customPlugins.size(); ++i) {
                            auto& plugin = m_customPlugins[i];
                            ImGui::PushID(i);
                            
                            bool changed = ImGui::Checkbox(plugin.name.c_str(), &plugin.enabled);
                            if (changed) {
                                updateSelectedPlugins();
                                std::string status = plugin.enabled ? "enabled" : "disabled";
                                addLog("Custom plugin " + plugin.name + " " + status);
                            }
                            
                            // Plugin info
                            ImGui::SameLine();
                            ImGui::TextDisabled("(?)");
                            if (ImGui::IsItemHovered()) {
                                ImGui::BeginTooltip();
                                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                                ImGui::Text("Path: %s", plugin.libraryPath.c_str());
                                if (!plugin.description.empty()) {
                                    ImGui::Text("Description: %s", plugin.description.c_str());
                                }
                                ImGui::PopTextWrapPos();
                                ImGui::EndTooltip();
                            }
                            
                            // Delete button
                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
                            if (ImGui::SmallButton("Delete")) {
                                removeCustomPlugin(i);
                                ImGui::PopStyleColor();
                                ImGui::PopID();
                                break; // Exit loop since we modified the vector
                            }
                            ImGui::PopStyleColor();
                            
                            ImGui::PopID();
                        }
                    }
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            
            ImGui::EndTabBar();
        }
        
        ImGui::Separator();
        
        // Bottom buttons
        if (ImGui::Button("Add Plugin", ImVec2(90, 30))) {
            m_showAddPluginDialog = true;
            // Clear the form
            memset(m_newPluginName, 0, sizeof(m_newPluginName));
            memset(m_newPluginPath, 0, sizeof(m_newPluginPath));
            memset(m_newPluginDesc, 0, sizeof(m_newPluginDesc));
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Select All", ImVec2(80, 30))) {
            for (auto& plugin : m_availablePlugins) {
                plugin.enabled = true;
            }
            for (auto& plugin : m_customPlugins) {
                plugin.enabled = true;
            }
            updateSelectedPlugins();
            addLog("All plugins enabled");
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Clear All", ImVec2(80, 30))) {
            for (auto& plugin : m_availablePlugins) {
                plugin.enabled = false;
            }
            for (auto& plugin : m_customPlugins) {
                plugin.enabled = false;
            }
            updateSelectedPlugins();
            addLog("All plugins disabled");
        }
        
        ImGui::SameLine();
        ImGui::Dummy(ImVec2(150, 0)); // spacing
        
        ImGui::SameLine();
        if (ImGui::Button("Close", ImVec2(80, 30))) {
            m_showPluginWindow = false;
        }
        
        // Show plugin count
        int enabledBuiltInCount = 0;
        for (const auto& plugin : m_availablePlugins) {
            if (plugin.enabled) enabledBuiltInCount++;
        }
        
        int enabledCustomCount = 0;
        for (const auto& plugin : m_customPlugins) {
            if (plugin.enabled) enabledCustomCount++;
        }
        
        int totalBuiltIn = (int)m_availablePlugins.size();
        int totalCustom = (int)m_customPlugins.size();
        int totalEnabled = enabledBuiltInCount + enabledCustomCount;
        int totalPlugins = totalBuiltIn + totalCustom;
        
        ImGui::SameLine();
        ImGui::Text("(%d/%d enabled)", totalEnabled, totalPlugins);
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Built-in: %d/%d", enabledBuiltInCount, totalBuiltIn);
            ImGui::Text("Custom: %d/%d", enabledCustomCount, totalCustom);
            ImGui::EndTooltip();
        }
    }
    ImGui::End();
    
    // Render add plugin dialog if shown
    if (m_showAddPluginDialog) {
        renderAddPluginDialog();
    }
}

void GuiApp::renderAddPluginDialog() {
    ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    
    if (ImGui::Begin("Add Custom Plugin", &m_showAddPluginDialog, ImGuiWindowFlags_Modal)) {
        ImGui::Text("Add a custom TensorRT plugin library:");
        ImGui::Separator();
        ImGui::Spacing();
        
        // Plugin name
        ImGui::Text("Plugin Name:");
        ImGui::InputText("##PluginName", m_newPluginName, sizeof(m_newPluginName));
        helpMarker("A descriptive name for your plugin");
        
        ImGui::Spacing();
        
        // Plugin library path
        ImGui::Text("Library Path:");
        ImGui::PushItemWidth(-100);
        ImGui::InputText("##PluginPath", m_newPluginPath, sizeof(m_newPluginPath));
        ImGui::PopItemWidth();
        ImGui::SameLine();
        
        if (ImGui::Button("Browse##Plugin")) {
            std::string path;
            if (openFileDialog(path, "Dynamic Libraries\0*.dll;*.so;*.dylib\0All Files\0*.*\0")) {
                strncpy_s(m_newPluginPath, path.c_str(), sizeof(m_newPluginPath) - 1);
            }
        }
        helpMarker("Path to the plugin library file (.dll, .so, or .dylib)");
        
        ImGui::Spacing();
        
        // Plugin description (optional)
        ImGui::Text("Description (optional):");
        ImGui::InputTextMultiline("##PluginDesc", m_newPluginDesc, sizeof(m_newPluginDesc), ImVec2(0, 60));
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Validation and buttons
        bool isValid = strlen(m_newPluginName) > 0 && strlen(m_newPluginPath) > 0;
        bool pathExists = isValid && validatePluginPath(std::string(m_newPluginPath));
        
        if (!isValid) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Please provide both name and path.");
        } else if (!pathExists) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Plugin file does not exist.");
        } else {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Ready to add plugin.");
        }
        
        ImGui::Spacing();
        
        // Buttons
        if (!isValid || !pathExists) {
            ImGui::BeginDisabled();
        }
        
        if (ImGui::Button("Add Plugin", ImVec2(100, 30))) {
            addCustomPlugin(
                std::string(m_newPluginName),
                std::string(m_newPluginPath),
                std::string(m_newPluginDesc)
            );
            m_showAddPluginDialog = false;
        }
        
        if (!isValid || !pathExists) {
            ImGui::EndDisabled();
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 30))) {
            m_showAddPluginDialog = false;
        }
    }
    ImGui::End();
}

void GuiApp::updateSelectedPlugins() {
    // Update the configuration with selected plugins
    // This will be used when creating the ExportConfig
}

void GuiApp::addCustomPlugin(const std::string& name, const std::string& path, const std::string& description) {
    // Check if plugin with same name already exists
    for (const auto& plugin : m_customPlugins) {
        if (plugin.name == name) {
            addLog("Plugin with name '" + name + "' already exists", true);
            return;
        }
    }
    
    CustomPluginInfo plugin(name, path, description);
    m_customPlugins.push_back(plugin);
    addLog("Added custom plugin: " + name);
    addLog("Path: " + path);
}

void GuiApp::removeCustomPlugin(size_t index) {
    if (index < m_customPlugins.size()) {
        std::string name = m_customPlugins[index].name;
        m_customPlugins.erase(m_customPlugins.begin() + index);
        addLog("Removed custom plugin: " + name);
        updateSelectedPlugins();
    }
}

bool GuiApp::validatePluginPath(const std::string& path) {
    return fileExists(path);
}
