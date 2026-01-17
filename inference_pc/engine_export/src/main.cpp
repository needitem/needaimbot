#include <iostream>
#include <exception>
#include "gui_app.h"
#include "config.h"
#include "engine_exporter.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
    try {
        // CLI mode: if arguments provided, run in command line mode
        if (argc > 1) {
            // Check for --gui flag to force GUI mode
            for (int i = 1; i < argc; ++i) {
                if (std::string(argv[i]) == "--gui") {
                    goto run_gui;
                }
            }

            // Parse command line arguments
            ExportConfig config = ConfigParser::parseCommandLine(argc, argv);

            // Check if help/version was requested (empty input path)
            if (config.input_onnx_path.empty()) {
                return 0;
            }

            // Validate config
            if (!config.is_valid()) {
                std::cerr << "Error: Invalid configuration. Use --help for usage.\n";
                return 1;
            }

            std::cout << "=== EngineExport CLI Mode ===\n";
            std::cout << "Input:  " << config.input_onnx_path << "\n";
            std::cout << "Output: " << config.get_output_path() << "\n";
            std::cout << "Resolution: " << config.input_resolution << "\n";
            std::cout << "FP16: " << (config.enable_fp16 ? "enabled" : "disabled") << "\n";
            std::cout << "FP8: " << (config.enable_fp8 ? "enabled" : "disabled") << "\n";
            std::cout << "Workspace: " << config.workspace_mb << " MB\n";
            std::cout << "\nBuilding engine...\n";

            // Run export
            EngineExporter exporter(config);
            if (!exporter.exportEngine()) {
                std::cerr << "Error: Engine export failed\n";
                return 1;
            }

            std::cout << "\nEngine exported successfully!\n";
            std::cout << "Output: " << config.get_output_path() << "\n";
            return 0;
        }

run_gui:
#ifdef _WIN32
        // Keep console window open for debugging
        // ShowWindow(GetConsoleWindow(), SW_HIDE);
#endif

        // Create and run GUI application
        GuiApp app;

        if (!app.initialize()) {
            std::cerr << "Failed to initialize application" << std::endl;
            return 1;
        }

        app.run();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}