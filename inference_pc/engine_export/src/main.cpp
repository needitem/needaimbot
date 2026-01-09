#include <iostream>
#include <exception>
#include "gui_app.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
    try {
        // Check for console mode (if arguments provided)
        if (argc > 1) {
            // For now, just show message about GUI mode
            std::cout << "EngineExport now runs in GUI mode.\n";
            std::cout << "Please run without arguments to open the GUI interface.\n";
            return 0;
        }

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