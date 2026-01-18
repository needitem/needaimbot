// EngineExport CLI - Command-line ONNX to TensorRT converter
#include "config.h"
#include "engine_exporter.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "=== EngineExport CLI ===\n";
    std::cout << "ONNX to TensorRT Engine Converter\n\n";

    // Parse command line arguments
    ExportConfig config = ConfigParser::parseCommandLine(argc, argv);

    if (!config.is_valid()) {
        return 1;
    }

    // Create exporter and run
    EngineExporter exporter(config);

    if (exporter.exportEngine()) {
        std::cout << "\nConversion successful!\n";
        return 0;
    } else {
        std::cerr << "\nConversion failed!\n";
        return 1;
    }
}
