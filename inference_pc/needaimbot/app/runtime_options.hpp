#pragma once

// Command-line argument parsing for the simple_inference executable.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

struct RuntimeOptions {
    bool debugFrameDump = false;
    bool buttonTest = false;
    bool hasConfigPath = false;
    bool hasDebugDir = false;
    bool helpRequested = false;
    bool collectCalib = false;
    int collectCalibCount = 500;
    std::filesystem::path configPath;
    std::filesystem::path debugDir;
    std::filesystem::path collectCalibDir;
};

inline void printUsage(const char* exeName) {
    std::cout << "Usage: " << (exeName ? exeName : "simple_inference")
              << " [config.json] [--debug] [--debug-dir DIR] [--button-test]\n"
              << "  --debug              Save latest received RGB frame once per second\n"
              << "  --debug-dir DIR      Debug output directory (default: inference_pc/debug)\n"
              << "  --button-test        Print Makcu mouse button mask and exit with Ctrl+C\n"
              << "  --collect-calib DIR [N]  Save N received RGB frames to DIR for int8 calibration (default N=500)\n";
}

inline bool parseRuntimeOptions(int argc, char* argv[], RuntimeOptions& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--debug" || arg == "--debug-frame" || arg == "--debug-frames") {
            options.debugFrameDump = true;
        } else if (arg == "--button-test") {
            options.buttonTest = true;
        } else if (arg == "--collect-calib") {
            if (i + 1 >= argc) {
                std::cerr << "[Args] --collect-calib requires a directory" << std::endl;
                return false;
            }
            options.collectCalib = true;
            options.collectCalibDir = argv[++i];
            // Optional count: only consume the next arg if it is a positive integer.
            if (i + 1 < argc) {
                const std::string nxt = argv[i + 1] ? argv[i + 1] : "";
                if (!nxt.empty() && nxt.find_first_not_of("0123456789") == std::string::npos) {
                    options.collectCalibCount = std::max(1, std::atoi(nxt.c_str()));
                    ++i;
                }
            }
        } else if (arg == "--debug-dir") {
            if (i + 1 >= argc) {
                std::cerr << "[Args] --debug-dir requires a path" << std::endl;
                return false;
            }
            options.debugFrameDump = true;
            options.hasDebugDir = true;
            options.debugDir = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            options.helpRequested = true;
            return true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "[Args] Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        } else if (!options.hasConfigPath) {
            options.hasConfigPath = true;
            options.configPath = arg;
        } else {
            std::cerr << "[Args] Extra positional argument: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}
