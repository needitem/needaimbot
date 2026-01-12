#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <set>

#include "other_tools.h"
#include "config.h"
#include "AppContext.h"

bool fileExists(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::string replace_extension(const std::string& filename, const std::string& new_extension)
{
    size_t last_dot = filename.find_last_of(".");
    if (last_dot == std::string::npos)
    {
        return filename + new_extension;
    }
    else
    {
        return filename.substr(0, last_dot) + new_extension;
    }
}

// Ensure models directory exists
static void ensureModelsDirectory() {
    if (!std::filesystem::exists("models/")) {
        std::filesystem::create_directories("models/");
    }
}

std::vector<std::string> getEngineFiles()
{
    ensureModelsDirectory();
    std::vector<std::string> engineFiles;

    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            // Support both .engine and .cache (obfuscated) extensions
            if (ext == ".engine" || ext == ".cache")
            {
                engineFiles.push_back(entry.path().filename().string());
            }
        }
    }
    return engineFiles;
}

std::vector<std::string> getModelFiles()
{
    ensureModelsDirectory();
    std::vector<std::string> modelsFiles;

    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            // Support .engine, .cache (obfuscated), and .onnx
            if (ext == ".engine" || ext == ".cache" || ext == ".onnx")
            {
                modelsFiles.push_back(entry.path().filename().string());
            }
        }
    }
    return modelsFiles;
}

std::vector<std::string> getOnnxFiles()
{
    ensureModelsDirectory();
    std::vector<std::string> onnxFiles;

    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx")
        {
            onnxFiles.push_back(entry.path().filename().string());
        }
    }
    return onnxFiles;
}

std::vector<std::string>::difference_type getModelIndex(std::vector<std::string> engine_models)
{
    auto& config = AppContext::getInstance().config;
    auto it = std::find(engine_models.begin(), engine_models.end(), config.profile().ai_model);

    if (it != engine_models.end())
    {
        return std::distance(engine_models.begin(), it);
    }
    else
    {
        return 0;
    }
}

std::vector<std::string> getAvailableModels()
{
    std::vector<std::string> availableModels;
    std::vector<std::string> engineFiles = getEngineFiles();
    std::vector<std::string> onnxFiles = getOnnxFiles();

    std::set<std::string> engineModels;
    for (const auto& file : engineFiles)
    {
        engineModels.insert(std::filesystem::path(file).stem().string());
    }

    for (const auto& file : engineFiles)
    {
        availableModels.push_back(file);
    }

    for (const auto& file : onnxFiles)
    {
        std::string modelName = std::filesystem::path(file).stem().string();
        if (engineModels.find(modelName) == engineModels.end())
        {
            availableModels.push_back(file);
        }
    }

    return availableModels;
}
