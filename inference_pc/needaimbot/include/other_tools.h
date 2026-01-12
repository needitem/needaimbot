#ifndef OTHER_TOOLS_H
#define OTHER_TOOLS_H

#include <string>
#include <vector>

bool fileExists(const std::string& path);
std::string replace_extension(const std::string& filename, const std::string& new_extension);
std::vector<std::string> getEngineFiles();
std::vector<std::string> getModelFiles();
std::vector<std::string> getOnnxFiles();
std::vector<std::string>::difference_type getModelIndex(std::vector<std::string> engine_models);
std::vector<std::string> getAvailableModels();

#endif 
