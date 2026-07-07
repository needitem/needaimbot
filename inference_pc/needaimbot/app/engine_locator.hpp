#pragma once

// Startup path resolution: finds the repo root (dev checkout heuristic), picks
// the config file location next to the executable, and locates/scores the
// TensorRT .engine file to load - all pure filesystem logic, no app state.

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace engine_locator {

inline bool fileExists(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

inline std::filesystem::path safeAbsolute(const std::filesystem::path& path) {
    std::error_code ec;
    auto absolutePath = std::filesystem::absolute(path, ec);
    return ec ? path : absolutePath.lexically_normal();
}

inline std::filesystem::path currentExecutablePath(const char* argv0) {
#ifndef _WIN32
    std::array<char, 4096> buffer{};
    const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length > 0) {
        buffer[static_cast<size_t>(length)] = '\0';
        return safeAbsolute(std::filesystem::path(buffer.data()));
    }
#endif
    return argv0 ? safeAbsolute(std::filesystem::path(argv0)) : std::filesystem::path();
}

inline std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

inline void addUniquePath(std::vector<std::filesystem::path>& paths, const std::filesystem::path& path) {
    if (path.empty()) return;
    const auto normalized = safeAbsolute(path);
    if (std::find(paths.begin(), paths.end(), normalized) == paths.end()) {
        paths.push_back(normalized);
    }
}

inline std::optional<std::filesystem::path> findRepoRootFrom(std::filesystem::path start) {
    if (start.empty()) return std::nullopt;
    start = safeAbsolute(start);
    if (fileExists(start)) {
        start = start.parent_path();
    }

    for (auto current = start; !current.empty(); current = current.parent_path()) {
        if (fileExists(current / "inference_pc" / "simple_main.cpp") &&
            fileExists(current / "inference_pc" / "CMakeLists.txt")) {
            return current;
        }
        if (current == current.root_path()) break;
    }
    return std::nullopt;
}

inline std::optional<std::filesystem::path> findRepoRoot(const std::filesystem::path& exeDir) {
    if (auto root = findRepoRootFrom(std::filesystem::current_path())) {
        return root;
    }
    return findRepoRootFrom(exeDir);
}

inline std::filesystem::path chooseConfigPath(const std::filesystem::path& exeDir) {
    return exeDir / "simple_config.json";
}

// Heuristic preference score for picking among multiple .engine files found in
// a search directory (prefers the newest/best-matched build tag by filename).
inline int engineScore(const std::filesystem::path& path) {
    const std::string name = toLower(path.filename().string());
    int score = 0;
    if (name.find("0.8.2") != std::string::npos) score += 80;
    if (name.find("256") != std::string::npos) score += 45;
    if (name.find("aux0") != std::string::npos) score += 5;
    if (name.find("320") != std::string::npos) score += 30;
    if (name.find("fp16io") != std::string::npos) score += 35;
    if (name.find("orin") != std::string::npos) score += 25;
    if (name.find("l5") != std::string::npos) score += 8;
    if (name.find("fp16") != std::string::npos) score += 20;
    if (name.find("trt") != std::string::npos) score += 10;
    if (name.find("dynamic") != std::string::npos) score -= 4;
    if (name.find("fp8") != std::string::npos) score -= 2;
    return score;
}

inline std::optional<std::filesystem::path> discoverEngine(
    const std::vector<std::filesystem::path>& searchDirs) {
    std::optional<std::filesystem::path> bestPath;
    int bestScore = std::numeric_limits<int>::min();

    for (const auto& dir : searchDirs) {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) break;
            const auto candidate = entry.path();
            if (!fileExists(candidate) || candidate.extension() != ".engine") continue;

            const int score = engineScore(candidate);
            if (!bestPath || score > bestScore ||
                (score == bestScore && candidate.filename().string() > bestPath->filename().string())) {
                bestPath = candidate;
                bestScore = score;
            }
        }
    }
    return bestPath;
}

// Resolves the configured engine path against a search order (cwd, config
// dir, repo root(s), exe dir); falls back to scanning those dirs for the
// best-scored *.engine file if the configured path isn't found directly.
inline std::optional<std::filesystem::path> resolveEnginePath(
    const std::string& configuredPath,
    const std::filesystem::path& configPath,
    const std::filesystem::path& exeDir,
    const std::optional<std::filesystem::path>& repoRoot) {
    const std::filesystem::path enginePath(configuredPath);
    std::vector<std::filesystem::path> candidates;
    std::vector<std::filesystem::path> searchDirs;

    addUniquePath(searchDirs, std::filesystem::current_path());
    addUniquePath(searchDirs, configPath.parent_path());
    if (repoRoot) {
        addUniquePath(searchDirs, *repoRoot);
        addUniquePath(searchDirs, *repoRoot / "inference_pc");
    }
    addUniquePath(searchDirs, exeDir);

    if (enginePath.is_absolute()) {
        addUniquePath(candidates, enginePath);
    } else {
        for (const auto& dir : searchDirs) {
            addUniquePath(candidates, dir / enginePath);
        }
    }

    for (const auto& candidate : candidates) {
        if (fileExists(candidate)) return candidate;
    }
    return discoverEngine(searchDirs);
}

}  // namespace engine_locator
