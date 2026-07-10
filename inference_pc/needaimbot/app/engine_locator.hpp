#pragma once

// Startup path resolution: finds the repo root (dev checkout heuristic), picks
// the config file location next to the executable, and resolves the configured
// TensorRT .engine path - all pure filesystem logic, no app state.

#include <algorithm>
#include <array>
#include <filesystem>
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

// Resolves the configured engine path only. An absolute path is used as-is; a
// relative path is resolved against a fixed search order (cwd, config dir, repo
// root(s), exe dir) so existing relative configs keep working. There is no
// scanning or heuristic selection of unspecified engines - if the configured
// file is not found, this returns nullopt and the caller errors out.
inline std::optional<std::filesystem::path> resolveEnginePath(
    const std::string& configuredPath,
    const std::filesystem::path& configPath,
    const std::filesystem::path& exeDir,
    const std::optional<std::filesystem::path>& repoRoot) {
    const std::filesystem::path enginePath(configuredPath);
    std::vector<std::filesystem::path> candidates;

    if (enginePath.is_absolute()) {
        addUniquePath(candidates, enginePath);
    } else {
        std::vector<std::filesystem::path> searchDirs;
        addUniquePath(searchDirs, std::filesystem::current_path());
        addUniquePath(searchDirs, configPath.parent_path());
        if (repoRoot) {
            addUniquePath(searchDirs, *repoRoot);
            addUniquePath(searchDirs, *repoRoot / "inference_pc");
        }
        addUniquePath(searchDirs, exeDir);
        for (const auto& dir : searchDirs) {
            addUniquePath(candidates, dir / enginePath);
        }
    }

    for (const auto& candidate : candidates) {
        if (fileExists(candidate)) return candidate;
    }
    return std::nullopt;
}

}  // namespace engine_locator
