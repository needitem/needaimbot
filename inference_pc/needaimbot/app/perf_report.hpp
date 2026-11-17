#pragma once

// PerfLogWriter: owns the performance-statistics log file, its header, size-
// based rotation, and batched flushing.  Extracted from main()'s local
// variables/lambdas.  Header-only, following the existing needaimbot/app/
// convention.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "config.hpp"
#include "pipeline_metrics.hpp"  // formatWallClock

class PerfLogWriter {
public:
    PerfLogWriter(const Config& cfg, const std::filesystem::path& configPath)
        : maxBytes_(cfg.perfLogMaxBytes > 0 ? static_cast<size_t>(cfg.perfLogMaxBytes) : 0)
        , enginePath_(cfg.enginePath)
        , intervalMs_(cfg.perfStatsIntervalMs)
        , lastFlush_(std::chrono::steady_clock::now())
    {
        if (!cfg.perfStatsEnabled || cfg.perfLogPath.empty()) return;

        std::filesystem::path logPath(cfg.perfLogPath);
        if (logPath.is_relative()) logPath = configPath.parent_path() / logPath;
        logPath_ = logPath.lexically_normal();

        const auto openMode = std::ios::out |
            (cfg.perfLogTruncateOnStart ? std::ios::trunc : std::ios::app);
        log_.open(logPath_.string(), openMode);
        if (log_.is_open()) {
            std::error_code ec;
            if (!cfg.perfLogTruncateOnStart) {
                const auto existing = std::filesystem::file_size(logPath_, ec);
                if (!ec) bytesWritten_ = static_cast<size_t>(existing);
            }
            const std::string hdr = buildHeader();
            log_ << hdr;
            bytesWritten_ += hdr.size();
            log_.flush();
            std::cout << "[Simple] Perf log -> " << logPath_.string()
                      << (maxBytes_ ? " (rotate at " +
                            std::to_string(maxBytes_ / (1024 * 1024)) + "MiB, 1 backup)"
                                    : std::string(" (no rotation)"))
                      << std::endl;
        } else {
            std::cerr << "[Simple] WARN: could not open perf log at " << logPath_.string()
                      << " (metrics will not be recorded)" << std::endl;
        }
    }

    bool isOpen() const { return log_.is_open(); }

    void flush() { if (log_.is_open()) log_.flush(); }

    void writeLine(const std::string& line) {
        log_ << line << '\n';
        bytesWritten_ += line.size() + 1;
    }

    // Size-based rotation and batched flush.  Call once per stat window.
    void maybeFlush(std::chrono::steady_clock::time_point now) {
        if (maxBytes_ && bytesWritten_ >= maxBytes_) {
            log_.close();
            std::error_code ec;
            std::filesystem::rename(logPath_,
                std::filesystem::path(logPath_.string() + ".1"), ec);
            log_.open(logPath_.string(), std::ios::out | std::ios::trunc);
            bytesWritten_ = 0;
            if (log_.is_open()) {
                const std::string hdr = buildHeader();
                log_ << hdr;
                bytesWritten_ += hdr.size();
            }
            log_.flush();
            lastFlush_ = now;
        } else if (now - lastFlush_ >= std::chrono::seconds(5)) {
            // Batch flushes: 1 line/s doesn't need a syscall per write.
            log_.flush();
            lastFlush_ = now;
        }
    }

private:
    std::string buildHeader() const {
        std::ostringstream h;
        h << "# session " << formatWallClock()
          << " engine=" << enginePath_
          << " interval=" << intervalMs_ << "ms"
          << " | fields: R/S/D=recv/submit/done per s, B=busyDrop F=submitFail"
          << " C/U=udp recv/drop, Cr=credit I=inflight, Aw=acquire ms(avg/max)"
          << " Awp=acquire p50/95/99 ms, Su=submit us(avg/max), Cb=callback ms(avg/max)"
          << " Cbp=callback p50/95/99 ms,"
          << " E2E=capture->complete ms(avg/max) E2Ep=p50/95/99 ms E2En=samples/window"
          << " (needs NTP-synced clocks; excludes mouse), NF/IF/MD=timeouts/invalid/movedrop,"
          << " G=graph/std/fallback, Stn=timed samples StDrop=untimeable (excluded),"
          << " St=[h2d pre inf post d2h] us(avg/max)\n";
        return h.str();
    }

    std::ofstream log_;
    std::filesystem::path logPath_;
    size_t bytesWritten_ = 0;
    size_t maxBytes_;
    std::chrono::steady_clock::time_point lastFlush_;
    std::string enginePath_;
    int intervalMs_;
};
