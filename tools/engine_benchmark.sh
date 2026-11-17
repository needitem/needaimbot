#!/usr/bin/env bash
#
# engine_benchmark.sh — TensorRT engine benchmarking and optimization tool
# Target: Jetson AGX Orin (SM87, CUDA 12.5, TensorRT 10.3.0)
#
# Usage:
#   ./tools/engine_benchmark.sh [OPTIONS]
#
# Options:
#   --engine <path>       Benchmark a single engine file
#   --onnx-dir <path>     Directory containing ONNX models (default: engines/)
#   --rebuild-int8        Rebuild engines with INT8+FP16 and sparsity variants
#   --rebuild-cached      Rebuild engines using a persistent timing cache
#   --profile             Profile engines with nsys
#   --output <path>       Also write results to file (default: engines/benchmark_results.txt)
#   --no-output-file      Do not write results to a file
#   --iterations <n>      Benchmark iterations (default: 100)
#   --help                Show this help
#

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
DEFAULT_ENGINE_DIR="engines"
DEFAULT_ONNX_DIR="engines"
DEFAULT_ITERATIONS=100
DEFAULT_WARMUP=1000
DEFAULT_AVG_RUNS=10
BUILDER_OPT_LEVEL=5
WORKSPACE_MB=4096

# ─── Globals ──────────────────────────────────────────────────────────────────

ENGINE_DIR=""
ONNX_DIR=""
SINGLE_ENGINE=""
DO_REBUILD_INT8=false
DO_REBUILD_CACHED=false
DO_PROFILE=false
ITERATIONS=$DEFAULT_ITERATIONS
OUTPUT_FILE=""
NO_OUTPUT_FILE=false

# ─── Helpers ──────────────────────────────────────────────────────────────────

usage() {
    sed -n '2,/^$/{ s/^# \?//; p }' "$0"
    exit 0
}

die() {
    printf "ERROR: %s\n" "$*" >&2
    exit 1
}

log() {
    printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"
}

separator() {
    printf '%.0s─' {1..78}
    printf '\n'
}

cleanup() {
    log "Interrupted — exiting."
    exit 130
}
trap cleanup INT TERM

# Tee to both stdout and file when output file is set
out() {
    if [[ -n "$OUTPUT_FILE" ]]; then
        printf '%s\n' "$*" | tee -a "$OUTPUT_FILE"
    else
        printf '%s\n' "$*"
    fi
}

# ─── Argument parsing ────────────────────────────────────────────────────────

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --engine)
                SINGLE_ENGINE="$2"; shift 2 ;;
            --onnx-dir)
                ONNX_DIR="$2"; shift 2 ;;
            --rebuild-int8)
                DO_REBUILD_INT8=true; shift ;;
            --rebuild-cached)
                DO_REBUILD_CACHED=true; shift ;;
            --profile)
                DO_PROFILE=true; shift ;;
            --output)
                OUTPUT_FILE="$2"; shift 2 ;;
            --no-output-file)
                NO_OUTPUT_FILE=true; shift ;;
            --iterations)
                ITERATIONS="$2"; shift 2 ;;
            --help|-h)
                usage ;;
            *)
                die "Unknown option: $1" ;;
        esac
    done

    # Resolve directories relative to script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

    ENGINE_DIR="$PROJECT_DIR/$DEFAULT_ENGINE_DIR"
    [[ -z "$ONNX_DIR" ]] && ONNX_DIR="$PROJECT_DIR/$DEFAULT_ONNX_DIR"

    if $NO_OUTPUT_FILE; then
        OUTPUT_FILE=""
    elif [[ -z "$OUTPUT_FILE" ]]; then
        OUTPUT_FILE="$ENGINE_DIR/benchmark_results.txt"
    fi
}

# ─── Preflight checks ────────────────────────────────────────────────────────

preflight() {
    [[ -x "$TRTEXEC" ]] || die "trtexec not found at $TRTEXEC"
    [[ -d "$ENGINE_DIR" ]] || die "Engine directory not found: $ENGINE_DIR"

    if $DO_PROFILE; then
        command -v nsys >/dev/null 2>&1 || die "nsys (Nsight Systems) not found in PATH"
    fi
}

# ─── Part 1: Benchmark ───────────────────────────────────────────────────────

# Parse trtexec output and extract key metrics
extract_metrics() {
    local output="$1"

    local mean_total mean_gpu p50 p90 p99 throughput
    # Throughput
    throughput=$(echo "$output" | grep -oP 'Throughput:\s+\K[\d.]+' || echo "N/A")
    # Mean latency (host)
    mean_total=$(echo "$output" | grep -P '^\s*mean\b' | head -1 | awk '{print $4}' || echo "N/A")
    # GPU compute mean
    mean_gpu=$(echo "$output" | grep -P '^\s*mean\b' | tail -1 | awk '{print $4}' || echo "N/A")
    # Percentiles — from the "Host Latency" section
    p50=$(echo "$output" | grep -oP 'percentile:\s+[\d.]+\s+ms\s+\(end to end [\d.]+ ms\)' | sed -n '1p' | grep -oP '[\d.]+' | head -1 || echo "N/A")
    p90=$(echo "$output" | grep -oP 'percentile:\s+[\d.]+\s+ms\s+\(end to end [\d.]+ ms\)' | sed -n '2p' | grep -oP '[\d.]+' | head -1 || echo "N/A")
    p99=$(echo "$output" | grep -oP 'percentile:\s+[\d.]+\s+ms\s+\(end to end [\d.]+ ms\)' | sed -n '3p' | grep -oP '[\d.]+' | head -1 || echo "N/A")

    # Fallback: parse percentile lines differently if above pattern fails
    if [[ "$p50" == "N/A" ]]; then
        p50=$(echo "$output" | grep -P 'Percentile' | grep '50%' | grep -oP '[\d.]+(?=\s*ms)' | head -1 || echo "N/A")
    fi
    if [[ "$p90" == "N/A" ]]; then
        p90=$(echo "$output" | grep -P 'Percentile' | grep '90%' | grep -oP '[\d.]+(?=\s*ms)' | head -1 || echo "N/A")
    fi
    if [[ "$p99" == "N/A" ]]; then
        p99=$(echo "$output" | grep -P 'Percentile' | grep '99%' | grep -oP '[\d.]+(?=\s*ms)' | head -1 || echo "N/A")
    fi

    echo "$mean_total|$mean_gpu|$p50|$p90|$p99|$throughput"
}

benchmark_engine() {
    local engine="$1"
    local name
    name=$(basename "$engine")

    log "Benchmarking: $name"

    local trt_output
    trt_output=$("$TRTEXEC" \
        --loadEngine="$engine" \
        --iterations="$ITERATIONS" \
        --warmUp="$DEFAULT_WARMUP" \
        --avgRuns="$DEFAULT_AVG_RUNS" \
        --useSpinWait \
        2>&1) || {
        log "WARNING: trtexec failed for $name"
        echo "$name|FAILED|FAILED|FAILED|FAILED|FAILED|FAILED"
        return
    }

    local metrics
    metrics=$(extract_metrics "$trt_output")
    echo "$name|$metrics"
}

run_benchmarks() {
    local engines=()

    if [[ -n "$SINGLE_ENGINE" ]]; then
        [[ -f "$SINGLE_ENGINE" ]] || die "Engine not found: $SINGLE_ENGINE"
        engines+=("$SINGLE_ENGINE")
    else
        while IFS= read -r -d '' f; do
            engines+=("$f")
        done < <(find "$ENGINE_DIR" -maxdepth 1 -name '*.engine' -print0 | sort -z)
    fi

    if [[ ${#engines[@]} -eq 0 ]]; then
        log "No .engine files found in $ENGINE_DIR"
        return
    fi

    separator
    out "TensorRT Engine Benchmark Results"
    out "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    out "Device: $(cat /proc/device-tree/model 2>/dev/null || echo 'unknown')"
    out "Iterations: $ITERATIONS  |  Warm-up: ${DEFAULT_WARMUP}ms  |  Avg runs: $DEFAULT_AVG_RUNS"
    separator
    out ""

    # Header
    local hdr
    hdr=$(printf "%-40s %8s %8s %8s %8s %8s %10s" \
        "Engine" "Mean(ms)" "GPU(ms)" "P50(ms)" "P90(ms)" "P99(ms)" "QPS")
    out "$hdr"
    out "$(printf '%.0s─' {1..96})"

    local results=()
    for engine in "${engines[@]}"; do
        local row
        row=$(benchmark_engine "$engine")
        results+=("$row")

        # Parse and format
        IFS='|' read -r ename mean_total mean_gpu p50 p90 p99 throughput <<< "$row"
        local line
        line=$(printf "%-40s %8s %8s %8s %8s %8s %10s" \
            "$ename" "$mean_total" "$mean_gpu" "$p50" "$p90" "$p99" "$throughput")
        out "$line"
    done

    out ""
    separator
}

# ─── Part 2: Rebuild with INT8 ───────────────────────────────────────────────

rebuild_int8() {
    log "Rebuilding engines with INT8+FP16..."

    local onnx_files=()
    while IFS= read -r -d '' f; do
        onnx_files+=("$f")
    done < <(find "$ONNX_DIR" -maxdepth 1 -name '*.onnx' -print0 | sort -z)

    if [[ ${#onnx_files[@]} -eq 0 ]]; then
        die "No ONNX models found in $ONNX_DIR"
    fi

    for onnx in "${onnx_files[@]}"; do
        local base
        base=$(basename "$onnx" .onnx)

        # INT8 + FP16 build
        local int8_engine="$ENGINE_DIR/${base}_int8.engine"
        log "Building INT8 engine: $int8_engine"
        "$TRTEXEC" \
            --onnx="$onnx" \
            --int8 --fp16 \
            --builderOptimizationLevel=$BUILDER_OPT_LEVEL \
            --memPoolSize=workspace:$WORKSPACE_MB \
            --saveEngine="$int8_engine" \
            2>&1 | tail -5

        if [[ -f "$int8_engine" ]]; then
            log "Built: $int8_engine ($(du -h "$int8_engine" | cut -f1))"
        else
            log "WARNING: Failed to build $int8_engine"
        fi

        # INT8 + FP16 + Sparsity build
        local sparse_engine="$ENGINE_DIR/${base}_int8_sparse.engine"
        log "Building INT8+Sparse engine: $sparse_engine"
        "$TRTEXEC" \
            --onnx="$onnx" \
            --int8 --fp16 \
            --sparsity=enable \
            --builderOptimizationLevel=$BUILDER_OPT_LEVEL \
            --memPoolSize=workspace:$WORKSPACE_MB \
            --saveEngine="$sparse_engine" \
            2>&1 | tail -5

        if [[ -f "$sparse_engine" ]]; then
            log "Built: $sparse_engine ($(du -h "$sparse_engine" | cut -f1))"
        else
            log "WARNING: Failed to build $sparse_engine"
        fi
    done

    log "INT8 rebuild complete. Run without --rebuild-int8 to benchmark new engines."
}

# ─── Part 3: Rebuild with timing cache ───────────────────────────────────────

rebuild_cached() {
    local cache_file="$ENGINE_DIR/timing_cache.bin"
    log "Rebuilding engines with timing cache: $cache_file"

    local onnx_files=()
    while IFS= read -r -d '' f; do
        onnx_files+=("$f")
    done < <(find "$ONNX_DIR" -maxdepth 1 -name '*.onnx' -print0 | sort -z)

    if [[ ${#onnx_files[@]} -eq 0 ]]; then
        die "No ONNX models found in $ONNX_DIR"
    fi

    for onnx in "${onnx_files[@]}"; do
        local base
        base=$(basename "$onnx" .onnx)
        local cached_engine="$ENGINE_DIR/${base}_cached.engine"

        log "Building with timing cache: $cached_engine"
        "$TRTEXEC" \
            --onnx="$onnx" \
            --fp16 \
            --builderOptimizationLevel=$BUILDER_OPT_LEVEL \
            --memPoolSize=workspace:$WORKSPACE_MB \
            --timingCacheFile="$cache_file" \
            --saveEngine="$cached_engine" \
            2>&1 | tail -5

        if [[ -f "$cached_engine" ]]; then
            log "Built: $cached_engine ($(du -h "$cached_engine" | cut -f1))"
        else
            log "WARNING: Failed to build $cached_engine"
        fi
    done

    log "Timing cache rebuild complete. Cache saved to: $cache_file"
}

# ─── Part 4: Profile with nsys ───────────────────────────────────────────────

profile_engines() {
    log "Profiling engines with Nsight Systems..."

    local engines=()

    if [[ -n "$SINGLE_ENGINE" ]]; then
        [[ -f "$SINGLE_ENGINE" ]] || die "Engine not found: $SINGLE_ENGINE"
        engines+=("$SINGLE_ENGINE")
    else
        while IFS= read -r -d '' f; do
            engines+=("$f")
        done < <(find "$ENGINE_DIR" -maxdepth 1 -name '*.engine' -print0 | sort -z)
    fi

    if [[ ${#engines[@]} -eq 0 ]]; then
        log "No .engine files found for profiling."
        return
    fi

    for engine in "${engines[@]}"; do
        local name
        name=$(basename "$engine" .engine)
        local profile_output="$ENGINE_DIR/${name}_profile"

        log "Profiling: $name"
        nsys profile \
            --stats=true \
            --output="$profile_output" \
            --force-overwrite=true \
            "$TRTEXEC" \
            --loadEngine="$engine" \
            --iterations=50 \
            --warmUp=500 \
            2>&1 | tail -20

        if [[ -f "${profile_output}.nsys-rep" ]]; then
            log "Profile saved: ${profile_output}.nsys-rep"
        else
            log "WARNING: Profile output not found for $name"
        fi
    done

    log "Profiling complete. Open .nsys-rep files in Nsight Systems GUI for analysis."
}

# ─── Main ─────────────────────────────────────────────────────────────────────

main() {
    parse_args "$@"
    preflight

    # Clear output file if writing
    if [[ -n "$OUTPUT_FILE" ]]; then
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        : > "$OUTPUT_FILE"
        log "Results will also be saved to: $OUTPUT_FILE"
    fi

    # Always run benchmarks first
    run_benchmarks

    # Optional phases
    if $DO_REBUILD_INT8; then
        rebuild_int8
    fi

    if $DO_REBUILD_CACHED; then
        rebuild_cached
    fi

    if $DO_PROFILE; then
        profile_engines
    fi

    log "Done."
}

main "$@"
