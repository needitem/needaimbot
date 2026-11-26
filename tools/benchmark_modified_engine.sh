#!/usr/bin/env bash
#
# benchmark_modified_engine.sh - Build and benchmark modified ONNX vs original
#
# Builds a TensorRT engine from a modified ONNX model and benchmarks it
# against the original engine. Reports latency comparison.
#
# Usage:
#   ./tools/benchmark_modified_engine.sh --onnx <modified.onnx> --original <original.engine>
#   ./tools/benchmark_modified_engine.sh --onnx <modified.onnx> --original <original.engine> --int8
#   ./tools/benchmark_modified_engine.sh --onnx <modified.onnx> --original <original.engine> --precision fp16
#
# Options:
#   --onnx <path>       Modified ONNX model to build and benchmark
#   --original <path>   Original TensorRT engine for comparison
#   --precision <mode>  Build precision: fp16 (default), int8, fp32
#   --iterations <n>    Benchmark iterations (default: 200)
#   --skip-build        Skip engine build, use existing --modified engine
#   --modified <path>   Pre-built modified engine (with --skip-build)
#   --output-engine <p> Where to save the built engine
#   --help              Show this help
#

set -euo pipefail

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
ITERATIONS=200
WARMUP=2000
AVG_RUNS=10
BUILDER_OPT=5
WORKSPACE_MB=4096

ONNX_PATH=""
ORIGINAL_ENGINE=""
MODIFIED_ENGINE=""
PRECISION="fp16"
SKIP_BUILD=false
OUTPUT_ENGINE=""

# ─── Helpers ──────────────────────────────────────────────────────────────────

die() { printf "ERROR: %s\n" "$*" >&2; exit 1; }
log() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }

separator() { printf '%.0s─' {1..72}; printf '\n'; }

usage() {
    sed -n '2,/^$/{ s/^# \?//; p }' "$0"
    exit 0
}

# ─── Argument parsing ────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx)          ONNX_PATH="$2"; shift 2 ;;
        --original)      ORIGINAL_ENGINE="$2"; shift 2 ;;
        --modified)      MODIFIED_ENGINE="$2"; shift 2 ;;
        --precision)     PRECISION="$2"; shift 2 ;;
        --iterations)    ITERATIONS="$2"; shift 2 ;;
        --skip-build)    SKIP_BUILD=true; shift ;;
        --output-engine) OUTPUT_ENGINE="$2"; shift 2 ;;
        --help|-h)       usage ;;
        *)               die "Unknown option: $1" ;;
    esac
done

# Validate
[[ -x "$TRTEXEC" ]] || die "trtexec not found at $TRTEXEC"
[[ -n "$ORIGINAL_ENGINE" ]] || die "Missing --original <engine>"
[[ -f "$ORIGINAL_ENGINE" ]] || die "Original engine not found: $ORIGINAL_ENGINE"

if $SKIP_BUILD; then
    [[ -n "$MODIFIED_ENGINE" ]] || die "With --skip-build, provide --modified <engine>"
    [[ -f "$MODIFIED_ENGINE" ]] || die "Modified engine not found: $MODIFIED_ENGINE"
else
    [[ -n "$ONNX_PATH" ]] || die "Missing --onnx <path>"
    [[ -f "$ONNX_PATH" ]] || die "ONNX model not found: $ONNX_PATH"
fi

# ─── Build engine from modified ONNX ─────────────────────────────────────────

if ! $SKIP_BUILD; then
    if [[ -z "$OUTPUT_ENGINE" ]]; then
        base=$(basename "$ONNX_PATH" .onnx)
        engine_dir=$(dirname "$ONNX_PATH")
        OUTPUT_ENGINE="${engine_dir}/${base}_${PRECISION}.engine"
    fi
    MODIFIED_ENGINE="$OUTPUT_ENGINE"

    separator
    log "Building TensorRT engine from modified ONNX"
    log "  ONNX:      $ONNX_PATH"
    log "  Precision:  $PRECISION"
    log "  Output:     $OUTPUT_ENGINE"
    separator

    BUILD_FLAGS=("--onnx=$ONNX_PATH" "--saveEngine=$OUTPUT_ENGINE"
                 "--builderOptimizationLevel=$BUILDER_OPT"
                 "--memPoolSize=workspace:${WORKSPACE_MB}")

    case "$PRECISION" in
        fp16)  BUILD_FLAGS+=("--fp16") ;;
        int8)  BUILD_FLAGS+=("--int8" "--fp16") ;;
        fp32)  ;;  # no extra flag
        *)     die "Unknown precision: $PRECISION (use fp16, int8, or fp32)" ;;
    esac

    build_start=$(date +%s)
    "$TRTEXEC" "${BUILD_FLAGS[@]}" 2>&1 | tail -10
    build_end=$(date +%s)
    build_time=$((build_end - build_start))

    if [[ ! -f "$OUTPUT_ENGINE" ]]; then
        die "Engine build failed -- no output file produced"
    fi

    log "Engine built in ${build_time}s: $(du -h "$OUTPUT_ENGINE" | cut -f1)"
    echo ""
fi

# ─── Benchmark both engines ──────────────────────────────────────────────────

extract_metric() {
    local output="$1" pattern="$2"
    echo "$output" | grep -oP "$pattern" | head -1 || echo "N/A"
}

run_benchmark() {
    local engine="$1"
    local label="$2"

    log "Benchmarking: $label ($(basename "$engine"))"
    local trt_output
    trt_output=$("$TRTEXEC" \
        --loadEngine="$engine" \
        --iterations="$ITERATIONS" \
        --warmUp="$WARMUP" \
        --avgRuns="$AVG_RUNS" \
        --useSpinWait \
        2>&1) || { echo "FAILED"; return; }

    # Extract metrics
    local mean_host mean_gpu throughput p99
    # Host latency mean
    mean_host=$(echo "$trt_output" | grep -A2 'Host Latency' | grep -oP 'mean\s*=\s*\K[\d.]+' | head -1 || echo "N/A")
    # GPU compute mean
    mean_gpu=$(echo "$trt_output" | grep -A2 'GPU Compute Time' | grep -oP 'mean\s*=\s*\K[\d.]+' | head -1 || echo "N/A")
    # Throughput
    throughput=$(echo "$trt_output" | grep -oP 'Throughput:\s+\K[\d.]+' || echo "N/A")
    # P99
    p99=$(echo "$trt_output" | grep -oP '99%\s+\K[\d.]+' | head -1 || echo "N/A")

    echo "${mean_host}|${mean_gpu}|${p99}|${throughput}"
}

separator
echo ""
echo "  BENCHMARK COMPARISON"
echo "  Original: $(basename "$ORIGINAL_ENGINE")"
echo "  Modified: $(basename "$MODIFIED_ENGINE")"
echo "  Iterations: $ITERATIONS  |  Warmup: ${WARMUP}ms"
echo ""
separator

ORIG_RESULT=$(run_benchmark "$ORIGINAL_ENGINE" "Original")
MOD_RESULT=$(run_benchmark "$MODIFIED_ENGINE" "Modified")

echo ""
separator
printf "%-12s %10s %10s %10s %12s\n" "" "Host(ms)" "GPU(ms)" "P99(ms)" "QPS"
separator

IFS='|' read -r orig_host orig_gpu orig_p99 orig_qps <<< "$ORIG_RESULT"
IFS='|' read -r mod_host mod_gpu mod_p99 mod_qps <<< "$MOD_RESULT"

printf "%-12s %10s %10s %10s %12s\n" "Original" "$orig_host" "$orig_gpu" "$orig_p99" "$orig_qps"
printf "%-12s %10s %10s %10s %12s\n" "Modified" "$mod_host" "$mod_gpu" "$mod_p99" "$mod_qps"

# Compute delta if both are numeric
if [[ "$orig_host" != "N/A" && "$mod_host" != "N/A" && \
      "$orig_host" != "FAILED" && "$mod_host" != "FAILED" ]]; then
    echo ""
    separator
    echo "  LATENCY DELTA"
    separator

    host_delta=$(python3 -c "
o, m = $orig_host, $mod_host
d = m - o
pct = (d / o * 100) if o > 0 else 0
sign = '+' if d >= 0 else ''
print(f'  Host latency:  {sign}{d:.3f} ms ({sign}{pct:.1f}%)')
")
    echo "$host_delta"

    gpu_delta=$(python3 -c "
o, m = $orig_gpu, $mod_gpu
d = m - o
pct = (d / o * 100) if o > 0 else 0
sign = '+' if d >= 0 else ''
print(f'  GPU compute:   {sign}{d:.3f} ms ({sign}{pct:.1f}%)')
" 2>/dev/null || echo "  GPU compute:   N/A")
    echo "$gpu_delta"

    qps_delta=$(python3 -c "
o, m = $orig_qps, $mod_qps
d = m - o
pct = (d / o * 100) if o > 0 else 0
sign = '+' if d >= 0 else ''
print(f'  Throughput:    {sign}{d:.1f} qps ({sign}{pct:.1f}%)')
" 2>/dev/null || echo "  Throughput:    N/A")
    echo "$qps_delta"
fi

echo ""
separator
echo "  Engine sizes:"
echo "    Original: $(du -h "$ORIGINAL_ENGINE" | cut -f1)"
echo "    Modified: $(du -h "$MODIFIED_ENGINE" | cut -f1)"
separator

log "Done."
