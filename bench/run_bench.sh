#!/bin/bash
# End-to-end bottleneck measurement on the inference PC (this Orin).
# 1) launch simple_inference with stage timing + force-aim
# 2) blast synthetic 256x256 frames over loopback UDP
# 3) capture the perf status line, then plot the stage breakdown
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root: inference_pc/..
ROOT="$(pwd)"
BIN="$ROOT/inference_pc/build/bin/Release/simple_inference"
CFG="$ROOT/bench/bench_config.json"
LOG="$ROOT/bench/run.log"
SENDER="$ROOT/tools/udp_frame_load_test.py"
DUR="${1:-30}"     # measurement seconds
FPS="${2:-200}"    # synthetic frame rate (saturate the pipeline)

echo "[bench] binary : $BIN"
echo "[bench] config : $CFG"
echo "[bench] dur=${DUR}s fps=${FPS}"

: > "$LOG"
stdbuf -oL "$BIN" "$CFG" > "$LOG" 2>&1 &
APP=$!
echo "[bench] simple_inference PID $APP, warming up..."
sleep 6
if ! kill -0 "$APP" 2>/dev/null; then
  echo "[bench] ERROR: app exited during startup. Tail:"; tail -30 "$LOG"; exit 1
fi

echo "[bench] starting UDP sender -> 127.0.0.1:5007 for ${DUR}s"
python3 "$SENDER" sender --host 127.0.0.1 --port 5007 \
        --width 256 --height 256 --fps "$FPS" --duration "$DUR" \
        > "$ROOT/bench/sender.log" 2>&1 &
SND=$!
wait "$SND"
sleep 1

echo "[bench] stopping app"
kill -INT "$APP" 2>/dev/null
for i in $(seq 1 20); do kill -0 "$APP" 2>/dev/null || break; sleep 0.3; done
kill -9 "$APP" 2>/dev/null

echo "[bench] sender log:"; tail -4 "$ROOT/bench/sender.log"
echo "[bench] plotting"
python3 "$ROOT/bench/plot_bottleneck.py" "$LOG" "$ROOT/bench/bottleneck.png"
