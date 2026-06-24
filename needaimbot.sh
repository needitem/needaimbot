#!/bin/bash
# ── tmux 운영 명령 (어디서든 실행) ───────────────────────────────
#   깔끔 재시작 : tmux kill-session -t start 2>/dev/null; tmux new-session -d -s start ~/portfolio/inferencetool/needaimbot.sh
#   접속(attach): tmux attach -t start        # 떼기: Ctrl-b 누른 뒤 d
#   로그만 보기 : tmux capture-pane -t start -p | tail
#   종료        : tmux send-keys -t start C-c && tmux kill-session -t start
# ────────────────────────────────────────────────────────────────
cd "$(dirname "$(readlink -f "$0")")/inference_pc"  # 스크립트 위치 기준 — 폴더 이동에 안전
exec ./build/bin/Release/simple_inference
