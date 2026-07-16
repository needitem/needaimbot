#!/bin/bash
# ── 실행 / 종료 (STABLE 백업 빌드) ────────────────────────────────
#   실행        : ./needaimbot_stable.sh      # 이 터미널에서 포그라운드 실행
#   종료        : Ctrl-C                      # 클럭 자동 복원됨
#   백그라운드  : nohup ./needaimbot_stable.sh >/tmp/needaimbot_stable.log 2>&1 &
#                 종료 시  →  pkill -INT -f simple_inference_stable
#
#   build_stable/ 의 검증된 백업 바이너리(simple_inference_stable)를 실행한다.
#   config도 그 바이너리 옆의 것(build_stable/bin/Release/simple_config.json)을
#   사용하므로 개발 빌드의 config/기능 변경에 영향받지 않는다.
# ────────────────────────────────────────────────────────────────
cd "$(dirname "$(readlink -f "$0")")/inference_pc"  # 스크립트 위치 기준 — 폴더 이동에 안전

# 세션 동안만 GPU/CPU 클럭을 최대로 고정 (지터 제거).
# 종료(정상/Ctrl-C) 시 원래 DVFS 상태로 되돌린다.
# store 파일은 needaimbot.sh 와 공유한다: 두 스크립트를 번갈아 실행해도
# '고정 전' DVFS 상태가 한 곳에 보존되어야 복원이 올바르다.
CLOCK_STORE="/tmp/l4t_dfs_needaimbot.conf"

INFER_PID=""
restore_clocks() {
    # 자식(추론 바이너리)이 살아있으면 먼저 정리 후 클럭 복원.
    if [ -n "$INFER_PID" ] && kill -0 "$INFER_PID" 2>/dev/null; then
        kill -INT "$INFER_PID" 2>/dev/null
        wait "$INFER_PID" 2>/dev/null
    fi
    if [ -f "$CLOCK_STORE" ]; then
        sudo jetson_clocks --restore "$CLOCK_STORE" \
            && echo "[needaimbot-stable] jetson_clocks 복원됨 (DVFS 재활성화)"
    fi
}
# EXIT: 정상 종료. INT/TERM/HUP: Ctrl-C, kill, 터미널 종료 등 어떤 종료에도 복원.
trap restore_clocks EXIT
trap 'exit 130' INT TERM HUP

echo "[needaimbot-stable] jetson_clocks 고정 중..."
# 고정 전 상태를 복원용으로 저장. 이미 파일이 있으면 그게 '고정 전' 상태이므로
# 덮어쓰지 않는다(덮어쓰면 이전 세션이 비정상 종료된 경우 '고정된' 상태가 저장돼
# 복원이 무의미해짐 + 대화형 y/n 프롬프트로 무인 실행이 멈춤).
if [ ! -f "$CLOCK_STORE" ]; then
    sudo jetson_clocks --store "$CLOCK_STORE"
fi
sudo jetson_clocks                          # 최대 클럭으로 고정
echo "[needaimbot-stable] jetson_clocks 고정 완료"

# 백그라운드 실행 + wait: 시그널이 스크립트에 전달되어 trap이 돌게 한다.
# (포그라운드 exec이면 종료 후 복원 코드가 실행되지 않음)
./build_stable/bin/Release/simple_inference_stable "$@" &
INFER_PID=$!
wait "$INFER_PID"
