#!/bin/bash
# 리그 측정 모드를 켜고 끈다.
#
# 왜 스크립트인가: 측정할 때마다 dev 설정의 진단 키를 손으로 켜고 껐는데, 되돌리는
# 걸 두 번 놓쳤다. 한 번은 calibration_step_px 가 켜진 채 남아 마우스가 주기적으로
# 튀었고, 한 번은 perf 로그가 계속 쌓였다. 끄기는 프리셋 재적용과 같은 동작이므로
# (진단 키가 프리셋에 들어 있다) 여기서 preset.sh 를 그대로 부른다.
#
#   deadtime : aim OFF 로 +-12 카운트를 주기 주입해 emit->visible 을 잰다.
#              **정지 표적**, 조준 버튼 누르지 말 것. 30초면 충분하다.
#              -> bench/deadtime_fit.py 로 적합
#   noise    : 주입 없이 검출만 기록한다. aim ON 이면 조준 버튼을 계속 누르고 있을 것
#              (안 누르면 컨트롤러가 안 돌아 aim-ON 조건이 아니다).
#              -> bench/sigma_law.py 계열 / 온라인 sigma 추정 검증
#   play     : noise 와 같은 설정. 실제 교전에서 그냥 평소대로 플레이하면 된다.
#              튜닝이 실제로 나은지 보는 유일한 방법이고, 실전 프레임레이트도 여기서만
#              나온다(지금까지 잰 206fps 는 정지·단순 장면 값이다).
#   off      : 프리셋 재적용으로 원상복구
#
# 로그 스위치는 둘이다. `calibration_log_enabled` = 검출/출력 추적(calib CSV),
# `perf_stats_enabled` = 파이프라인 단계 타이밍(perf 로그). 프레임 타임스탬프는 calib
# CSV 에 있으므로 **실측 fps 는 calib 만으로 나온다** - play 가 perf 를 안 켜는 이유다.
# 레이트는 **평균 프레임간격**으로 읽을 것 - 중앙값은 긴 꼬리를 무시해서 16% 낙관적이다.
#
# 로깅 비용은 사실상 없다: 기록은 미리 예약된 버퍼에 push_back 뿐이고(할당·I/O 없음,
# 종료 시 한 번에 덤프), perf 계측은 CPU 타임스탬프만 쓴다(GPU 이벤트·동기화 없음).
# **다만 버퍼가 20만 행**이라 206fps 에서 16분이면 찬다. 넘으면 앱이 경고를 찍고 그
# 뒤로는 기록하지 않는다 - 세션을 15분 이내로 끊을 것.
#
# 로그 경로는 **절대 경로**로 박는다. 상대 경로는 앱의 작업 디렉터리 기준인데
# needaimbot.sh 가 거기를 inference_pc/ 로 옮기므로 바이너리 옆이 아니다. 이걸 착각해
# 없는 파일을 분석하려 한 적이 있다. dev 설정은 gitignore 라 절대 경로를 넣어도 된다.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
ROOT="$PWD"
DEV=inference_pc/build/bin/Release
CFG="$DEV/simple_config.json"
OUT="$ROOT/inference_pc/measure"

usage() { echo "usage: $0 {deadtime|noise|play|off} [프리셋(기본 160)]"; exit 1; }
[ $# -ge 1 ] || usage
P="${2:-160}"

case "$1" in
  deadtime|noise|play)
    STEP=0; [ "$1" = "deadtime" ] && STEP=12
    [ -f "$CFG" ] || { echo "dev 설정 없음: $CFG  (tools/preset.sh $P --dev 먼저)"; exit 1; }
    mkdir -p "$OUT"
    # 앱은 종료할 때 이 경로를 그냥 덮어쓴다. 실제로 플레이 수집 하나를 테스트 실행으로
    # 날렸다. 무장할 때 기존 파일을 옆으로 치워 둔다 - 지우는 건 사람이 판단할 일이다.
    PREV="$OUT/calib_$1.csv"
    if [ -f "$PREV" ]; then
      n=1; while [ -f "$PREV.$n" ]; do n=$((n+1)); done
      mv "$PREV" "$PREV.$n"; echo "  이전 수집 보관: $(basename "$PREV").$n"
    fi
    python3 - "$CFG" "$STEP" "$1" "$OUT" <<'PY'
import json, os, sys
p, step, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
d = json.load(open(p))
# 두 로그는 다른 질문에 답한다. calib CSV = 검출기가 뭘 봤고 우리가 뭘 내보냈나
# (프레임 타임스탬프도 여기 있어서 실측 fps 가 나온다). perf 로그 = 우리 파이프라인
# 어디서 시간을 쓰나. play 는 앞엣것만 필요하다.
d["calibration_log_enabled"] = True
d["perf_stats_enabled"] = (mode != "play")
d["calibration_step_px"] = step
d["calibration_log_path"] = os.path.join(out, "calib_%s.csv" % mode)
d["perf_log_path"] = os.path.join(out, "perf_%s.log" % mode)
d["perf_log_truncate_on_start"] = True
json.dump(d, open(p, "w"), indent=4, ensure_ascii=False)
print("  측정 모드 '%s' 켬" % mode)
print("    calib  %s" % d["calibration_log_path"])
print("    perf   %s" % (d["perf_log_path"] if d["perf_stats_enabled"] else "(끔 - play 는 calib 만)"))
PY
    echo
    case "$1" in
      deadtime)
        echo "  1) 앱 재시작 (설정은 시작 시에만 읽음)"
        echo "  2) 정지 표적을 화면에 두고 30초 - 조준 버튼 누르지 말 것"
        echo "  3) tools/measure.sh off $P"
        echo "  4) python3 bench/deadtime_fit.py $OUT/calib_deadtime.csv" ;;
      noise)
        echo "  1) 앱 재시작"
        echo "  2) 30초 기록 - aim ON 조건이면 조준 버튼을 계속 누르고 있을 것"
        echo "  3) tools/measure.sh off $P"
        echo "  4) python3 bench/calibrate.py $OUT/calib_noise.csv" ;;
      play)
        echo "  1) 앱 재시작"
        echo "  2) 평소대로 플레이 - **15분 이내** (버퍼 20만 행)"
        echo "  3) tools/measure.sh off $P"
        echo "  4) python3 bench/calibrate.py $OUT/calib_play.csv"
        echo
        echo "  * 주입은 꺼져 있으니 마우스가 제멋대로 튀지 않는다."
        echo "  * 체감도 같이 봐줄 것 - 숫자가 못 보는 걸 본다." ;;
    esac
    ;;
  off)
    tools/preset.sh "$P" --dev
    echo "  측정 모드 해제 (진단 키까지 프리셋 값으로 복구)"
    ;;
  *) usage;;
esac
