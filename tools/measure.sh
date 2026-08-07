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
#   off      : 프리셋 재적용으로 원상복구
#
# 프레임레이트를 함께 봐야 하므로 두 모드 다 perf 로그를 켠다. 레이트는 **평균
# 프레임간격**으로 읽을 것 - 중앙값은 긴 꼬리를 무시해서 16% 낙관적으로 나온다.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
DEV=inference_pc/build/bin/Release
CFG="$DEV/simple_config.json"

usage() { echo "usage: $0 {deadtime|noise|off} [프리셋(기본 160)]"; exit 1; }
[ $# -ge 1 ] || usage
P="${2:-160}"

case "$1" in
  deadtime|noise)
    STEP=12; [ "$1" = "noise" ] && STEP=0
    [ -f "$CFG" ] || { echo "dev 설정 없음: $CFG  (tools/preset.sh $P --dev 먼저)"; exit 1; }
    python3 - "$CFG" "$STEP" "$1" <<'PY'
import json, sys
p, step, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(p))
d["calibration_step_px"] = step
d["calibration_log_path"] = "calib_%s.csv" % mode
d["perf_stats_enabled"] = True
d["perf_log_truncate_on_start"] = True
json.dump(d, open(p, "w"), indent=4, ensure_ascii=False)
print("  측정 모드 '%s' 켬  ->  %s" % (mode, d["calibration_log_path"]))
PY
    echo
    if [ "$1" = "deadtime" ]; then
      echo "  1) 앱 재시작 (설정은 시작 시에만 읽음)"
      echo "  2) 정지 표적을 화면에 두고 30초 - 조준 버튼 누르지 말 것"
      echo "  3) tools/measure.sh off $P"
      echo "  4) python3 bench/deadtime_fit.py $DEV/calib_deadtime.csv"
    else
      echo "  1) 앱 재시작"
      echo "  2) 30초 기록 - aim ON 조건이면 조준 버튼을 계속 누르고 있을 것"
      echo "  3) tools/measure.sh off $P"
    fi
    ;;
  off)
    tools/preset.sh "$P" --dev
    echo "  측정 모드 해제 (진단 키까지 프리셋 값으로 복구)"
    ;;
  *) usage;;
esac
