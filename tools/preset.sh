#!/bin/bash
# 캡처 해상도 프리셋 전환.
#
# movement_scale = 캡처크기 / 320 이라, 캡처를 바꾸면 model 좌표계의 오차·속도가
# 같이 스케일된다. 그 공간의 절대값과 비교되는 파라미터(softness, lead 게이트)를
# 함께 바꾸지 않으면 표적 근처에서 과도하게 공격적이 되어 흔들린다.
# 그래서 캡처 크기와 게인은 반드시 세트로 움직인다 = 이 스크립트의 존재 이유.
#
#   320 : 시야 넓음(화면 가로 16.7%). 표적이 크게 움직이거나 원거리 교전이 많은 게임.
#   160 : 시야 좁음(8.3%) 대신 검출 정밀도 높음 (실측 sigma X -34% / Y -23%).
#
# 주의: game_pc 쪽 CaptureWidth/Height 도 같이 바꿔야 한다. 한쪽만 바꾸면 어긋난다.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
STABLE=inference_pc/build_stable/bin/Release
DEV=inference_pc/build/bin/Release

usage() { echo "usage: $0 {320|160|show} [--dev]"; exit 1; }
[ $# -ge 1 ] || usage
P="$1"; shift
TGT="$STABLE"; NAME="stable"
[ "${1:-}" = "--dev" ] && { TGT="$DEV"; NAME="dev(build)"; }

case "$P" in
  show)
    for d in "$STABLE" "$DEV"; do
      [ -f "$d/simple_config.json" ] || continue
      python3 - "$d/simple_config.json" "$d" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
cap=d.get('pre_capture_shapes',[[0,0]])[0][0]
print("  %-46s 캡처 %s  softness %s/%s  conf %s  body_aim %s"%(
    sys.argv[2], cap, d.get('aim_softness_x'), d.get('aim_softness_y'),
    d.get('conf_threshold'), d.get('body_aim_point')))
PY
    done
    ;;
  320|160)
    SRC="$STABLE/simple_config.$P.json"
    [ -f "$SRC" ] || { echo "프리셋 없음: $SRC"; exit 1; }
    cp "$SRC" "$TGT/simple_config.json"
    echo "  $NAME -> ${P} 프리셋 적용"
    python3 - "$TGT/simple_config.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for k in ('pre_capture_shapes','conf_threshold','body_aim_point',
          'aim_softness_x','aim_softness_y','lead_vgate','lead_err_gate'):
    print("    %-22s %s"%(k,d.get(k)))
PY
    echo
    echo "  다음에 할 것:"
    echo "    1) game_pc config.ini : CaptureWidth=$P / CaptureHeight=$P  (필수)"
    echo "    2) game_pc 스트리머 재시작"
    echo "    3) 앱 재시작 (config 는 시작 시에만 읽음)"
    ;;
  *) usage;;
esac
