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
# 프리셋 원본은 추적되는 곳에 있다. build_stable/ 은 로컬 바이너리 때문에 통째로
# gitignore 라, 거기 두면 프리셋이 커밋에 안 실려 다른 설치 환경에 배포되지 않는다.
PRESETS=inference_pc/presets
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
      python3 - "$d/simple_config.json" "$d" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
cap=d.get('pre_capture_shapes',[[0,0]])[0][0]
print("  %-44s 캡처 %s  conf %s  body_aim %s"%(sys.argv[2],cap,
      d.get('conf_threshold'), d.get('body_aim_point')))
print("      우클릭 kp %s/%s  soft %s/%s  kd %s/%s  max_step %s"%(
      d.get('aim_kp_x'),d.get('aim_kp_y'),d.get('aim_softness_x'),
      d.get('aim_softness_y'),d.get('aim_kd_x'),d.get('aim_kd_y'),d.get('aim_max_step')))
print("      thumb   kp %s/%s  soft %s/%s  kd %s/%s"%(
      d.get('thumb_aim_kp_x'),d.get('thumb_aim_kp_y'),
      d.get('thumb_aim_softness_x'),d.get('thumb_aim_softness_y'),
      d.get('thumb_aim_kd_x'),d.get('thumb_aim_kd_y')))
print("      적응데드타임 %s  aim_h_ema %s  mincut %s  세로배율 %s"%(
      d.get('deadtime_adaptive'), d.get('aim_h_ema'), d.get('oneeuro_min_cutoff'),
      d.get('aim_y_scale')))
PYEOF
    done
    # 두 프리셋이 '캡처 종속 6개는 x2, 나머지는 동일' 규칙에서 벗어났는지 검사한다.
    # 한쪽만 튜닝하고 잊으면 preset.sh 한 번으로 그 튜닝이 조용히 되돌아간다.
    echo
    python3 - "$PRESETS" <<'PYEOF'
import json,sys,os
S=sys.argv[1]
try:
    a=json.load(open(os.path.join(S,"simple_config.320.json")))
    b=json.load(open(os.path.join(S,"simple_config.160.json")))
except FileNotFoundError:
    print("  프리셋 파일 없음 - 정합성 검사 건너뜀"); raise SystemExit
HALF={"aim_softness_x","aim_softness_y","thumb_aim_softness_x","thumb_aim_softness_y",
      "lead_vgate","lead_err_gate"}
OWN={"pre_capture_shapes","conf_threshold","head_aim_point","body_aim_point"}
bad=[]
for k in sorted(set(a)|set(b)):
    if k in OWN or k.startswith("_"): continue
    x,y=a.get(k),b.get(k)
    if k in HALF:
        try:
            if abs(y-2*x)>1e-6: bad.append("%s: 320=%s 160=%s (x2 아님)"%(k,x,y))
        except TypeError: bad.append("%s: 숫자가 아님"%k)
    elif x!=y:
        bad.append("%s: 320=%s 160=%s"%(k,x,y))
if bad:
    print("  [경고] 프리셋 불일치 %d건 - 한쪽만 갱신됐을 수 있다:"%len(bad))
    for t in bad[:12]: print("      "+t)
    sys.exit(1)
print("  프리셋 정합성 OK (캡처 종속 6개는 x2, 나머지 동일)")
PYEOF
    # 표는 데이터의 사본이므로 데이터에서 다시 만든다. 사람이 기억해야 하는 방식은
    # 이미 실패했다 - 이 README 의 표는 한 세대 뒤처져 있었다.
    python3 tools/gen_preset_readme.py --presets "$PRESETS"
    ;;

  320|160)
    SRC="$PRESETS/simple_config.$P.json"
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
    python3 tools/gen_preset_readme.py --presets "$PRESETS" >/dev/null
    echo
    echo "  다음에 할 것:"
    echo "    1) game_pc config.ini : CaptureWidth=$P / CaptureHeight=$P  (필수)"
    echo "    2) game_pc 스트리머 재시작"
    echo "    3) 앱 재시작 (config 는 시작 시에만 읽음)"
    ;;
  *) usage;;
esac
