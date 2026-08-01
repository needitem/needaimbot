#!/usr/bin/env python3
"""사격 중 조준점이 아래로 가는 원인 분해.

calib CSV 는 컨트롤러가 무엇을 보고(cx,cy,classId) 무엇을 내보냈는지(emit_dx,dy)
프레임 단위로 남긴다. "아래로 간다"가 셋 중 무엇인지 여기서 갈린다:

  A) 검출이 아래를 가리킨다      -> cy 가 사격 중 커짐 (조준점/박스 문제)
  B) 컨트롤러가 아래로 민다      -> emit_dy 합이 지속적으로 양수 (제어 문제)
  C) 반동을 정상 상쇄 중일 뿐    -> emit_dy 는 양수지만 cy 는 화면중앙 유지

사격 구간은 emit 활동이 급증하는 구간으로 추정한다(사격 플래그가 CSV 에 없음).
"""
import csv, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "calib_fire.csv"
rows = []
for r in csv.DictReader(open(path)):
    try:
        if int(r["hasTarget"]) != 1 or int(r["aiming"]) != 1:
            continue
        rows.append((int(r["t_us"]), int(r["classId"]), float(r["cx"]), float(r["cy"]),
                     float(r["h"]), int(r["emit_dx"]), int(r["emit_dy"])))
    except (ValueError, KeyError):
        pass
if len(rows) < 100:
    sys.exit("조준 중 샘플 부족 (%d) - 조준하며 사격한 로그가 필요합니다" % len(rows))

t = np.array([r[0] for r in rows]) / 1e6
cls = np.array([r[1] for r in rows])
cy = np.array([r[3] for r in rows])
edy = np.array([r[6] for r in rows], dtype=float)
SC = 160.0

print("조준 중 %d 프레임, %.1f초\n" % (len(rows), t[-1] - t[0]))
print("  전체 평균:  cy-중심 %+.2f model px   emit_dy 평균 %+.3f/frame   head비율 %.0f%%"
      % (cy.mean() - SC, edy.mean(), 100 * (cls == 1).mean()))
print()

# 연속 구간을 활동량으로 나눠 '사격 추정' 구간과 비교
win = 60
act = np.convolve(np.abs(edy), np.ones(win) / win, mode="same")
thr = np.percentile(act, 70)
hot, cold = act >= thr, act < thr
print("  %-22s %10s %10s" % ("", "고활동(사격추정)", "저활동"))
print("  " + "-" * 46)
for lbl, v in (("cy - 화면중심", cy - SC), ("emit_dy", edy),
               ("head 선택 비율%", (cls == 1) * 100.0), ("박스 높이", np.array([r[4] for r in rows]))):
    print("  %-22s %10.2f %10.2f" % (lbl, v[hot].mean(), v[cold].mean()))
print()
print("  판정:")
d_cy = (cy - SC)[hot].mean() - (cy - SC)[cold].mean()
d_edy = edy[hot].mean() - edy[cold].mean()
if d_cy > 1.0:
    print("   -> A) 검출이 아래를 가리킨다 (cy %+.2f). 조준점/박스 문제." % d_cy)
elif d_edy > 0.05 and abs(d_cy) < 1.0:
    print("   -> C) emit 은 아래로(%+.3f) 가지만 cy 는 유지 -> 반동을 정상 상쇄 중." % d_edy)
elif d_edy > 0.05:
    print("   -> B) 컨트롤러가 아래로 민다 (emit_dy %+.3f)." % d_edy)
else:
    print("   -> 사격 구간에서 하향 경향이 안 잡힘. 구간 추정이 빗나갔을 수 있음.")
