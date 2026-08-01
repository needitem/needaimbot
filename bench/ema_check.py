#!/usr/bin/env python3
"""'KF-CA' 의 정체 확인: 사실은 고정 게인 EMA 인가?

ctrl_zoo.KFCACtrl 의 P 갱신은 0열만 건드리고 예측은 대각에만 Q 를 더한다. 그래서
P[1][0]=P[2][0]=0 이 영원히 유지되고 K=[K0,0,0] 이 된다 - 속도/가속 상태는 측정으로
절대 보정되지 않고 0 에 머문다. 남는 것은 위치에 대한 1차 재귀뿐이다:
    P- = P + 3Q,  K = P-/(P-+R),  P+ = P- * R/(P-+R)
이 P 는 몇 프레임 만에 고정점으로 수렴하므로 결국 alpha 고정 EMA 다.

맞다면 결론이 바뀐다: 이긴 것은 칼만도 CA 모델도 아니고 '적응형 One Euro 대신
고정 EMA' 라는 훨씬 단순하고 훨씬 이식하기 쉬운 변경이다. 여기서 (1) 등가성을 수치로
확인하고 (2) alpha 를 직접 최적화한다.
"""
import math

from realjit import ev, sc
from holdout import ev_holdout
from ctrl_zoo_opt import base
from ctrl_zoo import KFCACtrl
from aim_opt import OptCtrl

KFCA = {"kp_x": 0.71, "kp_y": 0.764, "soft_x": 10.993, "soft_y": 14.79,
        "kd_x": 0.1288, "kd_y": 0.0367, "max_step": 24.256, "w": 1.591,
        "ff": 1.6282, "predict": 2.6993, "v_ema": 0.2165, "vgate": 5.7821,
        "ff_err_gate": 27.14, "mincut": 0.106, "kf_R": 19.032, "kf_Q": 5.0879}


def steady_alpha(R, Q):
    """위 재귀의 고정점 K. A = P- 로 두면 A^2 - 3Q*A - 3Q*R = 0."""
    a = 3.0*Q
    A = (a + math.sqrt(a*a + 4.0*a*R)) / 2.0
    return A / (A + R)


class EMACtrl(OptCtrl):
    """One Euro 자리에 고정 alpha 1차 EMA."""
    def reset(self):
        OptCtrl.reset(self); self.ex = self.ey = None

    def step(self, rx, ry, cls_changed=False, box_h=None):
        p = self.p
        a = p.get("ema_a", 0.58)
        if self.ex is None:
            self.ex, self.ey = rx, ry
        else:
            self.ex += a*(rx - self.ex)
            self.ey += a*(ry - self.ey)
        saved = p["oneeuro"]; p["oneeuro"] = False
        self.filt_x, self.filt_y = self.ex, self.ey
        out = OptCtrl.step(self, self.ex, self.ey, cls_changed)
        p["oneeuro"] = saved
        return out


def line(lbl, p, ctrl, ref=None):
    m = ev_holdout(p, ctrl=ctrl)
    e = sc(m)
    d = "" if ref is None else "  %+6.1f%%" % (100*(e-ref)/ref)
    print("  %-34s %8.3f  %5.2f %6.2f %5.1f%s"
          % (lbl, e, m['step_overshoot'], m['rev_peak'], m['reach_reach'], d))
    return e


def main():
    al = steady_alpha(KFCA["kf_R"], KFCA["kf_Q"])
    print("  정상상태 alpha = %.4f  (R=%.2f, Q=%.4f)\n" % (al, KFCA["kf_R"], KFCA["kf_Q"]))
    print("  %-34s %8s  %5s %6s %5s"
          % ("구성 (holdout)", "에러", "오버슛", "반전pk", "reach"))
    print("  " + "-" * 68)
    ref = line("현재 배포 (One Euro)", base(), None)
    e_kf = line("KF-CA", base(**KFCA), KFCACtrl, ref)
    e_em = line("고정 EMA alpha=%.4f" % al, base(**dict(KFCA, ema_a=al)), EMACtrl, ref)
    print("     -> KF-CA 대비 차이 %.4f%%  %s"
          % (100*(e_em-e_kf)/e_kf,
             "(동일 = EMA 가 정체)" if abs(e_em-e_kf)/e_kf < 2e-3 else "(다름)"))

    print("\n  [alpha 직접 스윕 - 같은 게인]")
    print("  " + "-" * 68)
    for a in (0.35, 0.45, 0.50, 0.55, 0.58, 0.62, 0.70, 0.80, 1.00):
        line("alpha %.2f" % a, base(**dict(KFCA, ema_a=a)), EMACtrl, ref)

    print("\n  [현재 배포 게인 그대로, One Euro 만 EMA 로 교체]")
    print("  " + "-" * 68)
    for a in (0.40, 0.50, 0.58, 0.70, 0.85):
        line("현재게인 + EMA %.2f" % a, base(ema_a=a), EMACtrl, ref)


if __name__ == "__main__":
    main()
