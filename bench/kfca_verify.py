#!/usr/bin/env python3
"""KF-CA 이득이 이식할 만큼 견고한지 검증.

ctrl_zoo.KFCACtrl 의 공분산 갱신은 정식 칼만이 아니다: 예측에서 F P F^T 를 생략하고
대각에만 Q 를 더하며, 갱신에서 P 의 0열만 줄인다. 즉 이긴 것은 '칼만필터'가 아니라
'이 특정 재귀식'이다. 그대로 CUDA 로 옮길 것인지 판단하려면:
  1) 정식 CA 칼만은 더 나은가 더 나쁜가
  2) kf_R/kf_Q 가 조금 틀려도 이득이 남는가 (실기에서 정확히 못 맞출 것이므로)
  3) 데드타임 분포가 달라져도 이득이 남는가 (sim 의 가장 큰 모델오차)
"""
import math
import sys

import realjit
from realjit import ev, sc
from holdout import ev_holdout
from ctrl_zoo_opt import base
from ctrl_zoo import KFCACtrl
from aim_opt import OptCtrl

KFCA = {"kp_x": 0.71, "kp_y": 0.764, "soft_x": 10.993, "soft_y": 14.79,
        "kd_x": 0.1288, "kd_y": 0.0367, "max_step": 24.256, "w": 1.591,
        "ff": 1.6282, "predict": 2.6993, "v_ema": 0.2165, "vgate": 5.7821,
        "ff_err_gate": 27.14, "mincut": 0.106, "kf_R": 19.032, "kf_Q": 5.0879}


class ProperKF(OptCtrl):
    """정식 등가속 칼만: F P F^T + Q 예측, 전체 P 갱신. dt=1 프레임."""
    def reset(self):
        OptCtrl.reset(self); self.sx = None

    @staticmethod
    def _pred(P, Q):
        # F = [[1,1,.5],[0,1,1],[0,0,1]]
        F = ((1., 1., .5), (0., 1., 1.), (0., 0., 1.))
        FP = [[sum(F[i][k]*P[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        out = [[sum(FP[i][k]*F[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
        # 연속백색가속 잡음의 이산화 (q * [[1/20,1/8,1/6],[1/8,1/3,1/2],[1/6,1/2,1]])
        G = ((.05, .125, 1/6.), (.125, 1/3., .5), (1/6., .5, 1.))
        for i in range(3):
            for j in range(3):
                out[i][j] += Q*G[i][j]
        return out

    def _kf(self, s, P, z, R, Q):
        s[0] += s[1] + 0.5*s[2]; s[1] += s[2]
        P[:] = self._pred(P, Q)
        S = P[0][0] + R
        K = [P[i][0]/S for i in range(3)]
        r = z - s[0]
        for i in range(3):
            s[i] += K[i]*r
        newP = [[P[i][j] - K[i]*P[0][j] for j in range(3)] for i in range(3)]
        P[:] = newP
        return s[0]

    def step(self, rx, ry, cls_changed=False, box_h=None):
        p = self.p
        R = p.get("kf_R", 66.0); Q = p.get("kf_Q", 0.5)
        if self.sx is None:
            self.sx = [rx, 0., 0.]; self.sy = [ry, 0., 0.]
            self.Px = [[1e3 if i == j else 0. for j in range(3)] for i in range(3)]
            self.Py = [[1e3 if i == j else 0. for j in range(3)] for i in range(3)]
            fx, fy = rx, ry
        else:
            fx = self._kf(self.sx, self.Px, rx, R, Q)
            fy = self._kf(self.sy, self.Py, ry, R, Q)
        saved = p["oneeuro"]; p["oneeuro"] = False
        self.filt_x, self.filt_y = fx, fy
        out = OptCtrl.step(self, fx, fy, cls_changed)
        p["oneeuro"] = saved
        return out


def line(lbl, p, ctrl, ref=None):
    m = ev_holdout(p, ctrl=ctrl)
    e = sc(m)
    d = "" if ref is None else "  %+6.1f%%" % (100*(e-ref)/ref)
    print("  %-40s %8.3f  %5.2f %6.2f %5.1f%s"
          % (lbl, e, m['step_overshoot'], m['rev_peak'], m['reach_reach'], d))
    return e


def main():
    cur = base()
    print("  %-40s %8s  %5s %6s %5s"
          % ("구성 (holdout seed 200-239)", "에러", "오버슛", "반전pk", "reach"))
    print("  " + "-" * 74)
    ref = line("현재 배포 설정", cur, None)
    line("KF-CA (탐색이 찾은 재귀식)", base(**KFCA), KFCACtrl, ref)
    line("정식 CA 칼만 (같은 R/Q)", base(**KFCA), ProperKF, ref)

    print("\n  [2] R/Q 오차 민감도 - 실기에서 값을 정확히 못 맞출 때")
    print("  " + "-" * 74)
    for fr in (0.5, 1.0, 2.0):
        for fq in (0.5, 1.0, 2.0):
            if fr == 1.0 and fq == 1.0:
                continue
            p = base(**dict(KFCA, kf_R=KFCA["kf_R"]*fr, kf_Q=KFCA["kf_Q"]*fq))
            line("R x%.1f  Q x%.1f" % (fr, fq), p, KFCACtrl, ref)

    print("\n  [3] 데드타임 분포 변동 - sim 최대 모델오차")
    print("  " + "-" * 74)
    orig = realjit.dt_sample
    for med, sig, tag in ((0.80, 0.335, "중앙값 0.80 (지연 개선시)"),
                          (1.50, 0.335, "중앙값 1.50 (지연 악화시)"),
                          (1.10, 0.550, "꼬리 두꺼움 sigma 0.55")):
        realjit.dt_sample = (lambda m, s: (lambda rng: m*math.exp(rng.gauss(0., s))))(med, sig)
        r2 = ev_holdout(cur, ctrl=None)
        e2 = sc(r2)
        m3 = ev_holdout(base(**KFCA), ctrl=KFCACtrl)
        print("  %-40s %8.3f  %5.2f %6.2f %5.1f  %+6.1f%%"
              % (tag, sc(m3), m3['step_overshoot'], m3['rev_peak'],
                 m3['reach_reach'], 100*(sc(m3)-e2)/e2))
    realjit.dt_sample = orig


if __name__ == "__main__":
    main()
