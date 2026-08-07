#!/usr/bin/env python3
"""최적 파라미터가 검출기 σ 에 대해 어떤 법칙을 따르는가.

값을 하드코딩할 수 없다는 게 출발점이다. 실측 σ 는 표적 거리·이동·장면에 따라 달라진다
(같은 리그에서 aim-off 4.51/2.84, aim-on 3.21/2.07, 정지표적 수집 1.69/1.13). sim 이
들고 있던 8.1/5.9 는 그 중 어느 것도 아니다.

대안은 σ 를 런타임에 추정하고 파라미터를 그 함수로 두는 것. 단 그게 성립하려면
**최적점이 σ 를 따라 규칙적으로 움직여야** 한다. 규칙이 없으면 추정해도 쓸 데가 없다.

이론 기대:
    softness  ~ σ        비선형 P 의 무릎을 노이즈 크기에 맞춘다
    mincut    ~ 1/σ      노이즈가 크면 더 세게 저역통과
    lead 게이트 ~ σ        노이즈 낀 속도/오차를 더 세게 게이팅
    kp        ~ 1/σ 약하게
"""
import json, math, random, statistics as st
import realjit
from realjit import run_real, sc
from ctrl_zoo_opt import base
import aim_opt

random.seed(5150)

FPS, DEAD_MS, SIG_DT, BASE = 201.0, 11.60, 0.295, 143.0
T = 1000.0/FPS
realjit.dt_sample = (lambda m: (lambda rng: m*math.exp(rng.gauss(0., SIG_DT))))(DEAD_MS/T)

SHIP = {"kp_x":0.445,"kp_y":0.406,"soft_x":8.6,"soft_y":6.57,"kd_x":0.052,"kd_y":0.037,
        "max_step":11.37,"w":2.85,"ff":2.924,"beta":0.02,"predict":5.47,"v_ema":0.137,
        "vgate":8.60,"ff_err_gate":22.25,"mincut":0.049,"ego_lag":2.85}
BOX = {"kp_x":(0.15,1.2),"kp_y":(0.15,1.2),"soft_x":(2.0,30.0),"soft_y":(2.0,30.0),
       "mincut":(0.01,0.8),"vgate":(1.0,40.0),"ff_err_gate":(4.0,80.0)}
NX0, NY0 = aim_opt.NOISE_X, aim_opt.NOISE_Y


def ev(p, n=20, lo=500):
    k = FPS/BASE; vs = BASE/FPS; o = {}
    spec = (("step","step",dict(noise=False,frames=int(150*k))),("hold","hold",dict(frames=int(420*k))),
            ("rev","reversal",dict(frames=int(300*k))),("t1","track",dict(vx=1.1*vs,frames=int(480*k))),
            ("t4","track",dict(vx=4.0*vs,frames=int(480*k))),("t8","track",dict(vx=8.0*vs,frames=int(480*k))),
            ("r40","reach",dict(noise=False,step_dist=40.,frames=int(220*k))))
    for nm,s_,kw in spec:
        acc={}
        for s in range(lo,lo+n):
            m=run_real(p,s_,s,**kw)
            for kk,v in m.items(): acc.setdefault(kk,[]).append(v)
        for kk,v in acc.items(): o[nm+"_"+kk]=sum(v)/len(v)
    o["r40_ms"]=o["r40_reach"]*T
    return o


def optimise(seed0, iters=200):
    p = dict(seed0); best = (sc(ev(p, n=10, lo=0)), dict(p))
    step, since = 0.30, 0
    for _ in range(iters):
        q = dict(p)
        for k,(lo_,hi) in BOX.items():
            if random.random() < 0.5:
                q[k] = min(hi, max(lo_, q[k] + random.gauss(0, step*(hi-lo_))))
        m = ev(q, n=10, lo=0)
        if m['step_overshoot'] <= 2.6 and m['step_osc'] <= 0.35 and sc(m) < best[0]:
            best = (sc(m), dict(q)); p, since = q, 0
        else:
            since += 1
            if since >= 14:
                step *= 0.65; since = 0
                if step < 0.01: break
    return best[1]


def main():
    print("  σ 배율별 최적점 (실측 리그 %.0ffps, 데드타임 %.2f프레임)\n" % (FPS, DEAD_MS/T))
    print("  %-7s %7s %7s %7s %7s %8s %8s %9s"
          % ("σ배율","kp_x","kp_y","soft_x","soft_y","mincut","vgate","err_gate"))
    print("  " + "-"*70)
    out = {}
    for mul in (0.4, 0.7, 1.0, 1.5, 2.5):
        aim_opt.NOISE_X, aim_opt.NOISE_Y = NX0*mul, NY0*mul
        realjit.NOISE_X, realjit.NOISE_Y = NX0*mul, NY0*mul
        b = optimise(base(**SHIP)); out[mul] = b
        print("  %-7.1f %7.3f %7.3f %7.2f %7.2f %8.4f %8.2f %9.2f"
              % (mul, b["kp_x"], b["kp_y"], b["soft_x"], b["soft_y"],
                 b["mincut"], b["vgate"], b["ff_err_gate"]))
    aim_opt.NOISE_X, aim_opt.NOISE_Y = NX0, NY0
    realjit.NOISE_X, realjit.NOISE_Y = NX0, NY0
    print("\n  로그-로그 기울기 (σ^k 의 k). R² 가 낮으면 법칙이 없는 것이다.")
    print("  기대: soft +1, mincut -1, 게이트 +1, kp 약간 음수\n")
    for k in BOX:
        xs=[math.log(m) for m in out]; ys=[math.log(out[m][k]) for m in out]
        mx,my=st.mean(xs),st.mean(ys)
        den=sum((x-mx)**2 for x in xs)
        sl=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den
        ss=sum((y-my)**2 for y in ys)
        pred=[my+sl*(x-mx) for x in xs]
        r2=1-sum((y-p)**2 for y,p in zip(ys,pred))/ss if ss>1e-12 else 0.0
        print("    %-12s k = %+.2f   R² = %.2f"%(k,sl,r2))


if __name__ == "__main__":
    main()
