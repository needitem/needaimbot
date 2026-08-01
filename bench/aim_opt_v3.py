#!/usr/bin/env python3
"""Full joint re-optimisation on the REALISTIC noise (now including the measured
head<->body class switching), with the GAINS unfrozen.

Why revisit: every pass so far froze kp/softness/kd/max_step to isolate the lead
term's effect, and tuned against a noise model without class switching. Both
assumptions are now lifted:
  - the lead carries the tracking load, so the P gains no longer have to be high
    for tracking - they only handle acquisition and disturbance. Lower gains may
    cut noise amplification (rest jitter) at no tracking cost.
  - class switching is ~9% of the total error and was absent from the tuning
    noise, so the previous optimum was found against the wrong distribution.
"""
import json
import random
import aim_opt as O
from aim_opt import base_params, evaluate, error_score

random.seed(20260725)

SHIPPED = base_params(ff=1.4, ego_lag=2.25, v_ema=0.2, vgate=9.0, ff_err_gate=18.0,
                      predict=2.6, pred_vgate=9.0, pred_err_gate=18.0,
                      mincut=0.3, dcut=1.0)

def sample():
    vg = round(random.uniform(4.0, 14.0), 2)
    eg = round(random.uniform(10.0, 28.0), 2)
    return dict(
        kp_x=round(random.uniform(0.40, 1.10), 3),
        kp_y=round(random.uniform(0.40, 1.10), 3),
        soft_x=round(random.uniform(5.0, 16.0), 2),
        soft_y=round(random.uniform(5.0, 16.0), 2),
        kd_x=round(random.uniform(0.0, 0.20), 3),
        kd_y=round(random.uniform(0.0, 0.20), 3),
        max_step=round(random.uniform(14.0, 38.0), 1),
        comp=round(random.uniform(0.8, 1.0), 3),
        w=round(random.uniform(0.8, 1.8), 3),
        oneeuro=True,
        mincut=round(random.uniform(0.10, 0.60), 3),
        beta=round(random.uniform(0.0, 0.12), 3),
        dcut=1.0,
        ff=round(random.uniform(0.6, 2.0), 3),
        ego_lag=round(random.choice([1.75, 2.0, 2.25, 2.5, 3.0]), 2),
        v_ema=round(random.uniform(0.12, 0.40), 3),
        vgate=vg, ff_err_gate=eg,          # shared pair, as shipped
        predict=round(random.uniform(1.0, 3.6), 3),
        pred_vgate=vg, pred_err_gate=eg,
    )

KEYS = ("kp_x","kp_y","soft_x","soft_y","kd_x","kd_y","max_step","comp","w",
        "mincut","beta","ff","ego_lag","v_ema","vgate","ff_err_gate","predict")

def show(tag, m):
    print("  {:22s} ov={:5.2f} osc={:4.2f} revpk={:5.1f} rch={:4.1f} | err={:7.3f} "
          "h{:5.2f} t1{:5.2f} t4{:6.2f} t8{:6.2f}".format(
              tag, m["step_overshoot"], m["step_osc"], m["reversal_peak"], m["reach_reach"],
              error_score(m), m["hold_rms"], m["track_rms"], m["track_fast_rms"],
              m["track_sprint_rms"]))

if __name__ == "__main__":
    b = evaluate(SHIPPED, seeds=14)
    be = error_score(b)
    print("SHIPPED on the realistic (class-switching) noise"); show("shipped", b)
    RING = b["step_overshoot"]; OSC = max(0.35, b["step_osc"])
    REV = b["reversal_peak"]*1.03; RCH = b["reach_reach"]*1.10
    print("\nconstraints: ov<={:.2f} osc<={:.2f} revpk<={:.1f} reach<={:.1f}; minimise err (base {:.3f})"
          .format(RING, OSC, REV, RCH, be))

    def ok(m):
        return (m["step_overshoot"] <= RING and m["step_osc"] <= OSC
                and m["reversal_peak"] <= REV and m["reach_reach"] <= RCH)

    print("\n[1] random screen 1500, seeds=4 ...")
    pool = []
    for _ in range(1500):
        p = sample(); m = evaluate(p, seeds=4)
        if ok(m) and error_score(m) < be:
            pool.append((error_score(m), p))
    pool.sort(key=lambda x: x[0])
    print("    feasible+better: {}/1500".format(len(pool)))
    if not pool:
        print("    -> shipped already sits on the constrained optimum for this space")
        raise SystemExit

    print("\n[2] re-score top 40, seeds=14 ...")
    top = []
    for _, p in pool[:40]:
        m = evaluate(p, seeds=14)
        if ok(m) and error_score(m) < be:
            top.append((error_score(m), p, m))
    top.sort(key=lambda x: x[0])
    for s, p, m in top[:5]: show("cand {:.3f}".format(s), m)
    if not top:
        print("    none survived re-scoring -> shipped is at the optimum")
        raise SystemExit

    print("\n[3] coordinate refine ...")
    bs, bp, bm = top[0]
    GRID = {"kp_x":[-.05,-.025,.025,.05], "kp_y":[-.05,-.025,.025,.05],
            "soft_x":[-1.0,-.5,.5,1.0], "soft_y":[-1.0,-.5,.5,1.0],
            "kd_x":[-.02,-.01,.01,.02], "kd_y":[-.02,-.01,.01,.02],
            "max_step":[-2,-1,1,2], "comp":[-.04,-.02,.02,.04],
            "w":[-.08,-.04,.04,.08], "mincut":[-.04,-.02,.02,.04],
            "beta":[-.02,-.01,.01,.02], "ff":[-.1,-.05,.05,.1],
            "v_ema":[-.04,-.02,.02,.04], "predict":[-.2,-.1,.1,.2]}
    PAIR = {"vgate":[-1.0,-.5,.5,1.0], "ff_err_gate":[-2,-1,1,2]}
    for rnd in range(5):
        improved = False
        for k, ds in GRID.items():
            for d in ds:
                q = dict(bp); q[k] = round(q[k]+d, 4)
                if k in ("kd_x","kd_y","beta"):
                    if q[k] < 0: continue
                elif q[k] <= 0: continue
                m = evaluate(q, seeds=14)
                if ok(m) and error_score(m) < bs - 1e-4:
                    bs, bp, bm = error_score(m), q, m; improved = True
        for k, ds in PAIR.items():          # keep the shared gate pair in sync
            for d in ds:
                q = dict(bp); q[k] = round(q[k]+d, 4)
                if q[k] <= 0: continue
                q["pred_vgate" if k == "vgate" else "pred_err_gate"] = q[k]
                m = evaluate(q, seeds=14)
                if ok(m) and error_score(m) < bs - 1e-4:
                    bs, bp, bm = error_score(m), q, m; improved = True
        print("    round {}: err={:.4f}".format(rnd+1, bs))
        if not improved: break

    print("\n[RESULT]")
    show("shipped", b); show("re-optimised", bm)
    print("\n  params (shipped -> new):")
    for k in KEYS:
        print("    {:12s} {:8.3f} -> {:8.3f}".format(k, SHIPPED[k], bp[k]))
    print("\n  ERROR {:.3f} -> {:.3f} ({:+.1f}%)".format(be, bs, 100*(bs-be)/be))
    print("  RING  ov {:.2f} -> {:.2f} | osc {:.2f} -> {:.2f} | revpk {:.1f} -> {:.1f}".format(
        b["step_overshoot"], bm["step_overshoot"], b["step_osc"], bm["step_osc"],
        b["reversal_peak"], bm["reversal_peak"]))
    print("  SPEED reach {:.1f} -> {:.1f}".format(b["reach_reach"], bm["reach_reach"]))
    print("\nBEST_JSON " + json.dumps({k: bp[k] for k in KEYS}))
