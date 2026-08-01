#!/usr/bin/env python3
"""Evaluation harness driven by the MEASURED dead-time distribution.

The earlier harness assumed a uniform +-0.5 frame jitter. The rig says otherwise:
aim-on E2E is p50 3.11ms / p95 8.74ms, i.e. a skewed heavy tail, which in frames
is median ~1.10 and p95 ~1.91. A uniform model understates exactly the tail
events that cause visible ringing, so the dead time here is drawn from a
lognormal fitted to those two points.

Noise, dropouts, outliers and head<->body switching all come from aim_opt.
"""
import math
import random

from aim_opt import base_params, OptCtrl, SC
from aim_opt import (NOISE_X, NOISE_Y, P_DROP, DRIFT, DRIFT_RHO,
                     P_OUTLIER, OUT_LO, OUT_HI, P_SWITCH, SW_LO, SW_HI, SW_XSD)


def dt_sample(rng):
    """median 1.10 frames, ~p95 1.91 (measured)."""
    return 1.10 * math.exp(rng.gauss(0.0, 0.335))


def run_real(p, scen, seed, frames=None, vx=1.1, step_dist=60.0, noise=True, ctrl=None):
    frames = frames or {"step": 150, "hold": 420, "track": 480,
                        "reversal": 300, "reach": 220}[scen]
    rng = random.Random(seed)
    c = (ctrl(p) if ctrl else OptCtrl(p)); c.reset()
    C = [SC, SC]
    if scen in ("step", "reach"):
        T = [SC + step_dist*.94, SC + step_dist*.34]; v = [0, 0]
    elif scen == "track":
        T = [SC - 60., SC]; v = [vx, vx*.2]
    elif scen == "reversal":
        T = [SC, SC]; v = [1.3, 0]
    else:
        T = [SC, SC]; v = [0, 0]
    d0 = math.hypot(T[0]-C[0], T[1]-C[1])
    au = ((T[0]-C[0])/d0, (T[1]-C[1])/d0) if d0 > 1e-9 else (1., 0.)
    hist = []; errs = []; par = []; drift = [0., 0.]
    sw = 0; so = (0., 0.); turn = None; peak = 0.; reach = None
    for f in range(frames):
        if scen == "reversal" and f > 0 and f % 75 == 0:
            v[0] = -v[0]; turn = f
        T[0] += v[0]; T[1] += v[1]
        if T[0] < 40: T[0] = 40; v[0] = abs(v[0])
        if T[0] > 280: T[0] = 280; v[0] = -abs(v[0])
        hist.append((T[0], T[1], C[0], C[1]))
        lat = max(0., dt_sample(rng))
        idx = max(0., f - lat); i0 = int(idx); i1 = min(i0+1, len(hist)-1); fr = idx - i0
        sT0 = hist[i0][0]*(1-fr) + hist[i1][0]*fr
        sT1 = hist[i0][1]*(1-fr) + hist[i1][1]*fr
        sC0 = hist[i0][2]*(1-fr) + hist[i1][2]*fr
        sC1 = hist[i0][3]*(1-fr) + hist[i1][3]*fr
        mx, my = sT0 - sC0 + SC, sT1 - sC1 + SC
        cls = False
        if noise:
            if rng.random() < P_DROP:
                dx = dy = 0; c.res_x = c.res_y = 0.
            else:
                s = math.sqrt(max(1e-9, 1 - DRIFT_RHO**2))
                drift[0] = DRIFT_RHO*drift[0] + rng.gauss(0, DRIFT*s)
                drift[1] = DRIFT_RHO*drift[1] + rng.gauss(0, DRIFT*s)
                mx += drift[0] + rng.gauss(0, NOISE_X)
                my += drift[1] + rng.gauss(0, NOISE_Y)
                if rng.random() < P_OUTLIER:
                    a = rng.random()*6.283
                    mg = OUT_LO + rng.random()*(OUT_HI-OUT_LO)
                    mx += mg*math.cos(a); my += mg*math.sin(a)
                if sw > 0:
                    mx += so[0]; my += so[1]; sw -= 1
                    if sw == 0: cls = True
                elif rng.random() < P_SWITCH:
                    sw = rng.randint(1, 4)
                    mg = rng.uniform(SW_LO, SW_HI)*(1 if rng.random() < .5 else -1)
                    so = (rng.gauss(0, SW_XSD), mg)
                    mx += so[0]; my += so[1]; cls = True
                dx, dy = c.step(mx, my, cls)
        else:
            dx, dy = c.step(mx, my, cls)
        C[0] += dx; C[1] += dy
        e = math.hypot(T[0]-C[0], T[1]-C[1]); errs.append(e)
        par.append((C[0]-(T[0]-d0*au[0]))*au[0] + (C[1]-(T[1]-d0*au[1]))*au[1])
        if reach is None and e < 4.: reach = f
        if scen == "reversal" and turn is not None and 0 <= f-turn < 50:
            peak = max(peak, e)
    if scen == "step":
        ov = max(0., max(par) - d0)
        ri = next((i for i, q in enumerate(par) if q >= d0-2.), frames)
        osc = 0; sg = 0
        for q in par[ri:]:
            s2 = 1 if (q-d0) > 1.5 else (-1 if (q-d0) < -1.5 else 0)
            if s2 and sg and s2 != sg: osc += 1
            if s2: sg = s2
        return dict(overshoot=ov, osc=osc)
    if scen == "reach":
        return dict(reach=reach if reach is not None else frames)
    if scen == "reversal":
        w = errs[100:]
        return dict(peak=peak, rms=math.sqrt(sum(x*x for x in w)/len(w)))
    w = errs[100:]
    return dict(rms=math.sqrt(sum(x*x for x in w)/len(w)))


def ev(p, seeds=24, ctrl=None):
    o = {}
    for nm, scn, kw in (("step", "step", dict(noise=False)),
                        ("hold", "hold", {}),
                        ("rev", "reversal", {}),
                        ("t1", "track", dict(vx=1.1)),
                        ("t4", "track", dict(vx=4.0)),
                        ("t8", "track", dict(vx=8.0)),
                        ("reach", "reach", dict(noise=False, step_dist=120.))):
        acc = {}
        for s in range(seeds):
            m = run_real(p, scn, s, ctrl=ctrl, **kw)
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            o[nm+"_"+k] = sum(v)/len(v)
    return o


def sc(m):
    return (.18*m['hold_rms'] + .18*m['t1_rms'] + .24*m['t4_rms']
            + .24*m['t8_rms'] + .16*m['rev_rms'])
