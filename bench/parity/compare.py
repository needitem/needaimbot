#!/usr/bin/env python3
"""Frame-by-frame CUDA <-> Python parity check.

Reads the CUDA run (inputs + emitted deltas) and pushes the SAME inputs through
the Python model the tuning was done on. Any disagreement means the tuning does
not describe the shipped controller.
"""
import sys
sys.path.insert(0, "..")
from aim_opt import OptCtrl, base_params


def py_run(rows, cls_reject=True, y_scale=1.0):
    p = base_params(kp_x=0.75, kp_y=0.82, soft_x=9.0, soft_y=8.0, kd_x=0.05, kd_y=0.06,
                    max_step=25.0, comp=1.0, w=1.0,
                    ff=1.4, ego_lag=2.25, v_ema=0.2,
                    vgate=9.0, ff_err_gate=18.0,
                    predict=2.6, pred_vgate=9.0, pred_err_gate=18.0,
                    oneeuro=True, mincut=0.3, beta=0.02, dcut=1.0,
                    cls_reject=1.0 if cls_reject else 0.0,
                    y_scale=y_scale)
    c = OptCtrl(p); c.reset()
    out = []
    prev_cls = None
    for rx, ry, cls, _, _ in rows:
        changed = (prev_cls is not None and cls != prev_cls)
        prev_cls = cls
        out.append(c.step(rx, ry, changed))
    return out


def main():
    rows = []
    for ln in open("cuda_out.txt"):
        a = ln.split()
        rows.append((float(a[0]), float(a[1]), int(a[2]), int(a[3]), int(a[4])))
    py = py_run(rows)

    n = len(rows)
    mism = [(i, rows[i][3], rows[i][4], py[i][0], py[i][1])
            for i in range(n) if (rows[i][3], rows[i][4]) != py[i]]
    tot_c = sum(abs(r[3]) + abs(r[4]) for r in rows)
    tot_p = sum(abs(d[0]) + abs(d[1]) for d in py)
    print("frames               : {}".format(n))
    print("exact-match frames   : {}/{}  ({:.1f}%)".format(n - len(mism), n, 100*(n-len(mism))/n))
    print("total |emit| CUDA/py : {} / {}".format(tot_c, tot_p))
    if mism:
        print("\nfirst mismatches (frame, cuda dx,dy, python dx,dy):")
        for m in mism[:12]:
            print("  f{:<4d} cuda=({:4d},{:4d})  py=({:4d},{:4d})".format(*m))
        worst = max(abs(m[1]-m[3]) + abs(m[2]-m[4]) for m in mism)
        print("\nworst single-frame delta difference: {} px".format(worst))
    else:
        print("\nPERFECT PARITY - the shipped kernel and the tuning model are the same controller.")
    return 0 if not mism else 1


if __name__ == "__main__":
    sys.exit(main())
