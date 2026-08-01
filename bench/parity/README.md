# CUDA <-> Python parity gate

All controller tuning is done in `bench/aim_opt.py`; the shipped math lives in
`needaimbot/cuda/pd_controller.cuh`. Those are two hand-maintained copies of the
same recurrence, so **the tuning is only valid while they agree exactly**.

`./run_parity.sh` runs the real device function over a deterministic input
sequence and compares every emitted (dx, dy) against the Python model, for each
config path. Run it after touching either side.

The sequence flips head<->body on 8% of frames, so the anchor-flip path is
covered too - that path is easy to diverge on and invisible without it.

Verified 2026-08-01: 400/400 exact-match on both paths
(default, `class_switch_reject=0`). The third path was `ego_frame_filter=1`,
removed with that branch.
