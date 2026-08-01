# CUDA <-> Python parity gate

All controller tuning is done in `bench/aim_opt.py`; the shipped math lives in
`needaimbot/cuda/pd_controller.cuh`. Those are two hand-maintained copies of the
same recurrence, so **the tuning is only valid while they agree exactly**.

`./run_parity.sh` runs the real device function over a deterministic input
sequence and compares every emitted (dx, dy) against the Python model, for each
config path. Run it after touching either side.

Verified 2026-07-25: 400/400 exact-match on all three paths
(default, `class_switch_reject=0`, `ego_frame_filter=1`).
