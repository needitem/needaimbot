#!/usr/bin/env python3
"""Balanced parameter optimization: no trajectory may regress vs baseline."""

import sys
import json
import random
import time

sys.path.insert(0, '/home/taeho/portfolio/inferencetool/bench/closedloop')
import sim

SEEDS = [42, 137, 256]
TRAJECTORIES = list(sim.TRAJECTORIES.keys())
N_SAMPLES = 1000

def evaluate(params):
    cfg = sim.AimConfig()
    cfg.kp_x = params['kp_x']
    cfg.kp_y = params['kp_y']
    cfg.p_softness_x = params['p_softness_x']
    cfg.p_softness_y = params['p_softness_y']
    cfg.kd_x = params['kd_x']
    cfg.kd_y = params['kd_y']
    cfg.ff_gain = params['ff_gain']
    cfg.predict_frames = params['predict_frames']
    cfg.deadtime_frames = params['deadtime_frames']
    cfg.oneeuro_min_cutoff = params['oneeuro_min_cutoff']
    cfg.oneeuro_beta = params['oneeuro_beta']
    cfg.max_step = params['max_step']
    cfg.sem_enabled = params['sem_enabled']
    cfg.cga_enabled = params['cga_enabled']
    cfg.jcad_enabled = params['jcad_enabled']
    cfg.pags_enabled = params['pags_enabled']

    sim.K_ADAPTIVE_KP_LARGE_THRESHOLD = params['akp_large_thresh']
    sim.K_ADAPTIVE_KP_SMALL_THRESHOLD = params['akp_small_thresh']
    sim.K_ADAPTIVE_KP_LARGE_MULT = params['akp_large_mult']
    sim.K_ADAPTIVE_KP_SMALL_MULT = params['akp_small_mult']

    per_traj = {}
    for traj in TRAJECTORIES:
        maes = []
        for seed in SEEDS:
            state = sim.AimState()
            camera = sim.FPSCamera(model_size=160)
            noise = sim.NoiseModel(seed=seed)
            traj_fn = sim.TRAJECTORIES[traj]
            n_frames = 300
            screen_cx = 80.0
            screen_cy = 80.0
            detection_buffer = []
            errors = []
            for frame in range(n_frames):
                world_x, world_y = traj_fn(frame)
                det_x, det_y = camera.world_to_screen(world_x, world_y)
                jx, jy = noise.jitter()
                det_x += jx
                det_y += jy
                conf = noise.confidence()
                target_class = 0
                target_class, flip_offset = noise.class_flip(target_class)
                det_y += flip_offset
                missed = noise.is_missed()
                detection_buffer.append((det_x, det_y, conf, target_class, missed))
                deadtime_delay = 3
                if len(detection_buffer) > deadtime_delay:
                    d = detection_buffer[-1 - deadtime_delay]
                    d_x, d_y, d_conf, d_class, d_missed = d
                    if d_missed:
                        state.frames_since_seen += 1
                        sim.push_inflight(state, 0.0, 0.0)
                        dx, dy = 0, 0
                    else:
                        state.frames_since_seen = 0
                        dx, dy = sim.compute_aim_movement(
                            d_x, d_y, screen_cx, screen_cy,
                            1.0, 1.0, cfg, state, d_class, d_conf)
                else:
                    dx, dy = 0, 0
                camera.apply_mouse(dx, dy)
                actual_x, actual_y = camera.world_to_screen(world_x, world_y)
                import math
                err = math.sqrt((actual_x - screen_cx)**2 + (actual_y - screen_cy)**2)
                errors.append(err)
            maes.append(sum(errors) / len(errors))
        per_traj[traj] = sum(maes) / len(maes)
    return per_traj

def random_params(rng):
    return {
        'kp_x': rng.uniform(0.6, 1.1),
        'kp_y': rng.uniform(0.5, 1.0),
        'p_softness_x': rng.uniform(5, 14),
        'p_softness_y': rng.uniform(8, 18),
        'kd_x': rng.uniform(0.03, 0.12),
        'kd_y': rng.uniform(0.02, 0.10),
        'ff_gain': rng.uniform(1.2, 2.5),
        'predict_frames': rng.uniform(2.5, 4.5),
        'deadtime_frames': rng.uniform(2.0, 3.5),
        'oneeuro_min_cutoff': rng.uniform(0.04, 0.12),
        'oneeuro_beta': rng.uniform(0.01, 0.04),
        'max_step': rng.uniform(14, 25),
        'sem_enabled': float(rng.randint(0, 1)),
        'cga_enabled': float(rng.randint(0, 1)),
        'jcad_enabled': float(rng.randint(0, 1)),
        'pags_enabled': float(rng.randint(0, 1)),
        'akp_large_thresh': rng.uniform(10, 25),
        'akp_small_thresh': rng.uniform(2, 8),
        'akp_large_mult': rng.uniform(1.05, 1.5),
        'akp_small_mult': rng.uniform(0.5, 0.95),
    }

def main():
    print("Computing baselines...")
    orig_akp_lt = sim.K_ADAPTIVE_KP_LARGE_THRESHOLD
    orig_akp_st = sim.K_ADAPTIVE_KP_SMALL_THRESHOLD
    orig_akp_lm = sim.K_ADAPTIVE_KP_LARGE_MULT
    orig_akp_sm = sim.K_ADAPTIVE_KP_SMALL_MULT

    baseline_params = {
        'kp_x': 0.765, 'kp_y': 0.698,
        'p_softness_x': 8.6, 'p_softness_y': 13.14,
        'kd_x': 0.052, 'kd_y': 0.037,
        'ff_gain': 1.7, 'predict_frames': 3.18,
        'deadtime_frames': 2.85, 'oneeuro_min_cutoff': 0.084,
        'oneeuro_beta': 0.02, 'max_step': 19.56,
        'sem_enabled': 0.0, 'cga_enabled': 0.0,
        'jcad_enabled': 0.0, 'pags_enabled': 0.0,
        'akp_large_thresh': orig_akp_lt, 'akp_small_thresh': orig_akp_st,
        'akp_large_mult': orig_akp_lm, 'akp_small_mult': orig_akp_sm,
    }
    baselines = evaluate(baseline_params)
    print("Baselines (MAE):")
    for t, v in baselines.items():
        print(f"  {t}: {v:.4f}")
    baseline_total = sum(baselines.values()) / len(baselines)
    print(f"  TOTAL AVG: {baseline_total:.4f}")

    rng = random.Random(12345)
    valid = []
    t0 = time.time()

    for i in range(N_SAMPLES):
        params = random_params(rng)
        per_traj = evaluate(params)

        is_valid = all(per_traj[t] <= baselines[t] for t in TRAJECTORIES)
        if is_valid:
            avg = sum(per_traj.values()) / len(per_traj)
            valid.append((avg, params, per_traj))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[{i+1}/{N_SAMPLES}] valid={len(valid)} | {rate:.1f} samples/s | elapsed={elapsed:.0f}s")

    valid.sort(key=lambda x: x[0])

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(valid)} valid candidates out of {N_SAMPLES}")
    print(f"{'='*70}")

    if not valid:
        print("NO valid candidates found (all regressed on at least one trajectory).")
        print("Consider widening the search or relaxing constraints.")
        result = {
            "status": "no_valid_candidates",
            "baselines": {k: round(v, 4) for k, v in baselines.items()},
            "total_candidates": N_SAMPLES,
            "valid_candidates": 0,
        }
        with open('/home/taeho/portfolio/inferencetool/bench/closedloop/opt_balanced.json', 'w') as f:
            json.dump(result, f, indent=2)
        return

    print(f"\nTop 5 candidates:")
    for rank, (avg, params, per_traj) in enumerate(valid[:5]):
        print(f"\n--- Rank {rank+1} (avg MAE: {avg:.4f}) ---")
        for t in TRAJECTORIES:
            bl = baselines[t]
            opt = per_traj[t]
            pct = (opt - bl) / bl * 100
            print(f"  {t:16s}: {bl:.4f} -> {opt:.4f} ({pct:+.1f}%)")

    best_avg, best_params, best_per_traj = valid[0]
    print(f"\n{'='*70}")
    print("BEST PARAMS:")
    for k, v in sorted(best_params.items()):
        if k.startswith(('sem_', 'cga_', 'jcad_', 'pags_')):
            print(f"  {k}: {'ON' if v else 'OFF'}")
        else:
            print(f"  {k}: {v:.6f}")

    print(f"\nPer-trajectory improvement:")
    improvements = {}
    for t in TRAJECTORIES:
        bl = baselines[t]
        opt = best_per_traj[t]
        pct = (opt - bl) / bl * 100
        improvements[t] = round(pct, 2)
        print(f"  {t:16s}: {bl:.4f} -> {opt:.4f} ({pct:+.1f}%)")

    total_opt = sum(best_per_traj.values()) / len(best_per_traj)
    total_pct = (total_opt - baseline_total) / baseline_total * 100
    print(f"  {'TOTAL AVG':16s}: {baseline_total:.4f} -> {total_opt:.4f} ({total_pct:+.1f}%)")

    result = {
        "best_params": {k: round(v, 6) for k, v in best_params.items()},
        "best_avg_mae": round(best_avg, 4),
        "baseline_avg_mae": round(baseline_total, 4),
        "total_improvement_pct": round(total_pct, 2),
        "per_trajectory_mae": {
            t: {"baseline": round(baselines[t], 4), "optimal": round(best_per_traj[t], 4),
                "improvement_pct": improvements[t]}
            for t in TRAJECTORIES
        },
        "total_candidates": N_SAMPLES,
        "valid_candidates": len(valid),
        "top_10": [
            {"avg_mae": round(a, 4), "params": {k: round(v, 6) for k, v in p.items()},
             "per_traj": {t: round(pt[t], 4) for t in TRAJECTORIES}}
            for a, p, pt in valid[:10]
        ],
    }

    with open('/home/taeho/portfolio/inferencetool/bench/closedloop/opt_balanced.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\nSaved to opt_balanced.json")

    with open('/home/taeho/portfolio/inferencetool/bench/closedloop/optimal_params.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Updated optimal_params.json")

if __name__ == "__main__":
    main()
