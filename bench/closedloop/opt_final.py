#!/usr/bin/env python3
"""Two-phase no-regression parameter optimization on realistic sim.

Phase 1: 1500 broad random samples
Phase 2: 500 local refinement samples around top candidates
Constraint: every trajectory MAE must be <= baseline MAE (no regression)
"""

import sys
import json
import random
import math
import time

sys.path.insert(0, '/home/taeho/portfolio/inferencetool/bench/closedloop')
import sim
from sim import AimConfig, CONFIGS, TRAJECTORIES, run_sim

SEEDS = [42, 137, 256, 999, 1337]
TRAJ_NAMES = list(TRAJECTORIES.keys())

# Save/restore module-level constants
ORIG_CONSTANTS = {
    'K_ADAPTIVE_KP_LARGE_THRESHOLD': sim.K_ADAPTIVE_KP_LARGE_THRESHOLD,
    'K_ADAPTIVE_KP_SMALL_THRESHOLD': sim.K_ADAPTIVE_KP_SMALL_THRESHOLD,
    'K_ADAPTIVE_KP_LARGE_MULT': sim.K_ADAPTIVE_KP_LARGE_MULT,
    'K_ADAPTIVE_KP_SMALL_MULT': sim.K_ADAPTIVE_KP_SMALL_MULT,
    'K_FF_V_EMA': sim.K_FF_V_EMA,
    'K_LEAD_VGATE': sim.K_LEAD_VGATE,
}

def set_sim_constants(params):
    sim.K_ADAPTIVE_KP_LARGE_THRESHOLD = params.get('akp_large_thresh', ORIG_CONSTANTS['K_ADAPTIVE_KP_LARGE_THRESHOLD'])
    sim.K_ADAPTIVE_KP_SMALL_THRESHOLD = params.get('akp_small_thresh', ORIG_CONSTANTS['K_ADAPTIVE_KP_SMALL_THRESHOLD'])
    sim.K_ADAPTIVE_KP_LARGE_MULT = params.get('akp_large_mult', ORIG_CONSTANTS['K_ADAPTIVE_KP_LARGE_MULT'])
    sim.K_ADAPTIVE_KP_SMALL_MULT = params.get('akp_small_mult', ORIG_CONSTANTS['K_ADAPTIVE_KP_SMALL_MULT'])
    sim.K_FF_V_EMA = params.get('ff_v_ema', ORIG_CONSTANTS['K_FF_V_EMA'])
    sim.K_LEAD_VGATE = params.get('lead_vgate', ORIG_CONSTANTS['K_LEAD_VGATE'])

def restore_sim_constants():
    for k, v in ORIG_CONSTANTS.items():
        setattr(sim, k, v)

def make_config(params):
    c = AimConfig()
    c.kp_x = params['kp_x']
    c.kp_y = params['kp_y']
    c.p_softness_x = params['p_softness_x']
    c.p_softness_y = params['p_softness_y']
    c.kd_x = params['kd_x']
    c.kd_y = params['kd_y']
    c.ff_gain = params['ff_gain']
    c.predict_frames = params['predict_frames']
    c.deadtime_frames = params['deadtime_frames']
    c.oneeuro_min_cutoff = params['oneeuro_min_cutoff']
    c.oneeuro_beta = params['oneeuro_beta']
    c.max_step = params['max_step']
    c.sem_enabled = params['sem_enabled']
    c.cga_enabled = params['cga_enabled']
    c.jcad_enabled = params['jcad_enabled']
    c.pags_enabled = params['pags_enabled']
    return c

def eval_candidate(params, baselines):
    set_sim_constants(params)
    cfg = make_config(params)
    CONFIGS['_candidate'] = lambda: cfg
    per_traj = {}
    for traj in TRAJ_NAMES:
        maes = []
        for seed in SEEDS:
            r = run_sim(traj, '_candidate', seed=seed)
            maes.append(r.mae)
        avg_mae = sum(maes) / len(maes)
        if avg_mae > baselines[traj] * 1.001:  # tiny tolerance for float noise
            restore_sim_constants()
            return None, None
        per_traj[traj] = avg_mae
    restore_sim_constants()
    total_mae = sum(per_traj.values()) / len(per_traj)
    return total_mae, per_traj

def random_params(rng):
    return {
        'kp_x': rng.uniform(0.5, 1.2),
        'kp_y': rng.uniform(0.3, 1.0),
        'p_softness_x': rng.uniform(4, 15),
        'p_softness_y': rng.uniform(6, 20),
        'kd_x': rng.uniform(0.02, 0.15),
        'kd_y': rng.uniform(0.02, 0.12),
        'ff_gain': rng.uniform(1.0, 3.0),
        'predict_frames': rng.uniform(2.0, 5.0),
        'deadtime_frames': rng.uniform(2.0, 4.0),
        'oneeuro_min_cutoff': rng.uniform(0.03, 0.15),
        'oneeuro_beta': rng.uniform(0.005, 0.05),
        'max_step': rng.uniform(10, 25),
        'sem_enabled': float(rng.randint(0, 1)),
        'cga_enabled': float(rng.randint(0, 1)),
        'jcad_enabled': float(rng.randint(0, 1)),
        'pags_enabled': float(rng.randint(0, 1)),
        'akp_large_thresh': rng.uniform(10, 30),
        'akp_small_thresh': rng.uniform(1, 8),
        'akp_large_mult': rng.uniform(1.0, 1.5),
        'akp_small_mult': rng.uniform(0.5, 1.0),
        'ff_v_ema': rng.uniform(0.05, 0.4),
        'lead_vgate': rng.uniform(5, 20),
    }

def perturb_params(base, rng, sigma=0.10):
    p = {}
    ranges = {
        'kp_x': (0.5, 1.2), 'kp_y': (0.3, 1.0),
        'p_softness_x': (4, 15), 'p_softness_y': (6, 20),
        'kd_x': (0.02, 0.15), 'kd_y': (0.02, 0.12),
        'ff_gain': (1.0, 3.0), 'predict_frames': (2.0, 5.0),
        'deadtime_frames': (2.0, 4.0),
        'oneeuro_min_cutoff': (0.03, 0.15), 'oneeuro_beta': (0.005, 0.05),
        'max_step': (10, 25),
        'akp_large_thresh': (10, 30), 'akp_small_thresh': (1, 8),
        'akp_large_mult': (1.0, 1.5), 'akp_small_mult': (0.5, 1.0),
        'ff_v_ema': (0.05, 0.4), 'lead_vgate': (5, 20),
    }
    for k, v in base.items():
        if k in ('sem_enabled', 'cga_enabled', 'jcad_enabled', 'pags_enabled'):
            if rng.random() < 0.1:
                p[k] = 1.0 - v
            else:
                p[k] = v
        elif k in ranges:
            lo, hi = ranges[k]
            new_v = v + rng.gauss(0, sigma * (hi - lo))
            p[k] = max(lo, min(hi, new_v))
        else:
            p[k] = v
    return p

def compute_baselines():
    print("Computing baselines...")
    baselines = {}
    restore_sim_constants()
    for traj in TRAJ_NAMES:
        maes = []
        for seed in SEEDS:
            r = run_sim(traj, 'baseline', seed=seed)
            maes.append(r.mae)
        baselines[traj] = sum(maes) / len(maes)
        print(f"  {traj}: {baselines[traj]:.4f}")
    total = sum(baselines.values()) / len(baselines)
    print(f"  TOTAL avg: {total:.4f}")
    return baselines

def main():
    t0 = time.time()
    rng = random.Random(12345)
    baselines = compute_baselines()
    baseline_total = sum(baselines.values()) / len(baselines)

    # Phase 1: Broad search
    print(f"\n{'='*60}")
    print(f"Phase 1: Broad random search (1500 samples)")
    print(f"{'='*60}")
    valid = []
    for i in range(1500):
        params = random_params(rng)
        score, per_traj = eval_candidate(params, baselines)
        if score is not None:
            valid.append((score, params, per_traj))
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/1500] valid: {len(valid)}, elapsed: {elapsed:.0f}s")

    valid.sort(key=lambda x: x[0])
    print(f"\nPhase 1 done: {len(valid)} valid candidates")
    if valid:
        print(f"Best phase 1 score: {valid[0][0]:.4f} (baseline: {baseline_total:.4f})")
        for traj in TRAJ_NAMES:
            bl = baselines[traj]
            opt = valid[0][2][traj]
            pct = (opt - bl) / bl * 100
            print(f"  {traj}: {bl:.4f} -> {opt:.4f} ({pct:+.1f}%)")

    if not valid:
        print("No valid candidates found in phase 1!")
        return

    # Phase 2: Local refinement
    print(f"\n{'='*60}")
    print(f"Phase 2: Local refinement (500 samples)")
    print(f"{'='*60}")
    top_n = min(10, len(valid))
    phase2_valid = list(valid)  # carry over phase 1 results
    count = 0
    for rank in range(top_n):
        base_params = valid[rank][1]
        samples_per = 500 // top_n
        for j in range(samples_per):
            params = perturb_params(base_params, rng, sigma=0.10)
            score, per_traj = eval_candidate(params, baselines)
            if score is not None:
                phase2_valid.append((score, params, per_traj))
            count += 1
            if count % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{count}/500] valid total: {len(phase2_valid)}, elapsed: {elapsed:.0f}s")

    phase2_valid.sort(key=lambda x: x[0])
    phase2_new = len(phase2_valid) - len(valid)
    print(f"\nPhase 2 done: {phase2_new} new valid candidates")
    print(f"Total valid: {len(phase2_valid)}")

    best_score, best_params, best_per_traj = phase2_valid[0]
    improvement = (best_score - baseline_total) / baseline_total * 100

    print(f"\n{'='*60}")
    print(f"BEST RESULT (total MAE: {best_score:.4f}, improvement: {improvement:+.1f}%)")
    print(f"{'='*60}")
    for traj in TRAJ_NAMES:
        bl = baselines[traj]
        opt = best_per_traj[traj]
        pct = (opt - bl) / bl * 100
        print(f"  {traj}: {bl:.4f} -> {opt:.4f} ({pct:+.1f}%)")

    print(f"\nBest params:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    # Build top 5
    top5 = []
    for score, params, per_traj in phase2_valid[:5]:
        top5.append({
            'avg_mae': round(score, 4),
            'params': {k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()},
            'per_traj': {k: round(v, 4) for k, v in per_traj.items()},
        })

    result = {
        'best_params': {k: round(v, 6) if isinstance(v, float) else v for k, v in best_params.items()},
        'best_avg_mae': round(best_score, 4),
        'baseline_avg_mae': round(baseline_total, 4),
        'total_improvement_pct': round(improvement, 2),
        'per_trajectory_mae': {
            traj: {
                'baseline': round(baselines[traj], 4),
                'optimal': round(best_per_traj[traj], 4),
                'improvement_pct': round((best_per_traj[traj] - baselines[traj]) / baselines[traj] * 100, 2),
            }
            for traj in TRAJ_NAMES
        },
        'valid_candidates_phase1': len(valid),
        'valid_candidates_phase2': phase2_new,
        'total_valid': len(phase2_valid),
        'top_5': top5,
    }

    out_path = '/home/taeho/portfolio/inferencetool/bench/closedloop/opt_final.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Also update optimal_params.json
    opt_path = '/home/taeho/portfolio/inferencetool/bench/closedloop/optimal_params.json'
    with open(opt_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Updated {opt_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

if __name__ == '__main__':
    main()
