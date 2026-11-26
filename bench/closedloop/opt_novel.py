#!/usr/bin/env python3
"""Phase 1: Search all 16 on/off combinations of SEM, CGA, JCAD, PAGS.
   Phase 2: Optimize adaptive kp thresholds using the best combo from Phase 1.
"""

import sys
import math
import json
import random
import itertools
import time

sys.path.insert(0, '/home/taeho/portfolio/inferencetool/bench/closedloop')
from sim import *
import sim as sim_module


def eval_config(cfg, n_frames=300, deadtime_delay=3, seeds=[42, 137, 256]):
    scores = []
    for traj_name in TRAJECTORIES:
        for seed in seeds:
            state = AimState()
            camera = FPSCamera()
            noise = NoiseModel(seed=seed)
            traj_fn = TRAJECTORIES[traj_name]
            model_size = 160
            screen_cx = model_size / 2.0
            screen_cy = model_size / 2.0
            scale_x = scale_y = 1.0
            target_class = 0
            detection_buffer = []
            errors = []
            for frame in range(n_frames):
                world_x, world_y = traj_fn(frame)
                det_x, det_y = camera.world_to_screen(world_x, world_y)
                jx, jy = noise.jitter()
                det_x += jx
                det_y += jy
                conf = noise.confidence()
                target_class, flip_offset = noise.class_flip(target_class)
                det_y += flip_offset
                missed = noise.is_missed()
                detection_buffer.append((det_x, det_y, conf, target_class, missed))
                if len(detection_buffer) > deadtime_delay:
                    d = detection_buffer[-1 - deadtime_delay]
                    d_x, d_y, d_conf, d_class, d_missed = d
                    if d_missed:
                        state.frames_since_seen += 1
                        push_inflight(state, 0.0, 0.0)
                        dx, dy = 0, 0
                    else:
                        state.frames_since_seen = 0
                        dx, dy = compute_aim_movement(
                            d_x, d_y, screen_cx, screen_cy,
                            scale_x, scale_y, cfg, state, d_class, d_conf)
                else:
                    dx, dy = 0, 0
                camera.apply_mouse(dx, dy)
                actual_x, actual_y = camera.world_to_screen(world_x, world_y)
                err = math.sqrt((actual_x - screen_cx)**2 + (actual_y - screen_cy)**2)
                errors.append(err)
            mae = sum(errors) / len(errors)
            rms = math.sqrt(sum(e * e for e in errors) / len(errors))
            p95 = sorted(errors)[int(0.95 * len(errors))]
            scores.append(0.5 * mae + 0.3 * rms + 0.2 * p95)
    return sum(scores) / len(scores)


# =========================================================================
# Phase 1: Algorithm Combination Search (16 combos)
# =========================================================================
print("=" * 60)
print("Phase 1: Algorithm Combination Search (16 combos)")
print("=" * 60)

algo_names = ["sem", "cga", "jcad", "pags"]
all_combos = []
best_algo_score = float('inf')
best_algo_combo = None

t0 = time.time()

for bits in range(16):
    sem_on = bool(bits & 1)
    cga_on = bool(bits & 2)
    jcad_on = bool(bits & 4)
    pags_on = bool(bits & 8)

    cfg = AimConfig()
    cfg.sem_enabled = 1.0 if sem_on else 0.0
    cfg.cga_enabled = 1.0 if cga_on else 0.0
    cfg.jcad_enabled = 1.0 if jcad_on else 0.0
    cfg.pags_enabled = 1.0 if pags_on else 0.0

    score = eval_config(cfg)
    combo = {"sem": sem_on, "cga": cga_on, "jcad": jcad_on, "pags": pags_on}
    all_combos.append({"combo": combo, "score": round(score, 6)})

    label = ",".join(k for k, v in combo.items() if v) or "none"
    print(f"  [{bits+1:2d}/16] {label:<20s} score={score:.4f}")

    if score < best_algo_score:
        best_algo_score = score
        best_algo_combo = combo

t1 = time.time()
print(f"\nPhase 1 done in {t1 - t0:.1f}s")
print(f"Best combo: {best_algo_combo}  score={best_algo_score:.4f}")

# =========================================================================
# Phase 2: Adaptive kp Optimization (100 samples)
# =========================================================================
print("\n" + "=" * 60)
print("Phase 2: Adaptive kp Optimization (100 samples)")
print("=" * 60)

rng = random.Random(99999)

# Apply best algo combo for phase 2
best_cfg_template = AimConfig()
best_cfg_template.sem_enabled = 1.0 if best_algo_combo["sem"] else 0.0
best_cfg_template.cga_enabled = 1.0 if best_algo_combo["cga"] else 0.0
best_cfg_template.jcad_enabled = 1.0 if best_algo_combo["jcad"] else 0.0
best_cfg_template.pags_enabled = 1.0 if best_algo_combo["pags"] else 0.0

adaptive_results = []
best_adaptive_score = float('inf')
best_adaptive_params = None

# Save original constants
orig_large_thresh = sim_module.K_ADAPTIVE_KP_LARGE_THRESHOLD
orig_small_thresh = sim_module.K_ADAPTIVE_KP_SMALL_THRESHOLD
orig_large_mult = sim_module.K_ADAPTIVE_KP_LARGE_MULT
orig_small_mult = sim_module.K_ADAPTIVE_KP_SMALL_MULT

t2 = time.time()

for i in range(100):
    large_thresh = rng.uniform(8, 25)
    small_thresh = rng.uniform(2, 8)
    large_mult = rng.uniform(1.1, 1.5)
    small_mult = rng.uniform(0.5, 0.9)

    # Monkey-patch module constants
    sim_module.K_ADAPTIVE_KP_LARGE_THRESHOLD = large_thresh
    sim_module.K_ADAPTIVE_KP_SMALL_THRESHOLD = small_thresh
    sim_module.K_ADAPTIVE_KP_LARGE_MULT = large_mult
    sim_module.K_ADAPTIVE_KP_SMALL_MULT = small_mult

    import copy
    cfg = copy.deepcopy(best_cfg_template)
    score = eval_config(cfg)

    params = {
        "large_threshold": round(large_thresh, 4),
        "small_threshold": round(small_thresh, 4),
        "large_mult": round(large_mult, 4),
        "small_mult": round(small_mult, 4),
    }
    adaptive_results.append({"params": params, "score": round(score, 6)})

    if score < best_adaptive_score:
        best_adaptive_score = score
        best_adaptive_params = params

    if (i + 1) % 10 == 0:
        print(f"  [{i+1:3d}/100] best so far: {best_adaptive_score:.4f}")

# Restore original constants
sim_module.K_ADAPTIVE_KP_LARGE_THRESHOLD = orig_large_thresh
sim_module.K_ADAPTIVE_KP_SMALL_THRESHOLD = orig_small_thresh
sim_module.K_ADAPTIVE_KP_LARGE_MULT = orig_large_mult
sim_module.K_ADAPTIVE_KP_SMALL_MULT = orig_small_mult

t3 = time.time()
print(f"\nPhase 2 done in {t3 - t2:.1f}s")
print(f"Best adaptive kp: {best_adaptive_params}  score={best_adaptive_score:.4f}")

# =========================================================================
# Top 10 adaptive results
# =========================================================================
adaptive_sorted = sorted(adaptive_results, key=lambda x: x["score"])
top_10 = adaptive_sorted[:10]

print("\nTop 10 adaptive kp configs:")
for rank, entry in enumerate(top_10, 1):
    p = entry["params"]
    print(f"  #{rank}: score={entry['score']:.4f}  "
          f"large_thresh={p['large_threshold']:.2f} small_thresh={p['small_threshold']:.2f} "
          f"large_mult={p['large_mult']:.3f} small_mult={p['small_mult']:.3f}")

# =========================================================================
# Save results
# =========================================================================
output = {
    "best_algo_combo": best_algo_combo,
    "best_algo_score": round(best_algo_score, 6),
    "all_combos": sorted(all_combos, key=lambda x: x["score"]),
    "best_adaptive_kp": best_adaptive_params,
    "best_adaptive_kp_score": round(best_adaptive_score, 6),
    "top_10_adaptive": top_10,
    "search_stats": {
        "algo_combos_tested": 16,
        "adaptive_kp_samples": 100,
        "phase1_time_s": round(t1 - t0, 1),
        "phase2_time_s": round(t3 - t2, 1),
    },
}

out_path = "/home/taeho/portfolio/inferencetool/bench/closedloop/opt_novel.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {out_path}")
print(f"Total time: {t3 - t0:.1f}s")
