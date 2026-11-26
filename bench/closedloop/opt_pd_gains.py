#!/usr/bin/env python3
"""Random-search optimizer for PD controller gains using closed-loop simulation."""

import sys
sys.path.insert(0, '/home/taeho/portfolio/inferencetool/bench/closedloop')
from sim import *

import json
import math
import random
import time


def eval_config(cfg, n_frames=300, deadtime_delay=3, seeds=[42, 137, 256]):
    """Evaluate a config across all trajectories and seeds."""
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
            scale_x = 1.0
            scale_y = 1.0
            target_class = 0
            detection_buffer = []
            errors = []

            for frame in range(n_frames):
                world_x, world_y = traj_fn(frame)
                det_x, det_y = camera.world_to_screen(world_x, world_y)
                jx, jy = noise.jitter()
                det_x += jx; det_y += jy
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
                            scale_x, scale_y, cfg, state,
                            d_class, d_conf)
                else:
                    dx, dy = 0, 0

                camera.apply_mouse(dx, dy)
                actual_x, actual_y = camera.world_to_screen(world_x, world_y)
                err = math.sqrt((actual_x - screen_cx)**2 + (actual_y - screen_cy)**2)
                errors.append(err)

            mae = sum(errors) / len(errors)
            rms = math.sqrt(sum(e*e for e in errors) / len(errors))
            sorted_e = sorted(errors)
            p95 = sorted_e[int(0.95 * len(sorted_e))]
            scores.append(0.5 * mae + 0.3 * rms + 0.2 * p95)

    return sum(scores) / len(scores)


def main():
    rng = random.Random(12345)
    n_samples = 200
    seeds = [42, 137, 256]

    results = []
    best_score = float('inf')
    best_params = None

    # Evaluate production defaults first
    default_cfg = AimConfig()
    default_score = eval_config(default_cfg, seeds=seeds)
    print(f"Production defaults score: {default_score:.4f}")
    print(f"  kp_x={default_cfg.kp_x}, kp_y={default_cfg.kp_y}, "
          f"p_softness_x={default_cfg.p_softness_x}, p_softness_y={default_cfg.p_softness_y}, "
          f"kd_x={default_cfg.kd_x}, kd_y={default_cfg.kd_y}")
    print()

    t0 = time.time()
    for i in range(n_samples):
        params = {
            'kp_x': rng.uniform(0.3, 1.2),
            'kp_y': rng.uniform(0.3, 1.0),
            'p_softness_x': rng.uniform(4, 16),
            'p_softness_y': rng.uniform(6, 20),
            'kd_x': rng.uniform(0.01, 0.15),
            'kd_y': rng.uniform(0.01, 0.15),
        }

        cfg = AimConfig(
            kp_x=params['kp_x'],
            kp_y=params['kp_y'],
            p_softness_x=params['p_softness_x'],
            p_softness_y=params['p_softness_y'],
            kd_x=params['kd_x'],
            kd_y=params['kd_y'],
        )

        score = eval_config(cfg, seeds=seeds)
        results.append({'params': params, 'score': score})

        if score < best_score:
            best_score = score
            best_params = params

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[{i+1}/{n_samples}] best={best_score:.4f} elapsed={elapsed:.1f}s")

    # Sort and get top 10
    results.sort(key=lambda r: r['score'])
    top_10 = [{'params': r['params'], 'score': round(r['score'], 6)} for r in results[:10]]

    output = {
        'best_params': {k: round(v, 6) for k, v in best_params.items()},
        'best_score': round(best_score, 6),
        'top_10': top_10,
        'search_stats': {
            'total_samples': n_samples,
            'search_space': 'pd_gains',
            'seeds': seeds,
        },
        'production_default_score': round(default_score, 6),
    }

    out_path = '/home/taeho/portfolio/inferencetool/bench/closedloop/opt_pd_gains.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSearch complete. Results saved to {out_path}")
    print(f"\nProduction default score: {default_score:.4f}")
    print(f"Best score found:        {best_score:.4f}")
    improvement = (default_score - best_score) / default_score * 100
    print(f"Improvement:             {improvement:+.2f}%")
    print(f"\nBest parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6f}")

    print(f"\nTop 10:")
    for i, r in enumerate(top_10):
        print(f"  #{i+1}: score={r['score']:.4f}  kp_x={r['params']['kp_x']:.4f} kp_y={r['params']['kp_y']:.4f} "
              f"soft_x={r['params']['p_softness_x']:.2f} soft_y={r['params']['p_softness_y']:.2f} "
              f"kd_x={r['params']['kd_x']:.4f} kd_y={r['params']['kd_y']:.4f}")


if __name__ == '__main__':
    main()
