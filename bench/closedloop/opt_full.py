#!/usr/bin/env python3
"""Full joint random search over ALL controller parameters simultaneously."""

import sys, math, json, random
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
                        dx, dy = compute_aim_movement(d_x, d_y, screen_cx, screen_cy, scale_x, scale_y, cfg, state, d_class, d_conf)
                else:
                    dx, dy = 0, 0
                camera.apply_mouse(dx, dy)
                actual_x, actual_y = camera.world_to_screen(world_x, world_y)
                err = math.sqrt((actual_x - screen_cx)**2 + (actual_y - screen_cy)**2)
                errors.append(err)
            mae = sum(errors) / len(errors)
            rms = math.sqrt(sum(e*e for e in errors) / len(errors))
            p95 = sorted(errors)[int(0.95 * len(errors))]
            scores.append(0.5*mae + 0.3*rms + 0.2*p95)
    return sum(scores) / len(scores)


def main():
    rng = random.Random(77777)
    n_samples = 500
    top_results = []

    print(f"Full Joint Random Search: {n_samples} samples, 6 trajectories x 3 seeds")
    print("=" * 70)

    for i in range(n_samples):
        params = {
            "kp_x": rng.uniform(0.3, 1.2),
            "kp_y": rng.uniform(0.3, 1.0),
            "p_softness_x": rng.uniform(4, 16),
            "p_softness_y": rng.uniform(6, 20),
            "kd_x": rng.uniform(0.01, 0.15),
            "kd_y": rng.uniform(0.01, 0.15),
            "ff_gain": rng.uniform(0.5, 3.0),
            "predict_frames": rng.uniform(1.0, 5.0),
            "deadtime_frames": rng.uniform(1.5, 4.0),
            "oneeuro_beta": rng.uniform(0.005, 0.05),
            "oneeuro_min_cutoff": rng.uniform(0.03, 0.15),
            "max_step": rng.uniform(10, 30),
            "sem_enabled": rng.choice([0.0, 1.0]),
            "cga_enabled": rng.choice([0.0, 1.0]),
            "jcad_enabled": rng.choice([0.0, 1.0]),
            "pags_enabled": rng.choice([0.0, 1.0]),
        }

        cfg = AimConfig()
        for k, v in params.items():
            setattr(cfg, k, v)

        score = eval_config(cfg)
        top_results.append({"params": params, "score": score})

        if (i + 1) % 50 == 0:
            top_results.sort(key=lambda x: x["score"])
            best = top_results[0]
            print(f"[{i+1:4d}/{n_samples}] best_score={best['score']:.4f}")

    top_results.sort(key=lambda x: x["score"])
    top_10 = top_results[:10]

    best = top_10[0]
    print(f"\n{'=' * 70}")
    print(f"BEST SCORE: {best['score']:.4f}")
    print(f"BEST PARAMS:")
    for k, v in sorted(best["params"].items()):
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nTOP 10:")
    for rank, entry in enumerate(top_10, 1):
        print(f"  #{rank}: score={entry['score']:.4f}")

    output = {
        "best_params": best["params"],
        "best_score": best["score"],
        "top_10": [{"params": e["params"], "score": e["score"]} for e in top_10],
        "search_stats": {
            "total_samples": n_samples,
            "search_space": "full_joint",
            "seeds": [42, 137, 256],
        },
    }

    out_path = "/home/taeho/portfolio/inferencetool/bench/closedloop/opt_full.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
