#!/usr/bin/env python3
"""Random-search optimizer for lead/prediction parameters in the closed-loop sim."""

import sys, math, json, random
sys.path.insert(0, '/home/taeho/portfolio/inferencetool/bench/closedloop')
from sim import *


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
    rng = random.Random(54321)
    n_samples = 200
    results = []

    print(f"Lead/Prediction Parameter Optimization")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} | Trajectories: {len(TRAJECTORIES)} | Seeds: [42, 137, 256]")
    print(f"Search space: ff_gain, predict_frames, deadtime_frames, oneeuro_beta, oneeuro_min_cutoff")
    print()

    for i in range(n_samples):
        params = {
            "ff_gain": rng.uniform(0.5, 3.0),
            "predict_frames": rng.uniform(1.0, 5.0),
            "deadtime_frames": rng.uniform(1.5, 4.0),
            "oneeuro_beta": rng.uniform(0.005, 0.05),
            "oneeuro_min_cutoff": rng.uniform(0.03, 0.15),
        }

        cfg = AimConfig(
            ff_gain=params["ff_gain"],
            predict_frames=params["predict_frames"],
            deadtime_frames=params["deadtime_frames"],
            oneeuro_beta=params["oneeuro_beta"],
            oneeuro_min_cutoff=params["oneeuro_min_cutoff"],
        )

        score = eval_config(cfg)
        results.append({"params": params, "score": score})

        if (i + 1) % 50 == 0:
            best_so_far = min(results, key=lambda r: r["score"])
            print(f"[{i+1:3d}/{n_samples}] best_score={best_so_far['score']:.4f}  "
                  f"ff={best_so_far['params']['ff_gain']:.3f} "
                  f"pred={best_so_far['params']['predict_frames']:.3f} "
                  f"dt={best_so_far['params']['deadtime_frames']:.3f} "
                  f"beta={best_so_far['params']['oneeuro_beta']:.4f} "
                  f"mco={best_so_far['params']['oneeuro_min_cutoff']:.4f}")

    # Sort by score
    results.sort(key=lambda r: r["score"])
    best = results[0]

    print(f"\n{'='*60}")
    print(f"BEST RESULT")
    print(f"  Score: {best['score']:.4f}")
    for k, v in best["params"].items():
        print(f"  {k}: {v:.6f}")

    print(f"\nTOP 10:")
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1}: score={r['score']:.4f}  "
              f"ff={r['params']['ff_gain']:.3f} "
              f"pred={r['params']['predict_frames']:.3f} "
              f"dt={r['params']['deadtime_frames']:.3f} "
              f"beta={r['params']['oneeuro_beta']:.4f} "
              f"mco={r['params']['oneeuro_min_cutoff']:.4f}")

    # Round params for JSON
    def round_params(p):
        return {k: round(v, 6) for k, v in p.items()}

    output = {
        "best_params": round_params(best["params"]),
        "best_score": round(best["score"], 4),
        "top_10": [{"params": round_params(r["params"]), "score": round(r["score"], 4)} for r in results[:10]],
        "search_stats": {
            "total_samples": n_samples,
            "search_space": "lead_prediction",
            "seeds": [42, 137, 256],
        }
    }

    out_path = "/home/taeho/portfolio/inferencetool/bench/closedloop/opt_lead.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
