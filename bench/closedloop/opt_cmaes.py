#!/usr/bin/env python3
"""CMA-ES optimization for aimbot controller parameters.

Uses a from-scratch CMA-ES implementation (no scipy dependency issues on Jetson).
Tests 4 discrete algorithm combos separately, each with full CMA-ES optimization
over 18 continuous parameters.
"""

import math
import json
import time
import random
import dataclasses
import sys
import os
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sim import (AimConfig, AimState, TRAJECTORIES, CONFIGS,
                 run_sim, SimResult, FPSCamera, NoiseModel,
                 PipelineModel, NetworkModel, TargetTrackingModel,
                 compute_aim_movement, rescale_aim_config_for_rate,
                 push_inflight)

TRAJECTORIES_LIST = list(TRAJECTORIES.keys())
SEEDS = [42, 137, 256, 999, 1337, 7777, 31415]
N_FRAMES = 300

# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------
@dataclass
class ParamDef:
    name: str
    lo: float
    hi: float
    init: float

PARAM_DEFS = [
    ParamDef("kp_x",              0.3,   1.5,   0.991),
    ParamDef("kp_y",              0.1,   1.2,   0.389),
    ParamDef("p_softness_x",      2.0,  25.0,   7.39),
    ParamDef("p_softness_y",      2.0,  25.0,  20.0),
    ParamDef("kd_x",              0.0,   0.3,   0.035),
    ParamDef("kd_y",              0.0,   0.3,   0.025),
    ParamDef("ff_gain",           0.0,   5.0,   2.08),
    ParamDef("predict_frames",    0.0,   6.0,   4.30),
    ParamDef("deadtime_frames",   1.0,   4.0,   3.90),
    ParamDef("oneeuro_min_cutoff",0.001, 0.5,   0.03),
    ParamDef("oneeuro_beta",      0.001, 0.5,   0.05),
    ParamDef("max_step",          3.0,  30.0,  14.55),
    ParamDef("akp_large_thresh",  5.0,  50.0,  26.67),
    ParamDef("akp_small_thresh",  0.5,  10.0,   1.79),
    ParamDef("akp_large_mult",    1.0,   2.0,   1.28),
    ParamDef("akp_small_mult",    0.5,   1.0,   0.97),
    ParamDef("ff_v_ema",          0.01,  0.5,   0.097),
    ParamDef("lead_vgate",        1.0,  20.0,   5.82),
]

ALGO_COMBOS = [
    {"sem": 1, "cga": 0, "jcad": 1, "pags": 1, "label": "SEM+JCAD+PAGS"},
    {"sem": 1, "cga": 0, "jcad": 0, "pags": 1, "label": "SEM+PAGS"},
    {"sem": 1, "cga": 1, "jcad": 1, "pags": 1, "label": "ALL4"},
    {"sem": 1, "cga": 0, "jcad": 0, "pags": 0, "label": "SEM_only"},
]

# ---------------------------------------------------------------------------
# Sigmoid transform for bounded parameters
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def inv_sigmoid(p: float) -> float:
    p = max(1e-10, min(1.0 - 1e-10, p))
    return math.log(p / (1.0 - p))

def to_bounded(z: np.ndarray) -> np.ndarray:
    result = np.empty_like(z)
    for i, pd in enumerate(PARAM_DEFS):
        s = sigmoid(z[i])
        result[i] = pd.lo + s * (pd.hi - pd.lo)
    return result

def from_bounded(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    for i, pd in enumerate(PARAM_DEFS):
        p = (x[i] - pd.lo) / (pd.hi - pd.lo)
        result[i] = inv_sigmoid(p)
    return result

# ---------------------------------------------------------------------------
# Run simulation with custom params
# ---------------------------------------------------------------------------
def run_sim_custom(cfg: AimConfig, traj_name: str, seed: int) -> SimResult:
    state = AimState()
    camera = FPSCamera()
    rng = random.Random(seed)
    noise = NoiseModel(seed=seed)
    pipeline = PipelineModel(rng)
    network = NetworkModel(rng)
    tracker = TargetTrackingModel(rng)
    traj_fn = TRAJECTORIES[traj_name]
    result = SimResult(traj_name, "custom", seed, N_FRAMES)

    model_size = 160
    screen_cx = model_size / 2.0
    screen_cy = model_size / 2.0
    scale_x = 1.0
    scale_y = 1.0
    target_class = 0
    prev_sign_x = 0
    prev_sign_y = 0

    for frame in range(N_FRAMES):
        frame_period_ms = pipeline.sample_frame_period_ms()
        world_x, world_y = traj_fn(frame)
        det_x, det_y = camera.world_to_screen(world_x, world_y)
        jx, jy = noise.jitter()
        det_x += jx
        det_y += jy
        conf = noise.confidence()
        target_class, flip_offset = noise.class_flip(target_class)
        det_y += flip_offset
        missed = noise.is_missed()
        if not missed and network.is_frame_lost():
            missed = True
            result.udp_losses += 1
        track_break = tracker.update(det_x, det_y, missed)
        if track_break and state.has_track:
            result.track_breaks += 1
            state.has_track = 0

        dx, dy = 0, 0
        if missed:
            result.misses += 1
            state.frames_since_seen += 1
            push_inflight(state, 0.0, 0.0)
        else:
            state.frames_since_seen = 0
            frame_cfg = dataclasses.replace(cfg)
            rate_scale = pipeline.compute_rate_scale(frame_period_ms)
            if rate_scale != 1.0:
                rescale_aim_config_for_rate(frame_cfg, rate_scale)
            frame_cfg.deadtime_frames = pipeline.compute_deadtime_frames(frame_period_ms)
            dx, dy = compute_aim_movement(
                det_x, det_y, screen_cx, screen_cy,
                scale_x, scale_y, frame_cfg, state,
                target_class, conf)

        camera.apply_mouse(dx, dy)
        result.total_movement += math.sqrt(dx*dx + dy*dy)
        actual_x, actual_y = camera.world_to_screen(world_x, world_y)
        err = math.sqrt((actual_x - screen_cx)**2 + (actual_y - screen_cy)**2)
        result.errors.append(err)

        cur_sign_x = 1 if (actual_x - screen_cx) > 0 else -1
        cur_sign_y = 1 if (actual_y - screen_cy) > 0 else -1
        if prev_sign_x != 0 and cur_sign_x != prev_sign_x:
            result.sign_flips_x += 1
        if prev_sign_y != 0 and cur_sign_y != prev_sign_y:
            result.sign_flips_y += 1
        prev_sign_x = cur_sign_x
        prev_sign_y = cur_sign_y

    return result

# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------
def make_config(params: np.ndarray, algo: dict) -> AimConfig:
    cfg = AimConfig()
    cfg.kp_x = float(params[0])
    cfg.kp_y = float(params[1])
    cfg.p_softness_x = float(params[2])
    cfg.p_softness_y = float(params[3])
    cfg.kd_x = float(params[4])
    cfg.kd_y = float(params[5])
    cfg.ff_gain = float(params[6])
    cfg.predict_frames = float(params[7])
    cfg.deadtime_frames = float(params[8])
    cfg.oneeuro_min_cutoff = float(params[9])
    cfg.oneeuro_beta = float(params[10])
    cfg.max_step = float(params[11])
    cfg.sem_enabled = float(algo["sem"])
    cfg.cga_enabled = float(algo["cga"])
    cfg.jcad_enabled = float(algo["jcad"])
    cfg.pags_enabled = float(algo["pags"])
    return cfg

def compute_baseline_maes() -> Dict[str, float]:
    baseline_cfg = CONFIGS['production']()
    maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_custom(baseline_cfg, traj, seed)
            errors.extend(r.errors)
        maes[traj] = sum(errors) / len(errors)
    return maes

def evaluate(params: np.ndarray, algo: dict, baseline_maes: dict,
             akp_params: np.ndarray) -> float:
    cfg = make_config(params, algo)
    traj_maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_custom(cfg, traj, seed)
            errors.extend(r.errors)
        traj_maes[traj] = sum(errors) / len(errors)

    avg_mae = sum(traj_maes.values()) / len(traj_maes)

    # Penalty for regression on any trajectory
    penalty = 0.0
    for traj in TRAJECTORIES_LIST:
        if traj_maes[traj] > baseline_maes[traj] * 1.001:
            excess = (traj_maes[traj] - baseline_maes[traj]) / baseline_maes[traj]
            penalty += excess * 50.0

    return avg_mae + penalty

# ---------------------------------------------------------------------------
# CMA-ES implementation (pure numpy)
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, x0: np.ndarray, sigma0: float = 0.5,
                 popsize: int = 20):
        self.dim = len(x0)
        self.mean = x0.copy()
        self.sigma = sigma0

        # Population size
        self.lam = popsize
        self.mu = self.lam // 2

        # Recombination weights
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        # Adaptation parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) /
                       ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # State
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.eigenvalues = np.ones(self.dim)
        self.eigenvectors = np.eye(self.dim)
        self.invsqrt_C = np.eye(self.dim)
        self.gen = 0
        self.eigen_update_gap = max(1, int(self.lam / (10 * self.dim)))
        self.chiN = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

    def ask(self) -> List[np.ndarray]:
        samples = []
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * (self.eigenvectors @ (self.eigenvalues * z))
            samples.append(x)
        return samples

    def tell(self, solutions: List[np.ndarray], fitnesses: List[float]):
        idx = np.argsort(fitnesses)
        old_mean = self.mean.copy()

        # Update mean
        self.mean = np.zeros(self.dim)
        for i in range(self.mu):
            self.mean += self.weights[i] * solutions[idx[i]]

        # Update evolution paths
        diff = self.mean - old_mean
        self.ps = (1 - self.cs) * self.ps + \
                  math.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  (self.invsqrt_C @ diff) / self.sigma

        hsig = (np.linalg.norm(self.ps) /
                math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) /
                self.chiN) < (1.4 + 2 / (self.dim + 1))
        hsig = float(hsig)

        self.pc = (1 - self.cc) * self.pc + \
                  hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  diff / self.sigma

        # Update covariance matrix
        artmp = np.zeros((self.dim, self.mu))
        for i in range(self.mu):
            artmp[:, i] = (solutions[idx[i]] - old_mean) / self.sigma

        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) +
                            (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * (artmp @ np.diag(self.weights) @ artmp.T)

        # Update step size
        self.sigma *= math.exp((self.cs / self.damps) *
                               (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = min(self.sigma, 10.0)

        # Eigen decomposition (periodic)
        self.gen += 1
        if self.gen % self.eigen_update_gap == 0:
            self._update_eigen()

    def _update_eigen(self):
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        try:
            D2, B = np.linalg.eigh(self.C)
            D2 = np.maximum(D2, 1e-20)
            self.eigenvalues = np.sqrt(D2)
            self.eigenvectors = B
            self.invsqrt_C = B @ np.diag(1.0 / np.sqrt(D2)) @ B.T
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.eigenvectors = np.eye(self.dim)
            self.invsqrt_C = np.eye(self.dim)

# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CMA-ES Optimization for Aimbot Controller")
    print("=" * 70)
    print(f"Params: {len(PARAM_DEFS)} continuous + 4 discrete algo combos")
    print(f"Seeds: {len(SEEDS)} | Trajectories: {len(TRAJECTORIES_LIST)}")
    print(f"Evals per fitness: {len(SEEDS)*len(TRAJECTORIES_LIST)} sim runs")
    print()

    # Compute baseline
    print("Computing baseline MAEs (production config)...")
    t0 = time.time()
    baseline_maes = compute_baseline_maes()
    baseline_avg = sum(baseline_maes.values()) / len(baseline_maes)
    print(f"Baseline avg MAE: {baseline_avg:.4f} ({time.time()-t0:.1f}s)")
    for t, m in baseline_maes.items():
        print(f"  {t:15s}: {m:.4f}")
    print()

    # Initial point in unbounded space
    x0_bounded = np.array([pd.init for pd in PARAM_DEFS])
    x0_z = from_bounded(x0_bounded)

    # Adaptive kp params placeholder (not in AimConfig, handled in sim internally)
    akp_params = np.array([26.67, 1.79, 1.28, 0.97])

    best_overall = None
    best_overall_fitness = float('inf')
    best_overall_combo = None
    all_results = []

    POPSIZE = 16
    MAX_GENS = 150

    for combo in ALGO_COMBOS:
        label = combo["label"]
        print(f"\n{'='*60}")
        print(f"Combo: {label}")
        print(f"{'='*60}")

        es = CMAES(x0_z, sigma0=0.5, popsize=POPSIZE)
        best_fitness = float('inf')
        best_params = None
        stagnation = 0
        t_start = time.time()

        for gen in range(MAX_GENS):
            solutions = es.ask()
            fitnesses = []

            for sol in solutions:
                bounded = to_bounded(sol)
                f = evaluate(bounded, combo, baseline_maes, akp_params)
                fitnesses.append(f)

            es.tell(solutions, fitnesses)

            gen_best = min(fitnesses)
            gen_best_idx = fitnesses.index(gen_best)

            if gen_best < best_fitness:
                best_fitness = gen_best
                best_params = to_bounded(solutions[gen_best_idx]).copy()
                stagnation = 0
            else:
                stagnation += 1

            if gen % 10 == 0 or gen_best < best_fitness + 0.001:
                elapsed = time.time() - t_start
                improv = (best_fitness - baseline_avg) / baseline_avg * 100
                print(f"  Gen {gen:3d} | best={best_fitness:.4f} "
                      f"({improv:+.1f}%) | gen_best={gen_best:.4f} "
                      f"| σ={es.sigma:.4f} | {elapsed:.0f}s")

            if stagnation > 50:
                print(f"  Early stop: {stagnation} gens without improvement")
                break

        # Final result for this combo
        if best_params is not None:
            improv = (best_fitness - baseline_avg) / baseline_avg * 100
            print(f"\n  Best for {label}: MAE={best_fitness:.4f} ({improv:+.1f}%)")

            # Detailed per-trajectory
            cfg = make_config(best_params, combo)
            traj_detail = {}
            for traj in TRAJECTORIES_LIST:
                errors = []
                for seed in SEEDS:
                    r = run_sim_custom(cfg, traj, seed)
                    errors.extend(r.errors)
                mae = sum(errors) / len(errors)
                traj_detail[traj] = mae
                b = baseline_maes[traj]
                ti = (mae - b) / b * 100
                status = "OK" if mae <= b * 1.001 else "REGRESS"
                print(f"    {traj:15s}: {mae:.4f} (baseline {b:.4f}, {ti:+.1f}%) [{status}]")

            result = {
                "combo": label,
                "best_mae": best_fitness,
                "improvement_pct": improv,
                "params": {pd.name: float(best_params[i]) for i, pd in enumerate(PARAM_DEFS)},
                "algo": {k: v for k, v in combo.items() if k != "label"},
                "per_trajectory": {t: {"mae": traj_detail[t],
                                       "baseline": baseline_maes[t],
                                       "improv_pct": (traj_detail[t]-baseline_maes[t])/baseline_maes[t]*100}
                                   for t in TRAJECTORIES_LIST},
                "generations": gen + 1,
            }
            all_results.append(result)

            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall = result
                best_overall_combo = label

    # Save results
    output = {
        "best_combo": best_overall_combo,
        "best_mae": best_overall_fitness,
        "baseline_avg_mae": baseline_avg,
        "improvement_pct": (best_overall_fitness - baseline_avg) / baseline_avg * 100 if best_overall else 0,
        "best": best_overall,
        "all_combos": all_results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "opt_cmaes.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    if best_overall:
        print(f"Best combo: {best_overall_combo}")
        print(f"Best MAE: {best_overall_fitness:.4f} "
              f"({output['improvement_pct']:+.1f}% vs baseline)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
