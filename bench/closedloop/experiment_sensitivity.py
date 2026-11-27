#!/usr/bin/env python3
"""Experiment: adaptive mouse sensitivity calibration.

Tests whether online estimation of the camera's effective sensitivity
(the ratio between commanded mouse deltas and observed screen-pixel
shifts) can improve tracking accuracy.

Motivation
----------
The production controller uses fixed scale_x=1.0, scale_y=1.0 to map
model-pixel movements to mouse counts.  The camera has a *non-linear*
speed-dependent sensitivity (FPSCamera.apply_mouse deceleration at high
speeds), and in production the actual game sensitivity may differ from
the assumed 0.022 deg/px.  An adaptive calibrator tracks the ratio
  observed_pixel_shift / commanded_mouse_movement
using an EMA and scales future outputs to compensate.

New parameters (appended after the existing 18)
-------------------------------------------------
  sens_ema   : EMA rate for sensitivity estimate [0.01, 0.3], init 0.05
  sens_init  : initial sensitivity estimate       [0.5, 2.0], init 1.0
"""

import math
import json
import time
import random
import dataclasses
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
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
# Parameter space: 18 existing + 2 sensitivity calibration
# Initial values from opt_cmaes2.json (current best)
# ---------------------------------------------------------------------------
@dataclass
class ParamDef:
    name: str
    lo: float
    hi: float
    init: float

PARAM_DEFS = [
    # 18 existing params (init from current best = production defaults)
    ParamDef("kp_x",              0.3,   1.5,   0.638),
    ParamDef("kp_y",              0.1,   1.2,   0.447),
    ParamDef("p_softness_x",      1.0,  25.0,   1.66),
    ParamDef("p_softness_y",      1.0,  25.0,  11.50),
    ParamDef("kd_x",              0.0,   0.3,   0.0025),
    ParamDef("kd_y",              0.0,   0.3,   0.005),
    ParamDef("ff_gain",           0.0,   5.0,   0.653),
    ParamDef("predict_frames",    0.0,   6.0,   5.84),
    ParamDef("deadtime_frames",   1.0,   4.0,   3.98),
    ParamDef("oneeuro_min_cutoff",0.001, 0.5,   0.034),
    ParamDef("oneeuro_beta",      0.001, 0.5,   0.008),
    ParamDef("max_step",          3.0,  30.0,   9.92),
    ParamDef("akp_large_thresh",  5.0,  50.0,  24.50),
    ParamDef("akp_small_thresh",  0.5,  10.0,   1.51),
    ParamDef("akp_large_mult",    1.0,   3.0,   1.19),
    ParamDef("akp_small_mult",    0.3,   1.0,   0.97),
    ParamDef("ff_v_ema",          0.01,  0.5,   0.126),
    ParamDef("lead_vgate",        0.5,  20.0,   5.52),
    # 2 new sensitivity-calibration params
    ParamDef("sens_ema",          0.01,  0.3,   0.05),
    ParamDef("sens_init",         0.5,   2.0,   1.0),
]

ALGO_COMBO = {"sem": 1, "cga": 0, "jcad": 1, "pags": 1,
              "label": "SEM+JCAD+PAGS"}

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
# Build AimConfig from parameter vector
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
    cfg.akp_large_thresh = float(params[12])
    cfg.akp_small_thresh = float(params[13])
    cfg.akp_large_mult = float(params[14])
    cfg.akp_small_mult = float(params[15])
    cfg.ff_v_ema = float(params[16])
    cfg.lead_vgate = float(params[17])
    cfg.sem_enabled = float(algo["sem"])
    cfg.cga_enabled = float(algo["cga"])
    cfg.jcad_enabled = float(algo["jcad"])
    cfg.pags_enabled = float(algo["pags"])
    return cfg

# ---------------------------------------------------------------------------
# Modified simulation loop with adaptive sensitivity calibration
# ---------------------------------------------------------------------------
def run_sim_sensitivity(cfg: AimConfig, traj_name: str, seed: int,
                        sens_ema: float, sens_init: float) -> SimResult:
    """Run simulation with online sensitivity estimation.

    After each frame, compares observed screen shift to commanded mouse
    delta.  An EMA-smoothed ratio estimate scales future controller
    outputs so that the effective gain self-corrects when the camera's
    true sensitivity differs from the controller's assumption.
    """
    state = AimState()
    camera = FPSCamera()
    rng = random.Random(seed)
    noise = NoiseModel(seed=seed)
    pipeline = PipelineModel(rng)
    network = NetworkModel(rng)
    tracker = TargetTrackingModel(rng)
    traj_fn = TRAJECTORIES[traj_name]
    result = SimResult(traj_name, "sensitivity", seed, N_FRAMES)

    model_size = 160
    screen_cx = model_size / 2.0
    screen_cy = model_size / 2.0
    scale_x = 1.0
    scale_y = 1.0
    target_class = 0
    prev_sign_x = 0
    prev_sign_y = 0

    # -- Sensitivity calibration state --
    sens_estimate = sens_init   # running estimate of effective sensitivity
    prev_det_x = 0.0           # previous frame detection x
    prev_det_y = 0.0           # previous frame detection y
    prev_dx = 0                # previous frame mouse output x
    prev_dy = 0                # previous frame mouse output y
    prev_missed = True         # was previous frame a miss?
    MIN_COMMAND = 0.5          # ignore tiny commands (noise-dominated)

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

        # -- Sensitivity estimation from previous frame --
        # Only update when both frames had valid detections and we
        # commanded a non-trivial movement.
        if (frame > 0 and not missed and not prev_missed):
            cmd_mag = math.sqrt(prev_dx * prev_dx + prev_dy * prev_dy)
            if cmd_mag > MIN_COMMAND:
                # Observed shift = how much the detection moved.
                # The detection shift includes both ego-pollution (our
                # mouse moved the camera) and target motion.  Since the
                # target's true motion is not known to the controller,
                # we approximate: for the sensitivity *ratio*, the
                # dominant component at high command magnitudes is the
                # ego-corrected shift (the mouse move we caused).
                # In practice the ratio converges because the EMA
                # averages out the target-motion noise.
                obs_shift = math.sqrt((det_x - prev_det_x) ** 2 +
                                      (det_y - prev_det_y) ** 2)
                measured_ratio = obs_shift / cmd_mag
                sens_estimate = ((1.0 - sens_ema) * sens_estimate +
                                 sens_ema * measured_ratio)

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
            frame_cfg.deadtime_frames = pipeline.compute_deadtime_frames(
                frame_period_ms)
            dx, dy = compute_aim_movement(
                det_x, det_y, screen_cx, screen_cy,
                scale_x, scale_y, frame_cfg, state,
                target_class, conf)

            # -- Apply sensitivity correction --
            # scale_adjust compensates for the difference between the
            # expected (init) and estimated (current) sensitivity.
            safe_sens = max(sens_estimate, 0.01)
            scale_adjust = sens_init / safe_sens
            dx = int(round(dx * scale_adjust))
            dy = int(round(dy * scale_adjust))
            # Clamp to HID range
            dx = max(-127, min(127, dx))
            dy = max(-127, min(127, dy))

        # Remember state for next frame's estimation
        prev_det_x = det_x
        prev_det_y = det_y
        prev_dx = dx
        prev_dy = dy
        prev_missed = missed

        camera.apply_mouse(dx, dy)
        result.total_movement += math.sqrt(dx * dx + dy * dy)
        actual_x, actual_y = camera.world_to_screen(world_x, world_y)
        err = math.sqrt((actual_x - screen_cx) ** 2 +
                        (actual_y - screen_cy) ** 2)
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
# Baseline (no sensitivity calibration) using same sim loop for fairness
# ---------------------------------------------------------------------------
def run_sim_baseline(cfg: AimConfig, traj_name: str, seed: int) -> SimResult:
    """Standard sim loop (no sensitivity calibration) -- identical to
    run_sim_custom in opt_cmaes2.py."""
    state = AimState()
    camera = FPSCamera()
    rng = random.Random(seed)
    noise = NoiseModel(seed=seed)
    pipeline = PipelineModel(rng)
    network = NetworkModel(rng)
    tracker = TargetTrackingModel(rng)
    traj_fn = TRAJECTORIES[traj_name]
    result = SimResult(traj_name, "baseline", seed, N_FRAMES)

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
            frame_cfg.deadtime_frames = pipeline.compute_deadtime_frames(
                frame_period_ms)
            dx, dy = compute_aim_movement(
                det_x, det_y, screen_cx, screen_cy,
                scale_x, scale_y, frame_cfg, state,
                target_class, conf)

        camera.apply_mouse(dx, dy)
        result.total_movement += math.sqrt(dx * dx + dy * dy)
        actual_x, actual_y = camera.world_to_screen(world_x, world_y)
        err = math.sqrt((actual_x - screen_cx) ** 2 +
                        (actual_y - screen_cy) ** 2)
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
# Evaluation functions
# ---------------------------------------------------------------------------
def compute_baseline_maes() -> Dict[str, float]:
    """Compute per-trajectory MAE for the production config (no sensitivity)."""
    baseline_cfg = AimConfig()  # defaults = current production
    maes = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_baseline(baseline_cfg, traj, seed)
            errors.extend(r.errors)
        maes[traj] = sum(errors) / len(errors)
    return maes

def evaluate(params: np.ndarray, algo: dict,
             baseline_maes: dict) -> Tuple[float, Dict[str, float]]:
    """Evaluate a candidate with 20 params (18 controller + 2 sensitivity).

    Returns (fitness, per_trajectory_maes).
    """
    cfg = make_config(params, algo)
    sens_ema = float(params[18])
    sens_init = float(params[19])

    traj_maes: Dict[str, float] = {}
    for traj in TRAJECTORIES_LIST:
        errors = []
        for seed in SEEDS:
            r = run_sim_sensitivity(cfg, traj, seed, sens_ema, sens_init)
            errors.extend(r.errors)
        traj_maes[traj] = sum(errors) / len(errors)

    avg_mae = sum(traj_maes.values()) / len(traj_maes)

    # Strict no-regression penalty
    penalty = 0.0
    for traj in TRAJECTORIES_LIST:
        if traj_maes[traj] > baseline_maes[traj] * 1.001:
            excess = ((traj_maes[traj] - baseline_maes[traj]) /
                      baseline_maes[traj])
            penalty += excess * 50.0

    return avg_mae + penalty, traj_maes

# ---------------------------------------------------------------------------
# CMA-ES implementation (pure numpy, same as opt_cmaes2.py)
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, x0: np.ndarray, sigma0: float = 0.5,
                 popsize: int = 20):
        self.dim = len(x0)
        self.mean = x0.copy()
        self.sigma = sigma0

        self.lam = popsize
        self.mu = self.lam // 2

        weights = (np.log(self.mu + 0.5) -
                   np.log(np.arange(1, self.mu + 1)))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        self.cc = ((4 + self.mueff / self.dim) /
                   (self.dim + 4 + 2 * self.mueff / self.dim))
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) /
                       ((self.dim + 2) ** 2 + self.mueff))
        self.damps = (1 + 2 * max(0, math.sqrt(
            (self.mueff - 1) / (self.dim + 1)) - 1) + self.cs)

        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.eigenvalues = np.ones(self.dim)
        self.eigenvectors = np.eye(self.dim)
        self.invsqrt_C = np.eye(self.dim)
        self.gen = 0
        self.eigen_update_gap = max(
            1, int(self.lam / (10 * self.dim)))
        self.chiN = math.sqrt(self.dim) * (
            1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

    def ask(self) -> List[np.ndarray]:
        samples = []
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * (
                self.eigenvectors @ (self.eigenvalues * z))
            samples.append(x)
        return samples

    def tell(self, solutions: List[np.ndarray],
             fitnesses: List[float]):
        idx = np.argsort(fitnesses)
        old_mean = self.mean.copy()

        self.mean = np.zeros(self.dim)
        for i in range(self.mu):
            self.mean += self.weights[i] * solutions[idx[i]]

        diff = self.mean - old_mean
        self.ps = ((1 - self.cs) * self.ps +
                   math.sqrt(self.cs * (2 - self.cs) * self.mueff) *
                   (self.invsqrt_C @ diff) / self.sigma)

        hsig = (np.linalg.norm(self.ps) /
                math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1)))
                / self.chiN) < (1.4 + 2 / (self.dim + 1))
        hsig = float(hsig)

        self.pc = ((1 - self.cc) * self.pc +
                   hsig *
                   math.sqrt(self.cc * (2 - self.cc) * self.mueff) *
                   diff / self.sigma)

        artmp = np.zeros((self.dim, self.mu))
        for i in range(self.mu):
            artmp[:, i] = ((solutions[idx[i]] - old_mean) /
                           self.sigma)

        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) +
                             (1 - hsig) * self.cc *
                             (2 - self.cc) * self.C) +
                  self.cmu *
                  (artmp @ np.diag(self.weights) @ artmp.T))

        self.sigma *= math.exp(
            (self.cs / self.damps) *
            (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = min(self.sigma, 10.0)

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
# Main: CMA-ES optimisation over 20 params
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Experiment: Adaptive Sensitivity Calibration")
    print("=" * 70)
    print(f"Params: {len(PARAM_DEFS)} (18 controller + 2 sensitivity)")
    print(f"Seeds: {len(SEEDS)} | Trajectories: {len(TRAJECTORIES_LIST)}")
    print(f"Evals per fitness: "
          f"{len(SEEDS) * len(TRAJECTORIES_LIST)} sim runs")
    print(f"Algo: {ALGO_COMBO['label']}")
    print()

    # ---- Compute baseline ----
    print("Computing baseline MAEs (production config, no sensitivity)...")
    t0 = time.time()
    baseline_maes = compute_baseline_maes()
    baseline_avg = sum(baseline_maes.values()) / len(baseline_maes)
    print(f"Baseline avg MAE: {baseline_avg:.4f} ({time.time() - t0:.1f}s)")
    for t, m in baseline_maes.items():
        print(f"  {t:15s}: {m:.4f}")
    print()

    # ---- Initial point (unbounded space) ----
    x0_bounded = np.array([pd.init for pd in PARAM_DEFS])
    x0_z = from_bounded(x0_bounded)

    # ---- CMA-ES ----
    POPSIZE = 16
    MAX_GENS = 100
    EARLY_STOP = 50

    es = CMAES(x0_z, sigma0=0.5, popsize=POPSIZE)
    best_fitness = float('inf')
    best_params = None
    best_traj_maes = None
    stagnation = 0
    t_start = time.time()
    history: List[dict] = []

    for gen in range(MAX_GENS):
        solutions = es.ask()
        fitnesses = []

        for sol in solutions:
            bounded = to_bounded(sol)
            f, _ = evaluate(bounded, ALGO_COMBO, baseline_maes)
            fitnesses.append(f)

        es.tell(solutions, fitnesses)

        gen_best = min(fitnesses)
        gen_best_idx = fitnesses.index(gen_best)

        if gen_best < best_fitness:
            best_fitness = gen_best
            best_params = to_bounded(solutions[gen_best_idx]).copy()
            # Re-evaluate best to get per-trajectory breakdown
            _, best_traj_maes = evaluate(
                best_params, ALGO_COMBO, baseline_maes)
            stagnation = 0
        else:
            stagnation += 1

        elapsed = time.time() - t_start
        improv = ((best_fitness - baseline_avg) /
                  baseline_avg * 100)
        history.append({
            "gen": gen,
            "best": round(best_fitness, 6),
            "gen_best": round(gen_best, 6),
            "sigma": round(es.sigma, 6),
        })

        if gen % 5 == 0 or gen_best < best_fitness + 0.001:
            print(f"  Gen {gen:3d} | best={best_fitness:.4f} "
                  f"({improv:+.2f}%) | gen_best={gen_best:.4f} "
                  f"| sigma={es.sigma:.4f} | stag={stagnation} "
                  f"| {elapsed:.0f}s")

        if stagnation >= EARLY_STOP:
            print(f"  Early stop: {stagnation} gens without improvement")
            break

    # ---- Final report ----
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    if best_params is not None:
        improv = ((best_fitness - baseline_avg) /
                  baseline_avg * 100)
        print(f"Best MAE: {best_fitness:.4f} "
              f"(baseline {baseline_avg:.4f}, {improv:+.2f}%)")
        print(f"Generations: {gen + 1} | Time: {total_time:.0f}s")
        print()

        # Sensitivity params
        print(f"Sensitivity params:")
        print(f"  sens_ema  = {best_params[18]:.4f}")
        print(f"  sens_init = {best_params[19]:.4f}")
        print()

        # Per-trajectory detail
        print(f"{'Trajectory':<16} {'Optimized':>9} {'Baseline':>9} "
              f"{'Delta':>9} {'Status':>8}")
        print("-" * 55)
        any_regression = False
        for traj in TRAJECTORIES_LIST:
            opt_mae = best_traj_maes[traj]
            base_mae = baseline_maes[traj]
            delta_pct = (opt_mae - base_mae) / base_mae * 100
            status = "OK" if opt_mae <= base_mae * 1.001 else "REGRESS"
            if status == "REGRESS":
                any_regression = True
            print(f"  {traj:15s} {opt_mae:9.4f} {base_mae:9.4f} "
                  f"{delta_pct:+8.2f}% {status:>8}")

        # All optimized params
        print(f"\nAll parameters:")
        for i, pd in enumerate(PARAM_DEFS):
            print(f"  {pd.name:22s} = {best_params[i]:.6f} "
                  f"(range [{pd.lo}, {pd.hi}])")

        # ---- Save results ----
        result = {
            "experiment": "adaptive_sensitivity_calibration",
            "baseline_avg_mae": baseline_avg,
            "best_mae": best_fitness,
            "improvement_pct": round(improv, 4),
            "any_regression": any_regression,
            "sens_ema": float(best_params[18]),
            "sens_init": float(best_params[19]),
            "params": {pd.name: float(best_params[i])
                       for i, pd in enumerate(PARAM_DEFS)},
            "algo": {k: v for k, v in ALGO_COMBO.items()
                     if k != "label"},
            "per_trajectory": {
                t: {"mae": best_traj_maes[t],
                    "baseline": baseline_maes[t],
                    "improv_pct": round(
                        (best_traj_maes[t] - baseline_maes[t]) /
                        baseline_maes[t] * 100, 4)}
                for t in TRAJECTORIES_LIST
            },
            "baseline_per_trajectory": baseline_maes,
            "generations": gen + 1,
            "population_size": POPSIZE,
            "early_stop_patience": EARLY_STOP,
            "seeds": SEEDS,
            "n_frames": N_FRAMES,
            "elapsed_seconds": round(total_time, 1),
            "convergence_history": history,
        }

        out_path = os.path.join(os.path.dirname(__file__),
                                "experiment_sensitivity_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        print("No valid solution found.")

    print(f"{'='*70}")

if __name__ == "__main__":
    main()
