#!/usr/bin/env python3
"""Multi-target SELECTION sim: extends aim_sim_ring.py (which models only the
controller on one pre-selected target) with the real per-frame target-selection
pipeline ported line-for-line from simple_postprocess.cu:

  - conf_threshold gating + allowed classes (decodeDetectionIfValid)
  - nearest-to-crosshair distance score (aim-point aware)
  - IoU + distance stickiness to the previous target (computeStickinessScore)
  - stickyMatch vs nearest arbitration (iou_stickiness_threshold)
  - track_persistence_frames gap bridging
  - head/body box selection + head_aim_point/body_aim_point aim geometry

This makes the DETECTION-side knobs testable that the controller-only sim could
not judge: conf_threshold, iou_stickiness_threshold, distance_stickiness_factor,
track_persistence_frames, head_aim_point, body_aim_point. Scenes contain TWO
enemies so target-switching actually happens.

Boxes/coords are in the 320 model frame; measurement = enemy_abs - crosshair + SC
with dead time d_true, exactly like the controller loop in aim_sim_ring.
"""
import math
import random
from aim_sim_ring import Ctrl, Cfg, SC

# ------------------------------------------------------------ box geometry (model px)
BODY_W, BODY_H = 22.0, 80.0
HEAD_W, HEAD_H = 12.0, 18.0
HEAD_DY = 40.0            # head-box center this far above the body-box center

class Det:
    __slots__ = ("cls", "x1", "y1", "x2", "y2", "conf", "eid")
    def __init__(self, cls, x1, y1, x2, y2, conf, eid=-1):
        self.cls, self.x1, self.y1, self.x2, self.y2, self.conf, self.eid = \
            cls, x1, y1, x2, y2, conf, eid

# ------------------------------------------------------------ ported selection math
def iou(a, b):
    if a is None or b is None:
        return 0.0
    aw, ah, bw, bh = a.x2 - a.x1, a.y2 - a.y1, b.x2 - b.x1, b.y2 - b.y1
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    uni = aw * ah + bw * bh - inter
    return inter / uni if uni > 0 else 0.0

def stickiness_score(det, prev, distance_factor):
    if prev is None or prev.cls < 0 or det.cls < 0:
        return 0.0
    i = iou(det, prev)
    if distance_factor <= 0.0 or det.cls != prev.cls:
        return i
    pw, ph = prev.x2 - prev.x1, prev.y2 - prev.y1
    if pw <= 0 or ph <= 0:
        return i
    prev_diag_sq = pw * pw + ph * ph
    window_sq = max(prev_diag_sq * distance_factor * distance_factor, 1.0)
    dcx = (det.x1 + det.x2) * 0.5 - (prev.x1 + prev.x2) * 0.5
    dcy = (det.y1 + det.y2) * 0.5 - (prev.y1 + prev.y2) * 0.5
    dist_sq = dcx * dcx + dcy * dcy
    if dist_sq >= window_sq:
        return i
    dist_score = 1.0 - math.sqrt(dist_sq / window_sq)
    return max(i, dist_score)

def aim_point(det, head_cls, head_ap, body_ap):
    h = det.y2 - det.y1
    cx = (det.x1 + det.x2) * 0.5
    cy = (det.y1 + h * head_ap) if det.cls == head_cls else (det.y1 + h * body_ap)
    return cx, cy

def select(boxes, prev, prev_frames_since_seen, cfg, head_cls, head_ap, body_ap):
    """Returns (chosen Det | None, is_gap). Ports stage1+stage2 selection.

    Two CANDIDATE knobs (both default off => shipped behaviour):
      cfg.class_switch_penalty : scale the squared distance of a candidate whose
          class differs from the currently tracked class by (1+p)^2. Hysteresis:
          switching anchor (head<->body) now needs the other class to be clearly
          closer, instead of letting detector noise flip it every frame.
      cfg.head_min_box_px : a head-class box shorter than this is not eligible as
          the aim target (falls back to the body box). Precision gate - the rig
          data shows a small/low-conf head box has ~8x the jitter of a big one.
    """
    penalty = getattr(cfg, "class_switch_penalty", 0.0)
    head_min = getattr(cfg, "head_min_box_px", 0.0)
    best_dist = None; best_dist_val = float("inf")
    best_iou = None; best_iou_val = -1.0
    prev_valid = prev is not None and prev.cls >= 0
    for det in boxes:
        # precision gate: an unreliably small head box is not an aim candidate
        if head_min > 0.0 and det.cls == head_cls and (det.y2 - det.y1) < head_min:
            continue
        cx, cy = aim_point(det, head_cls, head_ap, body_ap)
        d = (cx - SC) ** 2 + (cy - SC) ** 2       # distance to crosshair (aim-point aware)
        if penalty > 0.0 and prev_valid and det.cls != prev.cls:
            k = 1.0 + penalty
            d *= k * k                            # hysteresis against class flip-flop
        if d < best_dist_val:
            best_dist_val = d; best_dist = det
        if prev_valid:
            s = stickiness_score(det, prev, cfg.distance_stickiness_factor)
            if s > best_iou_val:
                best_iou_val = s; best_iou = det
    has_dist = best_dist is not None
    has_iou = best_iou is not None
    sticky = has_iou and best_iou_val > cfg.iou_stickiness_threshold
    if sticky:
        return best_iou, False
    if has_dist:
        return best_dist, False
    if prev_valid and prev_frames_since_seen < cfg.track_persistence_frames:
        return None, True        # gap within persistence window -> bridge
    return None, False           # fully lost


# ------------------------------------------------------------ detector (boxes per enemy)
def make_boxes(fx, fy, rng, cfg_det):
    """Emit body/head boxes (in frame coords) for an enemy centred at (fx,fy).

    Per-class sigmas are the MEASURED rig values (aim-point space, first-
    difference/sqrt2 from bench/calib_rig_sample_v*.csv), not assumed:
        body            sigma X 6.98 / Y 4.96   (box ~33x68, always present)
        head near/big   sigma X 0.89 / Y 1.34   (box ~14x17, conf .51)
        head far/small  sigma X 10.7 / Y 6.72   (box ~9x11,  conf .36)
    conf_threshold is applied by the caller (matches decodeDetectionIfValid)."""
    out = []
    if rng.random() < cfg_det["p_body"]:
        conf = _clamp(cfg_det["body_conf"] + rng.gauss(0, 0.08), 0.05, 0.99)
        cx = fx + rng.gauss(0, cfg_det["body_sx"])
        cy = fy + rng.gauss(0, cfg_det["body_sy"])
        out.append((Det(0, cx - BODY_W/2, cy - BODY_H/2,
                        cx + BODY_W/2, cy + BODY_H/2, conf), conf))
    if rng.random() < cfg_det["p_head"]:
        conf = _clamp(cfg_det["head_conf"] + rng.gauss(0, 0.08), 0.05, 0.99)
        hh = cfg_det["head_h"]
        hw = hh * 0.8
        hx = fx + rng.gauss(0, cfg_det["head_sx"])
        hy = (fy - HEAD_DY) + rng.gauss(0, cfg_det["head_sy"])
        out.append((Det(1, hx - hw/2, hy - hh/2, hx + hw/2, hy + hh/2, conf), conf))
    return out

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def det_cfg(regime="near", **kw):
    """Measured rig noise. regime 'near' = big/confident head box (sigma ~1px),
    'far' = small/low-conf head box (sigma ~10px). Body is the same either way."""
    if regime == "far":
        head = dict(head_h=11.0, head_sx=10.73, head_sy=6.72, head_conf=0.36)
    else:
        head = dict(head_h=17.0, head_sx=0.89, head_sy=1.34, head_conf=0.51)
    b = dict(body_sx=6.98, body_sy=4.96, body_conf=0.46,
             p_body=0.97, p_head=0.90, blur=0.0, p_outlier=0.02,
             out_lo=15.0, out_hi=50.0)
    b.update(head); b.update(kw); return b


# ------------------------------------------------------------ scenes
def scene_targets(scene, f, frames):
    """Return list of (eid, ex, ey) enemy body-centres (absolute) at frame f, and
    the intended-target eid (the one the user wants: enemy 0)."""
    if scene == "headbody":
        # single enemy, stationary-ish near crosshair: isolates conf/aim-point shake
        return [(0, SC + 8.0, SC + 30.0)], 0
    if scene == "duel_cross":
        # enemy0 (intended) drifts slowly; enemy1 sweeps across and passes CLOSE to
        # enemy0 mid-scene -> nearest-to-crosshair can flip to enemy1 (tests sticky).
        e0x = SC + 10.0 + 6.0 * math.sin(f * 0.02)
        e0y = SC + 25.0
        t = f / frames
        e1x = SC - 80.0 + 200.0 * t            # sweeps left->right through e0
        e1y = SC + 25.0 + 8.0 * math.sin(f * 0.05)
        return [(0, e0x, e0y), (1, e1x, e1y)], 0
    if scene == "chase":
        # enemy0 (intended) strafes steadily; a STATIONARY distractor enemy1 sits
        # ahead on the path. As the crosshair chases the moving e0 past e1, e1
        # briefly becomes nearest-to-crosshair -> nearest picks the wrong (still)
        # target and stays stuck on it while e0 leaves. Stickiness (esp. distance,
        # since a fast target's boxes don't overlap frame-to-frame) should keep
        # the aim on e0. This is the case stickiness is designed for.
        span = 150.0
        t = (f % int(span)) / span
        e0x = SC - 55.0 + 150.0 * t
        e0y = SC + 25.0
        e1x, e1y = SC + 45.0, SC + 25.0
        return [(0, e0x, e0y), (1, e1x, e1y)], 0
    if scene == "occl_distractor":
        # enemy0 intended; a distractor enemy1 sits ~35px away. enemy0 is occluded
        # in bursts -> does aim hold+re-acquire e0 (persistence+sticky) or jump to e1?
        e0x, e0y = SC + 6.0, SC + 25.0
        e1x, e1y = SC + 42.0, SC + 25.0
        return [(0, e0x, e0y), (1, e1x, e1y)], 0
    return [(0, SC, SC)], 0

def is_occluded(scene, eid, f):
    if scene == "occl_distractor" and eid == 0:
        return (f % 45) < 5           # enemy0 hidden 5 of every 45 frames
    return False


# ------------------------------------------------------------ closed loop
def run(cfg, scene, frames=600, seed=0, d_true=2, conf_threshold=0.15,
        head_ap=1.0, body_ap=0.15, head_cls=1, dcfg=None):
    rng = random.Random(seed)
    dcfg = dcfg or det_cfg()
    ctrl = Ctrl(cfg); ctrl.reset()
    C = [SC, SC]
    prev_sel = None
    prev_frames_since_seen = 0
    hist = []                       # (enemies list, Cx, Cy) per frame
    errs = []; yerrs = []
    chosen_eid_hist = []
    switch_count = 0
    prev_chosen_eid = None
    # CLASS (anchor) switching - the thing the rig CSVs showed costs 27-36% of Y
    # variance. Counted separately from enemy switching, and we record the
    # aim-point jump at each flip so the sim can be checked against the rig
    # (measured: 0.6-2.4% of frames, |dY| mean 11-21px).
    cls_switches = 0
    prev_cls = None
    prev_aim = None
    cls_jumps = []

    for f in range(frames):
        enemies, intended = scene_targets(scene, f, frames)
        hist.append((enemies, C[0], C[1]))

        # dead-time: selection sees the scene from d_true frames ago (crosshair too)
        idx = max(0, f - d_true)
        past_enemies, pastCx, pastCy = hist[idx]

        # build detections (frame coords) for all visible enemies, conf-gated
        boxes = []
        for (eid, ex, ey) in past_enemies:
            if is_occluded(scene, eid, idx):
                continue
            fx = ex - pastCx + SC
            fy = ey - pastCy + SC
            for det, conf in make_boxes(fx, fy, rng, dcfg):
                det.eid = eid
                if conf > conf_threshold:            # decodeDetectionIfValid gate
                    boxes.append(det)
        # rare gross outlier box (false positive) near crosshair, low conf
        if rng.random() < dcfg["p_outlier"]:
            oc = _clamp(0.38 + rng.gauss(0, 0.08), 0.05, 0.7)
            if oc > conf_threshold:
                ox = SC + rng.gauss(0, 20); oy = SC + rng.gauss(0, 20)
                boxes.append(Det(0, ox - BODY_W/2, oy - BODY_H/2, ox + BODY_W/2, oy + BODY_H/2, oc, eid=99))

        chosen, is_gap = select(boxes, prev_sel, prev_frames_since_seen, cfg,
                                head_cls, head_ap, body_ap)

        if chosen is not None:
            cx, cy = aim_point(chosen, head_cls, head_ap, body_ap)
            if prev_cls is not None and chosen.cls != prev_cls:
                cls_switches += 1
                if prev_aim is not None:
                    cls_jumps.append(abs(cy - prev_aim))
            prev_cls = chosen.cls
            prev_aim = cy
            dx, dy = ctrl.step(cx, cy, SC, SC)
            prev_sel = chosen
            prev_frames_since_seen = 0
            chosen_eid = chosen.eid
        elif is_gap:
            # bridge: hold, keep prev_sel alive, advance persistence counter
            dx, dy = 0, 0
            ctrl.res_x = ctrl.res_y = 0.0
            prev_frames_since_seen += 1
            chosen_eid = None
        else:
            dx, dy = 0, 0
            ctrl.res_x = ctrl.res_y = 0.0
            ctrl.has_track = 0
            prev_sel = None
            prev_frames_since_seen = 0
            chosen_eid = None

        C[0] += dx; C[1] += dy

        # metric vs intended enemy's head aim point (absolute)
        ie = next((e for e in enemies if e[0] == intended), None)
        if ie is not None:
            # TRUE head aim point (abs): head box centre sits HEAD_DY above the
            # body centre; y1 = c - h/2, aimY = y1 + h*head_ap. h from the regime.
            hh = dcfg["head_h"]
            aim_y = (ie[2] - HEAD_DY - hh * 0.5) + hh * head_ap
            e = math.hypot(ie[1] - C[0], aim_y - C[1])
            errs.append(e); yerrs.append(abs(aim_y - C[1]))
        if chosen_eid is not None:
            if prev_chosen_eid is not None and chosen_eid != prev_chosen_eid:
                switch_count += 1
            prev_chosen_eid = chosen_eid
        chosen_eid_hist.append(chosen_eid)

    w0 = 60
    rms = math.sqrt(sum(x*x for x in errs[w0:]) / max(1, len(errs) - w0))
    yrms = math.sqrt(sum(x*x for x in yerrs[w0:]) / max(1, len(yerrs) - w0))
    on_intended = [1 for e in chosen_eid_hist[w0:] if e == 0]
    on_wrong = [1 for e in chosen_eid_hist[w0:] if e is not None and e != 0]
    n = len(chosen_eid_hist[w0:])
    return dict(rms=rms, yrms=yrms, switches=switch_count,
                wrong_pct=100.0 * len(on_wrong) / max(1, n),
                gap_pct=100.0 * (n - len(on_intended) - len(on_wrong)) / max(1, n),
                cls_sw_pct=100.0 * cls_switches / max(1, frames),
                cls_jump=(sum(cls_jumps) / len(cls_jumps)) if cls_jumps else 0.0)

def avg(cfg, scene, n=6, **kw):
    acc = {}
    for s in range(n):
        m = run(cfg, scene, seed=s, **kw)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}
