from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import GenerationResult, ValidationScore


def _uniform_sample(items: Sequence[Path], n: int) -> List[Path]:
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    out: List[Path] = []
    for i in range(n):
        t = i / float(n - 1)
        idx = int(round(t * (len(items) - 1)))
        out.append(items[idx])
    return out


# ---------------------------------------------------------------------------
# Lazy imports -- heavy deps that may not be installed
# ---------------------------------------------------------------------------

_cv2: Any = None
_np: Any = None
_pybullet: Any = None
_PIL_Image: Any = None


def _ensure_cv2() -> Any:
    global _cv2, _np
    if _cv2 is not None:
        return _cv2
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python-headless and numpy are required: "
            "pip install opencv-python-headless numpy"
        ) from e
    _cv2 = cv2
    _np = np
    return _cv2


def _ensure_pybullet() -> Any:
    global _pybullet
    if _pybullet is not None:
        return _pybullet
    try:
        import pybullet as pb  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pybullet is required for the physics validator: "
            "pip install pybullet"
        ) from e
    _pybullet = pb
    return _pybullet


def _load_frame_gray(path: Path) -> Any:
    """Load a frame as a grayscale numpy array, handling both PPM and common formats."""
    cv2 = _ensure_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read frame: {path}")
    return img


def _load_frame_bgr(path: Path) -> Any:
    cv2 = _ensure_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read frame: {path}")
    return img


# ---------------------------------------------------------------------------
# Sub-score: flow consistency
# ---------------------------------------------------------------------------

def _flow_consistency_score(frames: List[Path]) -> Tuple[float, Dict[str, Any]]:
    """
    Compute dense optical flow between consecutive frames and measure
    directional consistency + magnitude smoothness.

    Returns (score, details_dict).
    """
    cv2 = _ensure_cv2()
    np = _np

    grays = [_load_frame_gray(f) for f in frames]
    if len(grays) < 2:
        return 1.0, {"reason": "too_few_frames"}

    angle_variances: List[float] = []
    mag_ratios: List[float] = []
    prev_median_mag: Optional[float] = None

    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1],
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        fx = flow[..., 0].flatten()
        fy = flow[..., 1].flatten()
        mag = np.sqrt(fx ** 2 + fy ** 2)
        angles = np.arctan2(fy, fx)

        # Only consider pixels with non-trivial motion
        moving = mag > 0.5
        if moving.sum() < 10:
            angle_variances.append(0.0)
            if prev_median_mag is not None:
                mag_ratios.append(1.0)
            prev_median_mag = 0.0
            continue

        moving_angles = angles[moving]
        circ_var = 1.0 - float(np.abs(np.mean(np.exp(1j * moving_angles))))
        angle_variances.append(circ_var)

        median_mag = float(np.median(mag[moving]))
        if prev_median_mag is not None and prev_median_mag > 0.5:
            ratio = max(median_mag, prev_median_mag) / min(median_mag, prev_median_mag)
            mag_ratios.append(ratio)
        prev_median_mag = median_mag

    mean_circ_var = sum(angle_variances) / max(1, len(angle_variances))
    direction_score = max(0.0, 1.0 - mean_circ_var)

    if mag_ratios:
        spike_count = sum(1 for r in mag_ratios if r > 4.0)
        smoothness_score = max(0.0, 1.0 - spike_count / len(mag_ratios))
    else:
        smoothness_score = 1.0

    score = 0.6 * direction_score + 0.4 * smoothness_score

    return float(max(0.0, min(1.0, score))), {
        "mean_circular_variance": round(mean_circ_var, 4),
        "direction_score": round(direction_score, 4),
        "smoothness_score": round(smoothness_score, 4),
        "frame_pairs": len(grays) - 1,
    }


# ---------------------------------------------------------------------------
# Sub-score: structural integrity
# ---------------------------------------------------------------------------

def _structural_integrity_score(frames: List[Path]) -> Tuple[float, Dict[str, Any]]:
    """
    Track large contours across frames and penalise sudden appearances,
    disappearances, or bounding-box collisions.

    Returns (score, details_dict).
    """
    cv2 = _ensure_cv2()
    np = _np

    if len(frames) < 2:
        return 1.0, {"reason": "too_few_frames"}

    def _top_contours(gray: Any, n: int = 8) -> List[Tuple[float, float, float, float]]:
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n]
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((float(x), float(y), float(w), float(h)))
        return boxes

    prev_boxes: Optional[List[Tuple[float, float, float, float]]] = None
    vanish_penalties: List[float] = []
    clip_penalties: List[float] = []

    for f in frames:
        gray = _load_frame_gray(f)
        h_img, w_img = gray.shape[:2]
        min_area = h_img * w_img * 0.005
        boxes = _top_contours(gray)
        boxes = [b for b in boxes if b[2] * b[3] >= min_area]

        if prev_boxes is not None:
            # Vanish penalty: fraction of previous large contours with no nearby match
            if prev_boxes:
                matched = 0
                for px, py, pw, ph in prev_boxes:
                    pcx, pcy = px + pw / 2, py + ph / 2
                    for bx, by, bw, bh in boxes:
                        bcx, bcy = bx + bw / 2, by + bh / 2
                        dist = math.sqrt((pcx - bcx) ** 2 + (pcy - bcy) ** 2)
                        if dist < max(pw, ph, bw, bh) * 1.5:
                            matched += 1
                            break
                vanish_ratio = 1.0 - matched / len(prev_boxes)
                vanish_penalties.append(vanish_ratio)

            # Clip penalty: new overlaps between boxes that didn't overlap before
            def _iou(a: Tuple, b: Tuple) -> float:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                iy = max(0, min(ay + ah, by + bh) - max(ay, by))
                inter = ix * iy
                union = aw * ah + bw * bh - inter
                return inter / union if union > 0 else 0.0

            if len(boxes) >= 2:
                overlap_count = 0
                pair_count = 0
                for i_a in range(len(boxes)):
                    for i_b in range(i_a + 1, len(boxes)):
                        pair_count += 1
                        if _iou(boxes[i_a], boxes[i_b]) > 0.15:
                            overlap_count += 1
                clip_penalties.append(overlap_count / max(1, pair_count))

        prev_boxes = boxes

    mean_vanish = sum(vanish_penalties) / max(1, len(vanish_penalties)) if vanish_penalties else 0.0
    mean_clip = sum(clip_penalties) / max(1, len(clip_penalties)) if clip_penalties else 0.0

    vanish_score = max(0.0, 1.0 - mean_vanish * 2.0)
    clip_score = max(0.0, 1.0 - mean_clip * 3.0)
    score = 0.6 * vanish_score + 0.4 * clip_score

    return float(max(0.0, min(1.0, score))), {
        "mean_vanish_ratio": round(mean_vanish, 4),
        "mean_clip_ratio": round(mean_clip, 4),
        "vanish_score": round(vanish_score, 4),
        "clip_score": round(clip_score, 4),
    }


# ---------------------------------------------------------------------------
# Sub-score: motion plausibility (PyBullet particle simulation)
# ---------------------------------------------------------------------------

def _motion_plausibility_score(
    frames: List[Path],
) -> Tuple[float, Dict[str, Any]]:
    """
    Track sparse feature points across frames with Lucas-Kanade, seed a
    PyBullet particle simulation with observed initial velocities, step
    forward and compare predicted vs observed positions.

    Returns (score, details_dict).
    """
    cv2 = _ensure_cv2()
    np = _np
    pb = _ensure_pybullet()

    if len(frames) < 3:
        return 1.0, {"reason": "too_few_frames"}

    gray0 = _load_frame_gray(frames[0])
    h_img, w_img = gray0.shape[:2]

    corners = cv2.goodFeaturesToTrack(gray0, maxCorners=50, qualityLevel=0.05, minDistance=10)
    if corners is None or len(corners) < 4:
        return 1.0, {"reason": "insufficient_features", "features_found": 0 if corners is None else len(corners)}

    grays = [gray0] + [_load_frame_gray(f) for f in frames[1:]]

    # Track points across all frames
    trajectories: List[List[Optional[Tuple[float, float]]]] = []
    for pt in corners:
        trajectories.append([(float(pt[0][0]), float(pt[0][1]))])

    prev_gray = grays[0]
    prev_pts = corners.copy()
    active = list(range(len(corners)))

    for fi in range(1, len(grays)):
        if len(active) < 2:
            break
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, grays[fi], prev_pts, None)
        new_active = []
        new_prev = []
        for idx_in_active, orig_idx in enumerate(active):
            if status[idx_in_active][0] == 1:
                x, y = float(next_pts[idx_in_active][0][0]), float(next_pts[idx_in_active][0][1])
                if 0 <= x < w_img and 0 <= y < h_img:
                    trajectories[orig_idx].append((x, y))
                    new_active.append(orig_idx)
                    new_prev.append(next_pts[idx_in_active])
                    continue
            trajectories[orig_idx].append(None)

        active = new_active
        prev_pts = np.array(new_prev, dtype=np.float32) if new_prev else np.empty((0, 1, 2), dtype=np.float32)
        prev_gray = grays[fi]

    # Filter to trajectories that survived at least 60% of frames
    min_len = max(3, int(len(grays) * 0.6))
    good: List[List[Tuple[float, float]]] = []
    for traj in trajectories:
        valid = [p for p in traj if p is not None]
        if len(valid) >= min_len:
            good.append(valid)

    if len(good) < 3:
        return 1.0, {"reason": "insufficient_tracks", "good_tracks": len(good)}

    # Setup PyBullet in DIRECT mode (no GUI)
    physics_client = pb.connect(pb.DIRECT)
    try:
        pb.setGravity(0, 0, 0, physicsClientId=physics_client)
        pb.setTimeStep(1.0 / 8.0, physicsClientId=physics_client)

        # Normalise pixel coords to a ~1m world
        scale = 1.0 / max(w_img, h_img)

        divergences: List[float] = []
        for traj in good:
            if len(traj) < 3:
                continue
            x0, y0 = traj[0]
            x1, y1 = traj[1]
            vx = (x1 - x0) * scale
            vy = (y1 - y0) * scale

            col_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.002, physicsClientId=physics_client)
            body_id = pb.createMultiBody(
                baseMass=0.001,
                baseCollisionShapeIndex=col_id,
                basePosition=[x0 * scale, y0 * scale, 0],
                physicsClientId=physics_client,
            )
            pb.resetBaseVelocity(body_id, [vx, vy, 0], physicsClientId=physics_client)

            traj_divs: List[float] = []
            for ti in range(2, len(traj)):
                pb.stepSimulation(physicsClientId=physics_client)
                pos, _ = pb.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
                pred_x, pred_y = pos[0] / scale, pos[1] / scale
                obs_x, obs_y = traj[ti]
                dist = math.sqrt((pred_x - obs_x) ** 2 + (pred_y - obs_y) ** 2)
                traj_divs.append(dist)

            pb.removeBody(body_id, physicsClientId=physics_client)

            if traj_divs:
                divergences.append(sum(traj_divs) / len(traj_divs))

    finally:
        pb.disconnect(physics_client)

    if not divergences:
        return 1.0, {"reason": "no_divergences_computed"}

    mean_div = sum(divergences) / len(divergences)
    norm_scale = max(w_img, h_img) * 0.15
    score = float(math.exp(-mean_div / norm_scale))

    return max(0.0, min(1.0, score)), {
        "mean_divergence_px": round(mean_div, 2),
        "tracks_evaluated": len(divergences),
        "norm_scale": round(norm_scale, 2),
    }


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class PyBulletPhysicsValidator:
    """
    Physics validator combining optical-flow analysis with PyBullet particle
    simulation.  Produces three sub-scores:

    * **flow_consistency** (weight 0.4) -- directional smoothness of dense
      optical flow across frames.
    * **structural_integrity** (weight 0.3) -- stability of large contours
      (no sudden vanishing / clipping).
    * **motion_plausibility** (weight 0.3) -- divergence between observed
      sparse-point trajectories and a zero-gravity PyBullet particle sim
      seeded with initial velocities.

    All dependencies (opencv-python-headless, numpy, pybullet) are **required**.
    Missing packages cause a hard error -- install them before running.
    """

    name = "physics"

    def __init__(
        self,
        *,
        n_frames: int = 16,
        flow_weight: float = 0.4,
        structure_weight: float = 0.3,
        motion_weight: float = 0.3,
    ) -> None:
        self._n_frames = n_frames
        self._flow_w = flow_weight
        self._structure_w = structure_weight
        self._motion_w = motion_weight

    def score(self, generation: GenerationResult) -> ValidationScore:
        if not generation.frames or len(generation.frames) < 3:
            return ValidationScore(
                name=self.name, score=0.5,
                feedback="Too few frames to validate physics",
            )

        _ensure_cv2()
        _ensure_pybullet()

        sampled = _uniform_sample(generation.frames, self._n_frames)

        # --- sub-scores ---
        flow_score, flow_details = _flow_consistency_score(sampled)
        struct_score, struct_details = _structural_integrity_score(sampled)
        motion_score, motion_details = _motion_plausibility_score(sampled)

        # --- composite ---
        total_w = self._flow_w + self._structure_w + self._motion_w
        composite = (
            self._flow_w * flow_score
            + self._structure_w * struct_score
            + self._motion_w * motion_score
        ) / total_w
        composite = float(max(0.0, min(1.0, composite)))

        # --- feedback ---
        issues: List[str] = []
        if flow_score < 0.5:
            issues.append("motion inconsistency (teleportation / direction reversal)")
        if struct_score < 0.5:
            issues.append("structural instability (boundaries shifting / clipping)")
        if motion_score < 0.5:
            issues.append("physically implausible trajectories")
        feedback = "; ".join(issues) if issues else ""

        details: Dict[str, Any] = {
            "flow": {"score": round(flow_score, 4), **flow_details},
            "structure": {"score": round(struct_score, 4), **struct_details},
            "motion": {"score": round(motion_score, 4), **motion_details},
            "weights": {
                "flow": self._flow_w,
                "structure": self._structure_w,
                "motion": self._motion_w,
            },
            "n_frames_sampled": len(sampled),
        }

        return ValidationScore(
            name=self.name,
            score=composite,
            details=details,
            feedback=feedback,
        )
