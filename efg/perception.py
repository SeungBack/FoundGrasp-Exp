"""One RGB-D inference/filter path shared by live operation and offline replay."""
from dataclasses import dataclass
import time

import numpy as np

from .postprocess import mask_depth_to_roi, shared_filter
from .roi import filter_roi


@dataclass
class DetectionFrame:
    grasps: object
    counts: dict
    detect_ms: float
    post_ms: float
    inference_ms: float | None
    raw_grasps: np.ndarray | None = None


def validate_frame(rgb, depth_m, K):
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or depth_m.shape != rgb.shape[:2] or K.shape != (3, 3)
            or not np.issubdtype(depth_m.dtype, np.floating)
            or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError("Expected RGB uint8, aligned depth metres and valid 3x3 K")
    if not np.isfinite(depth_m).all() or (depth_m < 0).any():
        raise ValueError("Depth must contain finite metres; invalid pixels must be zero")


def detect_frame(detector, rgb, depth_m, K, cfg, roi=None):
    """Inference timing excludes shared postprocessing; worker IPC is explicit."""
    validate_frame(rgb, depth_m, K)
    depth_in = (mask_depth_to_roi(depth_m, roi)
                if cfg['postprocess'].get('roi_depth_mask', True) else depth_m)
    started = time.perf_counter()
    gg = detector.detect(rgb, depth_in, K)
    detect_ms = (time.perf_counter() - started) * 1000
    post_started = time.perf_counter()
    arr = gg.grasp_group_array
    if arr.ndim != 2 or arr.shape[1] != 17:
        raise ValueError(f"Detector returned invalid GraspGroup shape: {arr.shape}")
    # Block malformed geometry before it can reach IK or robot motion.
    if len(arr):
        R = arr[:, 4:13].reshape(-1, 3, 3)
        if (not np.isfinite(arr).all() or (arr[:, 1:4] < 0).any()
                or not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=.01)
                or not np.allclose(np.linalg.det(R), 1., atol=.01)):
            raise ValueError("Detector returned nonfinite/negative dimensions or invalid rotations")
    raw = arr.copy()
    gg, counts = shared_filter(gg, depth_m, K, cfg)
    gg = filter_roi(gg, K, roi)
    counts['roi'] = len(gg)
    return DetectionFrame(gg, counts, detect_ms, (time.perf_counter() - post_started) * 1000,
                          getattr(detector, 'last_inference_ms', None), raw)
