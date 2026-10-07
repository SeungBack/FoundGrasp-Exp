"""FoundGrasp 백엔드 — third_party/FoundGrasp 사본을 import (원본 무수정)."""
from ..paths import FOUNDGRASP_ROOT, add_vendor_paths, import_grasp_group
from .base import GraspDetector


class FoundGraspDetector(GraspDetector):
    name = "foundgrasp"

    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        add_vendor_paths(FOUNDGRASP_ROOT)
        import_grasp_group()  # graspnetAPI 경고 억제 (FoundGrasp가 내부에서 import)
        from grasp_detector import FoundGraspDetector as _Impl

        self.top_k = int(cfg.get("top_k", 50))
        self.nms = bool(cfg.get("nms", True))
        self.collision = bool(cfg.get("collision_detection", True))
        self.voxel = float(cfg.get("collision_voxel_size", 0.01))
        self.impl = _Impl(cfg["config"], cfg["checkpoint"], device=device)

    def detect(self, rgb, depth_m, K, top_k=None):
        return self.impl.detect(
            rgb, depth_m, K,
            top_k=top_k or self.top_k, nms=self.nms,
            use_collision_detection=self.collision,
            collision_voxel_size=self.voxel,
        )
