"""FoundGrasp 독립 grasp_detector.detect()를 직접 사용하는 어댑터.

원본의 폭 보정, GPU collision/refiner 분기, grasp NMS 및 top-50을 유지한다.
실행 계층에서는 하드웨어 폭 한계·ROI·IK를 검사하고 폭 마진을 다시 곱하지 않는다.
"""
import os
import sys
from pathlib import Path


from ..paths import FOUNDGRASP_ROOT, add_vendor_paths, import_grasp_group
from .base import GraspDetector

# 논문 최종 레시피의 추론 설정 (Workspaces/FoundGrasp/docs/PLAN.md "Inference").
# VERIFIER_SAMPLER: 원본 레시피는 triton 이지만 그 커널(scripts/verifier_sample.py)이
# 논문 레포에 커밋되어 있지 않다 → F.grid_sample 경로 (같은 bilinear 샘플링, 속도만 차이).
# VERIFIER_COMPILE: torch.compile 은 첫 호출에 수 분이 걸려 기본 끔 (결과 동일, 속도만 차이).
REFINER_ENV_DEFAULTS = {
    "VERIFIER_SURF": "0.0025",
    "VERIFIER_BUDGET": "1024",
    "VERIFIER_CHUNK": "1024",
    "VERIFIER_PF_EXP": "3",
    "VERIFIER_SELECT": "1",
    "VERIFIER_SELECT_TOP": "256",
    "VERIFIER_SAMPLER": "grid_sample",
    "VERIFIER_COMPILE": "0",
}


class FoundGraspDetector(GraspDetector):
    name = "foundgrasp"

    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        import torch

        add_vendor_paths(FOUNDGRASP_ROOT, FOUNDGRASP_ROOT / "scripts")
        import_grasp_group()  # graspnetAPI 경고 억제 (FoundGrasp가 내부에서 import)
        from grasp_detector import FoundGraspDetector as _Impl

        for key in ("config", "checkpoint"):
            if not Path(cfg[key]).exists():
                raise SystemExit(f"FoundGrasp {key} 파일 없음: {cfg[key]}")
        self.torch = torch
        self.impl = _Impl(cfg["config"], cfg["checkpoint"], device=device)
        self.model = self.impl.model
        self.inf_cfg = self.impl.config.get("inference", {})
        self.collision = bool(cfg.get("collision_detection", True))
        self.collision_voxel = float(cfg.get("collision_voxel_size", 0.01))

        rf = cfg.get("refiner") or {}
        self.refiner = None
        self.refiner_max_w = 0.1  # 독립 detect()의 고정 refiner max_w
        self.last_refiner_ms = None
        self._captured = {}
        if rf.get("enabled", True):
            self._load_refiner(rf, device)
        self.impl.verifier = self.refiner
        self.impl._captured = self._captured

    def _load_refiner(self, rf, device):
        weights = Path(rf["weights"])
        if not weights.exists():
            raise SystemExit(
                f"refiner 가중치 없음: {weights}\n"
                "  refiner 없이 실행: --set detector.foundgrasp.refiner.enabled=false")
        env = dict(REFINER_ENV_DEFAULTS)
        env.update({k: str(v) for k, v in (rf.get("env") or {}).items()})
        os.environ.update(env)  # Verifier 생성·호출 시점에 읽힌다
        import verifier_points

        verifier_points.MODELS = str(weights.parent)  # 원본은 <repo>/logs/_verifier/models 고정
        self.refiner = verifier_points.Verifier(
            weights.stem, device=device, patch=self.model.patch_size, tof=False)
        self.model.decoder.register_forward_hook(
            lambda m, i, o: self._captured.update(vfm=i[0][-1], geo=i[1], dpt=o))
        self.refiner_env = env

    def describe(self):
        if self.refiner is None:
            return "foundgrasp (standalone, GPU CD, grasp NMS, top-50)"
        return f"foundgrasp (standalone +refiner, grasp NMS, top-50, max_w {self.refiner_max_w * 1000:.0f}mm)"

    def detect(self, rgb, depth_m, K):
        self.last_refiner_ms = None
        self.impl.last_verifier_ms = None
        if hasattr(self.refiner, "last_stats"):
            self.refiner.last_stats = None
        # Delegate rather than copy: preserve upstream ordering and defaults.
        # No additional autocast, width adjustment, NMS or candidate truncation.
        with self.torch.no_grad():
            gg = self.impl.detect(
                rgb, depth_m, K, top_k=50, nms=True,
                use_collision_detection=self.collision,
                collision_voxel_size=self.collision_voxel,
                store_vis=False, use_verifier=self.refiner is not None)
        self.last_refiner_ms = self.impl.last_verifier_ms
        stats = getattr(self.refiner, "last_stats", None)
        if stats:
            print(f"  refiner(NMS 전): 입력 {stats['input']} → 측정면 기각 {stats['surface_rejected']}, "
                  f"예산 제외 {stats['budget_skipped']}, 평가 {stats['evaluated']}, "
                  f"평가 후 0점 {stats['zero_after_network']}")
        return gg
