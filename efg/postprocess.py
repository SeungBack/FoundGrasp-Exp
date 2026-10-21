"""실행 계층 — 하드웨어 한계와 방법별 출력 계약을 적용.

논문 시뮬 벤치마크(Workspaces/FoundGrasp/sim/run_clutter_removal.py, ext/BASELINES.md
"width 규약")는 비교군에 적용한다. FoundGrasp 독립 detect() 출력에는
추가 공유 충돌 검사와 폭 마진을 적용하지 않는다.

  입력  ROI 밖 depth = 0 (전 방법 동일 입력. 비교군 점군 계열의 workspace crop 과 같은 의미)
  1) 폭 한계 필터   raw width <= max_open_m (한계 밖 후보는 클립 실행이 아니라 배제)
  2) 공유 충돌필터  ModelFreeCollisionDetector(voxel 0.01, approach 0.05, thresh 0.01),
                   장면 점군은 마스킹 전 전체 depth (ROI 밖 박스 벽도 장애물로 본다)
  3) score 내림차순 정렬 → (호출자) ROI 투영 필터 → 선정(기하·IK, 순위 순)
  실행  개방 폭 = min(open_multiplier x raw width, max_open_m)
"""
import numpy as np

from .paths import FOUNDGRASP_ROOT, add_vendor_paths, import_grasp_group

def uses_standalone_foundgrasp(cfg):
    return cfg.get("detector", {}).get("name") == "foundgrasp"


CD_CHUNK = 256  # 후보 묶음 크기 — 후보별 독립 계산이라 결과 동일, 메모리만 제한


def mask_depth_to_roi(depth_m, roi_px):
    """ROI 밖 depth를 0(무효)으로. roi_px=None이면 그대로."""
    if roi_px is None:
        return depth_m
    x1, y1, x2, y2 = [int(v) for v in roi_px]
    out = np.zeros_like(depth_m)
    out[y1:y2, x1:x2] = depth_m[y1:y2, x1:x2]
    return out


def backproject(depth_m, K):
    """depth (H,W) → 유효 픽셀의 카메라 좌표 점군 (N,3) float32."""
    H, W = depth_m.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth_m.astype(np.float32)
    valid = z > 0
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    return np.stack([x[valid], y[valid], z[valid]], axis=1)


def _subset(gg, mask):
    GraspGroup = import_grasp_group()
    return GraspGroup(gg.grasp_group_array[np.asarray(mask, dtype=bool)].copy())


def shared_filter(gg, depth_full, K, cfg):
    """폭 한계 → 공유 충돌필터 → score 정렬. 반환: (GraspGroup, 단계별 개수 dict)."""
    pp = cfg["postprocess"]
    counts = {"detected": len(gg)}
    rejected_score = 0
    if len(gg) and uses_standalone_foundgrasp(cfg):
        # Rejected zero-score grasps may survive upstream top-k; never execute them.
        rejected_score = int(np.count_nonzero(~(gg.scores > 0)))
        gg = _subset(gg, gg.scores > 0)
    if len(gg):
        limit = float(pp["max_open_m"])
        widths = np.asarray(gg.widths)
        keep = widths <= limit + 1e-6
        if rejected_score or not keep.all():
            values, frequency = np.unique(np.round(widths[np.isfinite(widths)] * 1000, 1),
                                          return_counts=True)
            common = np.argsort(-frequency, kind="stable")[:6]
            summary = ", ".join(f"{values[i]:g}mm×{frequency[i]}" for i in common)
            print(f"  폭 필터: 한계 {limit * 1000:g}mm, 초과/무효 {np.count_nonzero(~keep)}개, "
                  f"score 기각 {rejected_score}개 | 유효 score 후보 폭(빈도 상위): {summary}")
        gg = _subset(gg, keep)
    elif rejected_score:
        print(f"  후보 필터: score 기각 {rejected_score}개 (폭 검사 전)")
    counts["width_ok"] = len(gg)
    cd = pp.get("collision") or {}
    if len(gg) and cd.get("enabled", True) and not uses_standalone_foundgrasp(cfg):
        add_vendor_paths(FOUNDGRASP_ROOT)
        from utils.collision_detector import ModelFreeCollisionDetector

        # Open3D's Vector3dVector takes a slow element-wise conversion path for
        # float32 arrays. Pre-convert once to contiguous float64 (same values).
        scene = np.ascontiguousarray(backproject(depth_full, K), dtype=np.float64)
        mfc = ModelFreeCollisionDetector(scene,
                                         voxel_size=float(cd.get("voxel_size_m", 0.01)))
        backend = cd.get("backend", "auto")
        if backend not in ("auto", "cpu", "cuda"):
            raise ValueError(f"알 수 없는 collision backend: {backend}")
        gpu = None
        if backend != "cpu":
            import torch
            if torch.cuda.is_available():
                from utils.collision_detector import ModelFreeCollisionDetectorGPU
                # Keep the original Open3D voxel centroids and float64 arithmetic.
                # The vendor GPU constructor's own voxelization uses another grid.
                gpu = ModelFreeCollisionDetectorGPU(
                    torch.empty((0, 3), device="cuda", dtype=torch.float64),
                    voxel_size=mfc.voxel_size, device="cuda")
                gpu.scene_points = torch.as_tensor(mfc.scene_points, device="cuda",
                                                  dtype=torch.float64)
                gpu.num_points = len(mfc.scene_points)
            elif backend == "cuda":
                raise RuntimeError("collision.backend=cuda 이지만 CUDA를 사용할 수 없습니다")
        GraspGroup = import_grasp_group()
        hit = np.zeros(len(gg), dtype=bool)
        for i in range(0, len(gg), CD_CHUNK):
            part = GraspGroup(gg.grasp_group_array[i:i + CD_CHUNK])
            params = dict(approach_dist=float(cd.get("approach_dist_m", 0.05)),
                          collision_thresh=float(cd.get("thresh", 0.01)))
            if gpu is None:
                hit[i:i + CD_CHUNK] = mfc.detect(part, **params)
            else:
                hit[i:i + CD_CHUNK] = gpu.detect_soft(
                    torch.as_tensor(part.grasp_group_array, device="cuda",
                                    dtype=torch.float64), **params)
        gg = _subset(gg, ~hit)
    counts["collision_free"] = len(gg)
    if len(gg):
        gg = gg.sort_by_score()
    return gg, counts


def open_width_m(raw_width_m, cfg):
    """실행 개방 폭 = min(open_multiplier x raw, max_open_m)."""
    pp = cfg["postprocess"]
    multiplier = 1.0 if uses_standalone_foundgrasp(cfg) else float(pp.get("open_multiplier", 1.2))
    return float(min(multiplier * float(raw_width_m),
                     float(pp["max_open_m"])))
