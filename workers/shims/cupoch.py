"""cupoch 최소 대체 모듈 — RNGNet·HGGD 의 dataset/collision_detector.py 가 쓰는 부분만.

  scene_cloud = cupoch.geometry.PointCloud()
  scene_cloud.from_points_dlpack(to_dlpack(points))
  scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
  np.array(scene_cloud.points.cpu())

cupoch 휠이 RTX 5090(sm_120)용 커널을 포함하지 않아 GPU에서 실패할 때만 워커의
PYTHONPATH 앞에 둔다 (configs/default.yaml detector.<rngnet|hggd>.pythonpath).
voxel_down_sample 은 cupoch/Open3D 와 같은 정의: voxel 격자(최소 경계 기준)마다 점 평균.
"""
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack


class _HostPoints:
    def __init__(self, t):
        self._t = t

    def cpu(self):
        return self._t.detach().cpu().numpy()


class PointCloud:
    def __init__(self, points=None):
        self._pts = points if points is not None else torch.zeros((0, 3))

    def from_points_dlpack(self, capsule):
        self._pts = from_dlpack(capsule).reshape(-1, 3).float()
        return True

    def voxel_down_sample(self, voxel_size):
        p = self._pts
        if len(p) == 0:
            return PointCloud(p)
        lo = p.min(0).values - voxel_size * 0.5
        key = torch.floor((p - lo) / voxel_size).long()
        _, inv = torch.unique(key, dim=0, return_inverse=True)
        n = int(inv.max()) + 1
        s = torch.zeros((n, 3), device=p.device, dtype=p.dtype).index_add_(0, inv, p)
        c = torch.zeros((n,), device=p.device, dtype=p.dtype).index_add_(
            0, inv, torch.ones_like(inv, dtype=p.dtype))
        return PointCloud(s / c[:, None])

    @property
    def points(self):
        return _HostPoints(self._pts)


geometry = SimpleNamespace(PointCloud=PointCloud)
__version__ = "shim"
_ = np  # numpy 는 호출측이 np.array 로 감싼다
