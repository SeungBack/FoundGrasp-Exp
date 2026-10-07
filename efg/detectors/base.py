"""파지 검출 백엔드 인터페이스 — 비교군은 이 클래스를 구현해 등록한다.

계약:
  입력  rgb (H,W,3) uint8 RGB / depth_m (H,W) float32 미터(무효=0) / K (3,3)
  출력  graspnetAPI.grasp.GraspGroup, **카메라 color optical frame**,
        score 내림차순. 그립 프레임 규약(graspnetAPI): 회전행렬 1열=접근 방향,
        2열=핑거 개폐 방향. width·depth는 미터.
이 규약만 지키면 이후 단계(ROI 필터·base 변환·선정·모션)는 그대로 쓰인다.
"""
from abc import ABC, abstractmethod


class GraspDetector(ABC):
    name = "base"

    def __init__(self, cfg, device="cuda"):
        """cfg: 설정의 detector.<name> 블록 (백엔드 고유 항목)."""
        self.cfg = cfg or {}
        self.device = device

    @abstractmethod
    def detect(self, rgb, depth_m, K, top_k=None):
        """단일 프레임 → GraspGroup (카메라 좌표계, score 내림차순)."""
        raise NotImplementedError
