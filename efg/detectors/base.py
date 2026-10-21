"""파지 검출 백엔드 인터페이스 — 모든 방법(FoundGrasp·비교군)이 이 계약을 지킨다.

입출력 계약:
  입력  rgb (H,W,3) uint8 RGB / depth_m (H,W) float32 미터(무효=0) / K (3,3)
  출력  graspnetAPI.grasp.GraspGroup, **카메라 color optical frame**.
        그립 프레임 규약(graspnetAPI): 회전행렬 1열=접근 방향, 2열=핑거 개폐 방향.
        width·depth는 미터.
  - FoundGrasp / AnyGrasp width = 필요한 개방 폭. FoundGrasp는 그대로 실행한다.
    AnyGrasp SDK 출력에 추정 배율을 나누지 않으며, 실행 시 open_multiplier를 곱한다.
    기존 EconomicGrasp/RNGNet/HGGD 워커는 raw 폭 계약을 유지한다 (이번 검증 대상 아님).
  - 방법 고유의 충돌검사는 포함한다 (논문 파이프라인의 일부: FoundGrasp refiner의
    측정면 검사 또는 점군 CD, RNGNet·HGGD·AnyGrasp 내장 CD).
  - 기본 FoundGrasp는 독립 detect()의 최종 NMS·top-50을 유지한다.
    configs/benchmark.yaml은 두 방법의 외부 NMS·top-k를 끄고 공유 충돌 검사를 적용한다.
  - ROI·하드웨어 폭 한계·IK는 실행 계층에서 적용한다.
"""
from abc import ABC, abstractmethod


class GraspDetector(ABC):
    name = "base"

    def __init__(self, cfg, device="cuda"):
        """cfg: 설정의 detector.<name> 블록 (백엔드 고유 항목)."""
        self.cfg = cfg or {}
        self.device = device

    @abstractmethod
    def detect(self, rgb, depth_m, K):
        """단일 프레임 → GraspGroup (카메라 좌표계, 순서 무관)."""
        raise NotImplementedError

    def describe(self):
        """로그·기록용 한 줄 설명 (방법 + 핵심 설정)."""
        return self.name

    def close(self):
        """자원 정리 (워커 프로세스 종료 등). 기본은 없음."""
