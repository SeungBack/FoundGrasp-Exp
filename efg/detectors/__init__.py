"""파지 검출 백엔드 레지스트리 — 설정의 detector.name 하나로 방법을 바꾼다.

  foundgrasp      FoundGrasp-Base + grasp refiner (refiner.enabled=false → detector only)
  economicgrasp   EconomicGrasp (ECCV'24)   — 워커, env efg-economicgrasp
  anygrasp        AnyGrasp SDK              — 워커, env efg-anygrasp
  rngnet          RNGNet (CoRL'24)          — 워커, env efg-hggd
  hggd            HGGD (RA-L'23)            — 워커, env efg-hggd

비교군 추가 절차:
  1) 같은 프로세스에서 돌 수 있으면 efg/detectors/<name>.py 에 GraspDetector 구현
     (base.py 계약), 별도 환경이 필요하면 워커 스크립트(sim_worker.py 형식)만 작성.
  2) 아래 REGISTRY 에 등록 (워커는 WORKER_BACKENDS 에 이름만 추가).
  3) 설정에 detector.<name>: {...} 블록을 두고 detector.name 으로 선택.
import는 build 시점에 한다 — 백엔드마다 torch 등 무거운 의존성이 다르므로.
"""
import importlib

from .base import GraspDetector  # noqa: F401  (re-export)

WORKER_BACKENDS = ("economicgrasp", "anygrasp", "rngnet", "hggd")

REGISTRY = {
    "foundgrasp": ("efg.detectors.foundgrasp", "FoundGraspDetector"),
    **{n: ("efg.detectors.worker", "WorkerDetector") for n in WORKER_BACKENDS},
}


def available():
    return sorted(REGISTRY)


def build_detector(cfg):
    """설정(cfg['detector'])으로 백엔드 인스턴스 생성."""
    d = cfg["detector"]
    name = d.get("name", "foundgrasp")
    protocol = cfg.get('postprocess', {}).get('protocol', 'legacy')
    if protocol not in ('legacy', 'common'):
        raise ValueError(f"Unknown postprocess.protocol: {protocol}")
    if protocol == 'common' and name not in ('foundgrasp', 'anygrasp'):
        raise ValueError("common protocol currently supports foundgrasp / anygrasp only")
    if name not in REGISTRY:
        raise SystemExit(f"알 수 없는 detector.name '{name}' — "
                         f"사용 가능: {', '.join(available())}")
    module, cls_name = REGISTRY[name]
    cls = getattr(importlib.import_module(module), cls_name)
    block = d.get(name)
    if block is None:
        raise SystemExit(f"설정 누락: detector.{name} 블록")
    kwargs = {"name": name} if cls_name == "WorkerDetector" else {}
    return cls(block, device=d.get("device", "cuda"), **kwargs)
