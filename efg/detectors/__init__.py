"""파지 검출 백엔드 레지스트리.

비교군 추가 절차:
  1) efg/detectors/<name>.py 에 GraspDetector 구현 (base.py 계약 참조).
     원본 코드는 third_party/<Name>/ 에 두고 add_vendor_paths(...)로 등록.
  2) 아래 REGISTRY 에 "<name>": ("<module>", "<Class>") 추가.
  3) 설정에 detector.<name>: {...} 블록을 두고 detector.name 으로 선택.
import는 build 시점에 한다 — 백엔드마다 torch 등 무거운 의존성이 다르므로.
"""
import importlib

from .base import GraspDetector  # noqa: F401  (re-export)

REGISTRY = {
    "foundgrasp": ("efg.detectors.foundgrasp", "FoundGraspDetector"),
}


def available():
    return sorted(REGISTRY)


def build_detector(cfg):
    """설정(cfg['detector'])으로 백엔드 인스턴스 생성."""
    d = cfg["detector"]
    name = d.get("name", "foundgrasp")
    if name not in REGISTRY:
        raise SystemExit(f"알 수 없는 detector.name '{name}' — "
                         f"사용 가능: {', '.join(available())}")
    module, cls_name = REGISTRY[name]
    cls = getattr(importlib.import_module(module), cls_name)
    block = d.get(name)
    if block is None:
        raise SystemExit(f"설정 누락: detector.{name} 블록")
    return cls(block, device=d.get("device", "cuda"))
