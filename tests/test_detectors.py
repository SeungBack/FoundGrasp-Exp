"""검출기 레지스트리 — 모델 로드 없이 등록·오류 경로만."""
import pytest

from efg.detectors import REGISTRY, available, build_detector
from efg.detectors.base import GraspDetector


def test_foundgrasp_registered():
    assert "foundgrasp" in available()
    module, cls = REGISTRY["foundgrasp"]
    assert cls == "FoundGraspDetector"


def test_unknown_name_and_missing_block():
    with pytest.raises(SystemExit, match="알 수 없는 detector.name"):
        build_detector({"detector": {"name": "nope"}})
    with pytest.raises(SystemExit, match="설정 누락"):
        build_detector({"detector": {"name": "foundgrasp"}})


def test_custom_backend_can_be_registered(monkeypatch):
    """비교군 추가 절차 검증: 구현 → 레지스트리 등록 → 설정으로 선택."""
    import types, sys

    class Dummy(GraspDetector):
        name = "dummy"

        def detect(self, rgb, depth_m, K, top_k=None):
            return []

    mod = types.ModuleType("efg.detectors._dummy_test")
    mod.Dummy = Dummy
    monkeypatch.setitem(sys.modules, "efg.detectors._dummy_test", mod)
    monkeypatch.setitem(REGISTRY, "dummy", ("efg.detectors._dummy_test", "Dummy"))
    det = build_detector({"detector": {"name": "dummy", "device": "cpu", "dummy": {}}})
    assert det.name == "dummy" and det.device == "cpu"
