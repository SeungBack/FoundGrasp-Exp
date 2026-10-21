"""Standalone FoundGrasp delegation and robot execution boundary."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg.detectors.foundgrasp import FoundGraspDetector
from efg.paths import import_grasp_group
from efg import postprocess


@pytest.mark.parametrize("with_refiner", [False, True])
@pytest.mark.parametrize("common", [False, True])
def test_delegates_nms_topk_and_collision_to_original(with_refiner, common):
    detector = FoundGraspDetector.__new__(FoundGraspDetector)
    detector.torch = SimpleNamespace(no_grad=nullcontext)
    detector.refiner = object() if with_refiner else None
    detector.collision = True
    detector.collision_voxel = .01
    detector.top_k = 0 if common else 50
    detector.final_nms = not common
    detector.impl = SimpleNamespace(last_verifier_ms=999)
    expected = object()

    def detect(*args, **kwargs):
        assert detector.impl.last_verifier_ms is None
        detector.impl.last_verifier_ms = 7 if with_refiner else None
        return expected

    detector.impl.detect = Mock(side_effect=detect)
    assert detector.detect("rgb", "depth", "K") is expected
    detector.impl.detect.assert_called_once_with(
        "rgb", "depth", "K", top_k=0 if common else 50, nms=not common,
        use_collision_detection=True, collision_voxel_size=.01,
        store_vis=False, use_verifier=with_refiner)
    assert detector.last_refiner_ms == (7 if with_refiner else None)


def test_standalone_skips_extra_collision_and_extra_width_margin(monkeypatch):
    rows = np.zeros((3, 17))
    rows[:, 0] = [.9, .8, 0]
    rows[:, 1] = [.06, .10, .04]
    gg = import_grasp_group()(rows)
    cfg = {"detector": {"name": "foundgrasp"}, "postprocess": {
        "max_open_m": .095, "open_multiplier": 1.2,
        "collision": {"enabled": True}}}
    monkeypatch.setattr(postprocess, "backproject", Mock(side_effect=AssertionError("extra CD")))
    result, counts = postprocess.shared_filter(gg, None, None, cfg)
    assert len(result) == 1
    assert counts == {"detected": 3, "score_ok": 2, "width_ok": 1, "collision_free": 1}
    assert postprocess.open_width_m(.06, cfg) == pytest.approx(.06)
    cfg["detector"]["name"] = "economicgrasp"
    assert postprocess.open_width_m(.06, cfg) == pytest.approx(.072)
