"""Standalone FoundGrasp delegation and robot execution boundary."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg.detectors.foundgrasp import FoundGraspDetector
from efg.paths import FOUNDGRASP_ROOT, add_vendor_paths, import_grasp_group
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


@pytest.mark.parametrize("selected_depth, writebacks", [(.02, 1), (.025, 0)])
def test_native_selected_width_is_restored_without_changing_score_or_depth(
        monkeypatch, selected_depth, writebacks):
    """Exercise real p_free's writeback branch with a known network verdict."""
    import torch

    add_vendor_paths(FOUNDGRASP_ROOT, FOUNDGRASP_ROOT / 'scripts')
    from verifier_points import Verifier

    monkeypatch.delenv('VERIFIER_NONET', raising=False)
    monkeypatch.setenv('VERIFIER_BUDGET', '0')
    # The refiner branch runs entirely on CPU; only the learned verdict is fixed.
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    native = Verifier.__new__(Verifier)
    native.grid = True
    native.features = []
    native.ax = native.P = None
    native.flag = 0
    native.compiled = False
    native.G = SimpleNamespace(fc_score=lambda q, kind: q[:, 0])
    native.model = SimpleNamespace(
        cfg={'encoder': 'pillar', 'mu': [.1], 'fc_target': 'cls'},
        surf_tau=0., select=([0.], None), fold=lambda maps: None)
    native.fast = lambda gi, *args, **kwargs: (
        torch.zeros(len(gi)), torch.ones(len(gi), 1), None,
        torch.tensor([[.1, selected_depth]]).repeat(len(gi), 1))
    xyz = torch.tensor([[[0., .1], [0., .1]], [[0., 0.], [.1, .1]],
                        [[.5, .5], [.5, .5]]])
    rows = np.zeros((1, 17), dtype=np.float32)
    rows[0, :4] = [.7, .12, .02, .02]
    rows[0, 4:13] = np.eye(3).flatten()
    rows[0, 15] = .5
    gg = import_grasp_group()(rows)
    tensor = torch.from_numpy(gg.grasp_group_array)
    verdict = native.p_free(tensor, xyz, {}, max_w=.1)
    gg.grasp_group_array[:, 0] *= verdict.numpy()
    # Upstream misses width writeback precisely when depth stays unchanged.
    assert gg.widths[0] == pytest.approx(.12 if writebacks else .1)
    scored = gg.grasp_group_array.copy()

    detector = FoundGraspDetector.__new__(FoundGraspDetector)
    detector.torch = SimpleNamespace(no_grad=nullcontext)
    detector.refiner = native
    detector.refiner_max_w = .1
    detector.collision = True
    detector.collision_voxel = .01
    detector.impl = SimpleNamespace(last_verifier_ms=None, detect=Mock(return_value=gg))
    result = detector.detect('rgb', 'depth', 'K')
    assert result.widths[0] == pytest.approx(.1)
    assert result.depths[0] == pytest.approx(selected_depth)
    assert detector.last_refiner_width_writebacks == writebacks
    np.testing.assert_array_equal(result.grasp_group_array[:, 0], scored[:, 0])
    np.testing.assert_array_equal(result.grasp_group_array[:, 2:], scored[:, 2:])


def test_width_writeback_does_not_retain_rejections_or_hide_invalid_geometry(monkeypatch):
    monkeypatch.delenv('VERIFIER_NONET', raising=False)
    detector = FoundGraspDetector.__new__(FoundGraspDetector)
    detector.refiner = SimpleNamespace(
        grid=True, model=SimpleNamespace(cfg={'encoder': 'pillar'}, select=([0.], None)))
    detector.refiner_max_w = .1
    rows = np.zeros((9, 17))
    rows[:, 0] = [1., 0., -1., np.nan, np.inf, 1., 1., 1., 1.]
    # A selected float32 100mm width copied into float64 is not a missed writeback.
    rows[:, 1] = [.12, .12, .12, .12, .12, .06, np.inf, np.nan, float(np.float32(.1))]
    expected = rows.copy()
    expected[0, 1] = .1
    gg = import_grasp_group()(rows)
    assert detector._restore_refiner_selected_widths(gg) == 1
    np.testing.assert_array_equal(gg.grasp_group_array, expected)


@pytest.mark.parametrize('grid, select, encoder, nonet', [
    (False, ([0.], None), 'pillar', '0'),
    (True, None, 'pillar', '0'),
    (True, ([0.], None), 'other', '0'),
    (True, ([0.], None), 'pillar', '1'),
])
def test_width_writeback_skips_modes_without_configuration_selection(
        monkeypatch, grid, select, encoder, nonet):
    monkeypatch.setenv('VERIFIER_NONET', nonet)
    detector = FoundGraspDetector.__new__(FoundGraspDetector)
    detector.refiner = SimpleNamespace(
        grid=grid, model=SimpleNamespace(cfg={'encoder': encoder}, select=select))
    detector.refiner_max_w = .1
    rows = np.zeros((1, 17))
    rows[0, :2] = [.7, .12]
    gg = import_grasp_group()(rows)
    assert detector._restore_refiner_selected_widths(gg) == 0
    np.testing.assert_array_equal(gg.grasp_group_array, rows)
