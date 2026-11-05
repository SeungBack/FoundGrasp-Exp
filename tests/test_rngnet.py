"""RNGNet boundary/geometry regression tests without model files or a GPU."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from workers.rngnet_worker import (RNGNetRuntime, grasp_array, prepare_frame,
                                   project_centers, xyz_from_depth, normalize_depth)
from efg.postprocess import open_width_m


def test_depth_normalization_ignores_invalid_padding_without_filling_depth():
    import torch

    observed = torch.tensor([[.45, .50], [.55, .60]])
    # The same observation embedded in a different ROI must keep its baseline.
    masked = torch.zeros((8, 10))
    masked[2:4, 3:5] = observed
    original = masked.clone()
    centered = normalize_depth(masked)
    torch.testing.assert_close(centered[2:4, 3:5], normalize_depth(observed))
    torch.testing.assert_close(centered[2:4, 3:5], observed - observed.mean())
    torch.testing.assert_close(centered[masked == 0],
                               torch.full_like(masked[masked == 0], -observed.mean()))
    torch.testing.assert_close(masked, original)


def test_depth_normalization_preserves_upstream_clamp_and_handles_empty_input():
    import torch

    observed = torch.tensor([.1, .1, .1, 1.5])
    torch.testing.assert_close(normalize_depth(observed),
                               (observed - observed.mean()).clamp(-1, 1))
    assert normalize_depth(torch.zeros((1, 640, 360))) is None


def test_frame_with_no_valid_depth_after_resize_skips_networks(monkeypatch):
    import torch

    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    runtime = RNGNetRuntime.__new__(RNGNetRuntime)
    runtime.depth_max_m, runtime.torch = 1.5, torch
    runtime.evaluation = SimpleNamespace()
    runtime.anchornet, runtime.localnet = Mock(), Mock()
    depth = np.zeros((720, 1280), dtype=np.float32)
    depth[1, 1] = .5  # Nearest 2x downsampling discards this isolated pixel.
    result = runtime.predict(np.zeros((720, 1280, 3), dtype=np.uint8),
                             depth, np.eye(3))
    assert result.shape == (0, 17)
    runtime.anchornet.assert_not_called()
    runtime.localnet.assert_not_called()


def test_rgb_depth_resize_scales_supplied_intrinsics_without_changing_camera_frame():
    rgb = np.full((360, 640, 3), [255, 30, 10], dtype=np.uint8)
    depth = np.full((360, 640), .5, dtype=np.float32)
    K = np.array([[450., 0, 320], [0, 460., 180], [0, 0, 1]])
    rgb_v, depth_v, K_v = prepare_frame(rgb, depth, K)
    np.testing.assert_array_equal(rgb_v[360, 640], [255, 30, 10])
    assert rgb_v.shape == (720, 1280, 3)
    assert depth_v.shape == (720, 1280) and depth_v.dtype == np.float32
    assert depth_v[360, 640] == 500.
    np.testing.assert_allclose(K_v, np.diag([2., 2., 1.]) @ K)
    xyz = xyz_from_depth(depth_v, K_v)
    np.testing.assert_allclose(xyz[360, 640], [0, 0, .5])
    np.testing.assert_allclose(xyz[400, 680], [20 / 450 * .5, 20 / 460 * .5, .5])
    np.testing.assert_array_equal(K, [[450., 0, 320], [0, 460., 180], [0, 0, 1]])


def test_masked_nonfinite_and_far_depth_never_becomes_a_synthetic_wall():
    depth = np.array([[.5, 0, -1], [np.nan, np.inf, 2.]], dtype=np.float32)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    _, actual, _ = prepare_frame(rgb, depth, np.eye(3))
    assert set(np.unique(actual)) == {0., 500.}
    assert np.isfinite(actual).all()
    assert depth[0, 2] == -1  # caller data is not changed


@pytest.mark.parametrize('K', [
    np.eye(2), [[0, 0, 1], [0, 1, 1], [0, 0, 1]],
    [[1, 0, 1], [0, -1, 1], [0, 0, 1]],
    [[1, 1, 1], [0, 1, 1], [0, 0, 1]],
    [[1, 0, 1], [0, 1, 1], [0, 1, 1]],
    [[np.nan, 0, 1], [0, 1, 1], [0, 0, 1]],
])
def test_invalid_intrinsics_are_rejected(K):
    with pytest.raises(ValueError, match='intrinsics'):
        prepare_frame(np.zeros((2, 3, 3), dtype=np.uint8), np.ones((2, 3)), K)


@pytest.mark.parametrize('maximum', [0, -1, np.nan, np.inf])
def test_invalid_depth_limits_are_rejected(maximum):
    with pytest.raises(ValueError, match='depth_max'):
        prepare_frame(np.zeros((2, 3, 3), dtype=np.uint8), np.ones((2, 3)),
                      np.eye(3), depth_max_m=maximum)


def test_centers_with_no_depth_are_discarded_instead_of_borrowing_object_depth():
    depth = np.zeros((720, 1280), dtype=np.float32)
    depth[200, 400] = 500.
    # There is real depth next to the second centre, but none at that centre.
    depth[202, 404] = 900.
    K = np.array([[900., 0, 640], [0, 800., 360], [0, 0, 1]])
    centers = np.array([[200, 100], [201, 101], [-1, 0], [640, 360], [np.nan, 1]])
    keep, xyz = project_centers(centers, depth, K)
    np.testing.assert_array_equal(keep, [True, False, False, False, False])
    np.testing.assert_allclose(xyz, [[(400 - 640) / 900 * .5, (200 - 360) / 800 * .5, .5]])
    assert xyz.shape == (1, 3)


def test_all_empty_centers_and_frame_return_shape_without_invoking_networks():
    keep, xyz = project_centers(np.empty((0, 2)), np.zeros((720, 1280)), np.eye(3))
    assert keep.shape == (0,) and xyz.shape == (0, 3)
    runtime = RNGNetRuntime.__new__(RNGNetRuntime)
    runtime.depth_max_m = 1.5
    runtime.anchornet = Mock(side_effect=AssertionError('empty scene must not run network'))
    out = runtime.predict(np.zeros((2, 3, 3), dtype=np.uint8), np.zeros((2, 3)), np.eye(3))
    assert out.shape == (0, 17) and out.dtype == np.float64
    runtime.anchornet.assert_not_called()


class Group(SimpleNamespace):
    def __len__(self):
        return len(self.scores)


def group():
    return Group(scores=np.array([.8]), widths=np.array([.06]),
                 heights=np.array([.02]), depths=np.array([.02]),
                 rotations=np.eye(3)[None], translations=np.array([[.1, .2, .3]]))


def test_graspnet_export_preserves_rotation_translation_and_single_width_margin():
    original = group()
    output = grasp_array(original)
    np.testing.assert_allclose(output[0, :4], [.8, .06, .02, .02])
    np.testing.assert_allclose(output[0, 4:13], np.eye(3).reshape(9))
    np.testing.assert_allclose(output[0, 13:16], [.1, .2, .3])
    assert output[0, 16] == -1
    cfg = {'detector': {'name': 'rngnet'},
           'postprocess': {'max_open_m': .1, 'open_multiplier': 1.2}}
    assert open_width_m(output[0, 1], cfg) == pytest.approx(.072)
    original.translations[0, 0] = np.nan
    with pytest.raises(RuntimeError, match='non-finite'):
        grasp_array(original)


def test_missing_model_files_fail_before_loading_torch_or_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match='repository'):
        RNGNetRuntime(tmp_path, tmp_path / 'missing.pt')
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/localgraspnet.py').touch()
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        RNGNetRuntime(tmp_path, tmp_path / 'missing.pt')


@pytest.mark.parametrize('sparse', [False, True])
def test_one_valid_region_keeps_batch_axis_and_applies_native_collision(monkeypatch, sparse):
    import torch

    # Exercise actual tensor assembly on CPU; upstream models/decoding are mocks.
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    for name in ('tensor', 'linspace'):
        original = getattr(torch, name)
        def on_cpu(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, on_cpu)

    class Rect:
        centers = np.array([[320, 180]])

        def __getitem__(self, keep):
            result = Rect()
            result.centers = self.centers[keep]
            return result

    runtime = RNGNetRuntime.__new__(RNGNetRuntime)
    runtime.depth_max_m, runtime.torch = 1.5, torch
    runtime.anchors = {'gamma': torch.zeros(7), 'beta': torch.zeros(7)}
    runtime.anchornet = Mock(return_value=((None,) * 5, None))
    runtime.localnet = Mock(return_value=(None, torch.zeros((1, 49)),
        torch.zeros((1, 49, 3)), torch.zeros((1, 6)), torch.zeros((1, 6)), torch.zeros((1, 6))))
    candidate = group()
    runtime.evaluation = SimpleNamespace(
        anchor_output_process=Mock(return_value=(np.ones((640, 360)),)),
        sample_2d_centers=Mock(return_value=Rect()),
        get_thetas_widths=Mock(return_value=(np.array([0.]), np.array([.06]))),
        detect_6d_grasp_multi=Mock(return_value=(None, candidate)),
        collision_detect=Mock(return_value=(candidate, np.array([True]))))
    rgb = np.full((720, 1280, 3), [255, 30, 10], dtype=np.uint8)
    K = np.array([[900., 0, 640], [0, 850., 360], [0, 0, 1]])
    depth = np.full((720, 1280), .3, dtype=np.float32)
    if sparse:
        depth[:] = 0
        depth[360, 640] = .3
    out = runtime.predict(rgb, depth, K)
    model_input = runtime.anchornet.call_args.args[0]
    assert model_input.shape == (1, 4, 640, 360)
    # ROI sparsity must not shift the observed surface's anchor depth baseline.
    assert abs(float(model_input[0, 0, 320, 180])) < 1e-6
    np.testing.assert_allclose(model_input[0, 1:, 320, 180], np.array([255, 30, 10]) / 255., rtol=1e-6)
    assert runtime.localnet.call_args.args[0].shape == (1, 64, 64, 6)
    assert runtime.evaluation.get_thetas_widths.call_args.args[0].shape == (1, 6)
    np.testing.assert_allclose(runtime.evaluation.detect_6d_grasp_multi.call_args.args[4][0],
                               [[0, 0, .3]], rtol=1e-6)
    call = runtime.evaluation.collision_detect.call_args
    assert call.kwargs == {'mode': 'graspnet'}
    assert call.args[0].shape == (1 if sparse else 25600, 3)
    assert call.args[1] is candidate
    np.testing.assert_allclose(out, grasp_array(candidate))
