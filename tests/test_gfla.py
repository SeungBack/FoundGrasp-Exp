"""GFLA input/output and calibration boundaries; no GPU, robot or model weights."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from workers import gfla_worker as worker


def test_checkpoint_legacy_decoder_allowlist_keeps_every_active_weight_strict():
    import torch
    model = torch.nn.Linear(2, 1)
    model.csf_net = SimpleNamespace(return_coord_only=True)
    state = model.state_dict()
    state['csf_net.conv1.weight'] = torch.ones(1)
    ignored = worker.load_checkpoint(model, state)
    assert ignored == ['csf_net.conv1.weight']
    assert 'csf_net.conv1.weight' in state
    assert len(worker.UNUSED_CSF_CHECKPOINT_KEYS) == 24
    missing = dict(state)
    missing.pop('weight')
    with pytest.raises(RuntimeError, match='Missing key'):
        worker.load_checkpoint(model, missing)
    invalid_shape = dict(state, weight=torch.zeros(3, 4))
    with pytest.raises(RuntimeError, match='size mismatch'):
        worker.load_checkpoint(model, invalid_shape)
    with pytest.raises(RuntimeError, match='unknown keys'):
        worker.load_checkpoint(model, dict(state, unknown=torch.zeros(1)))
    model.csf_net.return_coord_only = False
    with pytest.raises(RuntimeError, match='unexpected active'):
        worker.load_checkpoint(model, state)


def frame(height=720, width=1280):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 180
    rgb[..., 2] = 20
    depth = np.full((height, width), .6, dtype=np.float32)
    K = np.array([[900., 0., width / 2], [0., 910., height / 2], [0., 0., 1.]])
    T = np.diag([1., -1., -1., 1.])
    T[2, 3] = .7
    return rgb, depth, K, T


def test_native_preprocessing_preserves_metres_rgb_and_supplied_intrinsics():
    rgb, depth, K, _ = frame()
    depth[100:120, 300:320] = 0
    image, resized_depth, intr = worker.prepare_rgbd(rgb, depth, K)
    assert image.shape == (384, 640, 3)
    assert resized_depth.shape == (384, 640)
    assert image.dtype == resized_depth.dtype == intr.dtype == np.float32
    np.testing.assert_array_equal(image[0, 0], [180., 0., 20.])
    np.testing.assert_allclose(intr, [[450., 0., 320.], [0., 455., 192.], [0., 0., 1.]])
    assert resized_depth.max() == pytest.approx(.6)
    assert np.all(resized_depth[62:72, 150:160] == 0)
    np.testing.assert_array_equal(depth[100:120, 300:320], 0)
    assert K[1, 2] == 360  # Caller calibration is not changed in place.


def test_nondefault_aspect_ratio_accounts_for_all_padding_in_K():
    rgb, depth, K, _ = frame(480, 848)
    image, resized_depth, intr = worker.prepare_rgbd(rgb, depth, K)
    assert image.shape == (256, 448, 3)
    assert resized_depth.shape == (256, 448)
    assert intr[0, 2] == pytest.approx(K[0, 2] / 2 + 12)
    assert intr[1, 2] == pytest.approx(K[1, 2] / 2 + 8)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -1.])
def test_invalid_depth_becomes_zero_without_mutating_input(bad_value):
    rgb, depth, K, _ = frame(64, 64)
    depth[:32] = bad_value
    _, out, _ = worker.prepare_rgbd(rgb, depth, K)
    assert np.isfinite(out).all() and np.all(out[:16] == 0)
    assert np.all(np.isnan(depth[:32])) if np.isnan(bad_value) else np.all(depth[:32] == bad_value)


@pytest.mark.parametrize('change', ['shape', 'nan', 'reflection', 'scale', 'bottom_row'])
def test_malformed_table_calibration_rejected(change):
    T = frame()[3]
    if change == 'shape':
        T = T[:3]
    elif change == 'nan':
        T[0, 3] = np.nan
    elif change == 'reflection':
        T[0, 0] = -1
    elif change == 'scale':
        T[0, 0] = 2
    else:
        T[3, 2] = 1
    with pytest.raises(ValueError, match='rigid 4x4'):
        worker.validate_table_transform(T)


def test_missing_capture_pose_never_substitutes_identity():
    run = Mock()
    rgb, depth, K, _ = frame(64, 64)
    with pytest.raises(ValueError, match='capture-time T_base_cam'):
        worker.predict(run, rgb, depth, K)
    run.assert_not_called()


def test_empty_depth_skips_model_with_valid_calibration():
    run = Mock()
    rgb, depth, K, T = frame(64, 64)
    depth[:] = 0
    assert worker.predict(run, rgb, depth, K, T_base_cam=T).shape == (0, 17)
    run.assert_not_called()


def test_predict_passes_capture_transform_and_preserves_native_width_and_score():
    rgb, depth, K, T = frame(64, 64)
    rows = np.zeros((1, 17), dtype=np.float32)
    rows[0, :4] = [.83, .076, .02, .01]  # Includes upstream WIDTH_ADD=16mm.
    rows[0, 4:13] = np.eye(3).reshape(-1)
    rows[0, 13:16] = [.03, -.02, .6]
    run = Mock(return_value=[rows])
    out = worker.predict(run, rgb, depth, K, T_base_cam=T)
    rgb_t, depth_t, K_t, T_t = run.call_args.args[0]
    assert rgb_t.shape == (1, 32, 32, 3)
    assert depth_t.shape == (1, 1, 32, 32)
    np.testing.assert_allclose(K_t.numpy()[0, :2], K[:2] / 2)
    np.testing.assert_allclose(T_t.numpy()[0], T)
    np.testing.assert_array_equal(out, rows)
    assert out.dtype == np.float64 and not np.shares_memory(out, rows)


def test_each_request_uses_its_own_capture_pose():
    rgb, depth, K, T = frame(64, 64)
    run = Mock(return_value=[np.empty((0, 17))])
    worker.predict(run, rgb, depth, K, T_base_cam=T)
    changed = T.copy()
    changed[0, 3] = .2
    worker.predict(run, rgb, depth, K, T_base_cam=changed)
    assert run.call_args_list[0].args[0][3][0, 0, 3].item() == 0.
    assert run.call_args_list[1].args[0][3][0, 0, 3].item() == pytest.approx(.2)


def test_native_empty_contact_result_is_empty_but_other_errors_propagate():
    rgb, depth, K, T = frame(64, 64)
    assert worker.predict(Mock(side_effect=worker.NoContactFamilies()), rgb, depth, K,
                          T_base_cam=T).shape == (0, 17)
    with pytest.raises(RuntimeError, match='model failure'):
        worker.predict(Mock(side_effect=RuntimeError('model failure')), rgb, depth, K, T_base_cam=T)


@pytest.mark.parametrize('result', [None, [], [np.zeros((1, 16))], [np.full((1, 17), np.nan)]])
def test_bad_native_output_is_not_silently_accepted(result):
    rgb, depth, K, T = frame(64, 64)
    with pytest.raises(RuntimeError, match='GFLA'):
        worker.predict(Mock(return_value=result), rgb, depth, K, T_base_cam=T)


def test_runtime_weight_layout_leaves_source_checkout_unchanged(tmp_path):
    repo = tmp_path / 'repo'
    for relative in ('models/CFG_main.py', 'cfg/config.yaml', 'cfg/trainer_config.yaml'):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
    weight = repo / 'FastSAM-s.pt'
    weight.write_bytes(b'test checkpoint')
    runtime = worker.prepare_runtime(repo, tmp_path / 'worker', weight, 's')
    assert (runtime / 'base_weights\\FastSAM-s.pt').resolve() == weight
    assert not (repo / 'base_weights\\FastSAM-s.pt').exists()
    assert worker.prepare_runtime(repo, tmp_path / 'worker', weight, 's') == runtime


@pytest.mark.parametrize('fail', [False, True])
def test_launcher_import_uses_visible_device_and_restores_subprocess(monkeypatch, fail):
    original = Mock(return_value=b'other command')
    monkeypatch.setattr(worker.subprocess, 'check_output', original)
    monkeypatch.setattr(worker.sys, 'platform', 'linux')
    monkeypatch.delitem(worker.sys.modules, 'win32com', raising=False)
    monkeypatch.delitem(worker.sys.modules, 'win32com.client', raising=False)
    monkeypatch.setattr(worker.importlib.util, 'find_spec', lambda name: None)
    torch_module = SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=Mock(return_value=(5 * 2**20, 8 * 2**20)),
        get_device_name=Mock(return_value='Visible GPU, 0')))
    def importing(name):
        assert name == 'LauncherTemplate'
        import win32com.client
        with pytest.raises(RuntimeError, match='Windows shortcut creation is unavailable'):
            win32com.client.Dispatch('WScript.Shell')
        output = worker.subprocess.check_output(
            'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,nounits,noheader')
        assert output == b'0, Visible GPU  0, 3, 8\n'
        assert worker.subprocess.check_output(['git', '--version']) == b'other command'
        if fail:
            raise ImportError('dependency')
        return 'loaded'
    monkeypatch.setattr(worker.importlib, 'import_module', importing)
    if fail:
        with pytest.raises(ImportError, match='dependency'):
            worker.import_launcher(torch_module)
    else:
        assert worker.import_launcher(torch_module) == 'loaded'
    assert worker.subprocess.check_output is original
    assert 'win32com' not in worker.sys.modules
    assert 'win32com.client' not in worker.sys.modules
    original.assert_called_once_with(['git', '--version'])
