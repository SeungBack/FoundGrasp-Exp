"""Shared execution and per-frame calibration boundaries; no robot/camera needed."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg.camera import load_frame_npz, save_frame_npz
from efg.config import add_config_args, config_from_args, load_config
from efg.detectors import build_detector
from efg.detectors.worker import WorkerDetector
from efg.paths import import_grasp_group
from efg.perception import detect_frame
from efg.recording import file_digest, save_detection, write_manifest
from efg.worker_protocol import validate_camera_pose


def frame():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.full((4, 4), .6, dtype=np.float32)
    K = np.array([[10., 0, 2], [0, 10., 2], [0, 0, 1]])
    T = np.diag([1., -1., -1., 1.])
    T[:3, 3] = [.4, -.2, .7]
    return rgb, depth, K, T


@pytest.mark.parametrize('name', ['anygrasp', 'rngnet', 'gfla'])
def test_detector_switch_preserves_shared_execution_and_builds_worker(monkeypatch, name):
    import efg.detectors.worker as workers
    args = add_config_args(argparse.ArgumentParser()).parse_args([
        '--config', 'configs/benchmark.yaml', '--grasp-detector', name])
    cfg = config_from_args(args)
    reference = load_config('configs/benchmark.yaml')
    for section in ('robot', 'drop', 'gripper', 'calib', 'selection', 'retry', 'postprocess'):
        assert cfg[section] == reference[section]
    factory = Mock()
    monkeypatch.setattr(workers, 'WorkerDetector', factory)
    build_detector(cfg)
    factory.assert_called_once_with(cfg['detector'][name], device='cuda', name=name)


def test_camera_pose_is_required_only_by_declared_backend_and_recorded(tmp_path):
    rgb, depth, K, T = frame()
    cfg = load_config('configs/benchmark.yaml', ['detector.name=gfla'])
    det = Mock(requires_camera_pose=True, last_inference_ms=3.)
    det.detect.return_value = import_grasp_group()()
    with pytest.raises(ValueError, match='requires capture-time'):
        detect_frame(det, rgb, depth, K, cfg)
    det.detect.assert_not_called()
    result = detect_frame(det, rgb, depth, K, cfg, [1, 1, 3, 3],
                          T_base_cam=T, camera_pose_source='measured_robot_and_hand_eye')
    np.testing.assert_array_equal(det.detect.call_args.kwargs['T_base_cam'], T)
    assert np.count_nonzero(det.detect.call_args.args[1]) == 4
    np.testing.assert_array_equal(depth, np.full((4, 4), .6, dtype=np.float32))
    save_detection(tmp_path / 'sample', rgb, depth, K, result, rectified=True)
    loaded_rgb, loaded_depth, loaded_K, context = load_frame_npz(
        tmp_path / 'sample_frame.npz', with_metadata=True)
    np.testing.assert_array_equal(loaded_rgb, rgb)
    np.testing.assert_array_equal(loaded_depth, depth)
    np.testing.assert_array_equal(loaded_K, K)
    np.testing.assert_array_equal(context['T_base_cam'], T)
    assert context['camera_pose_source'] == 'measured_robot_and_hand_eye'
    assert context['rectified'] is True
    det.requires_camera_pose = False
    detect_frame(det, rgb, depth, K, cfg, T_base_cam=T)
    assert det.detect.call_args.kwargs == {}


@pytest.mark.parametrize('invalid', ['missing', 'shape', 'nan', 'reflection', 'scale', 'bottom'])
def test_invalid_pose_never_reaches_model(invalid):
    rgb, depth, K, T = frame()
    if invalid == 'missing':
        T = None
    elif invalid == 'shape':
        T = np.eye(3)
    elif invalid == 'nan':
        T[0, 3] = np.nan
    elif invalid == 'reflection':
        T[0, 0] *= -1
    elif invalid == 'scale':
        T[:3, :3] *= 2
    else:
        T[3, 3] = 0
    det = Mock(requires_camera_pose=True)
    with pytest.raises(ValueError):
        detect_frame(det, rgb, depth, K, load_config(), T_base_cam=T)
    det.detect.assert_not_called()


def test_pose_less_legacy_frame_stays_pose_less(tmp_path):
    rgb, depth, K, _ = frame()
    path = tmp_path / 'old.npz'
    save_frame_npz(path, rgb, depth, K)
    *_, metadata = load_frame_npz(path, with_metadata=True)
    assert metadata == {'rectified': False}


def test_camera_frame_roundtrip_preserves_pose_without_pickle(tmp_path):
    rgb, depth, K, T = frame()
    path = tmp_path / 'calibrated.npz'
    save_frame_npz(path, rgb, depth, K, rectified=True, T_base_cam=T,
                   camera_pose_source='explicit_camera_pose_file')
    *_, metadata = load_frame_npz(path, with_metadata=True)
    np.testing.assert_array_equal(metadata['T_base_cam'], T)
    assert metadata['camera_pose_source'] == 'explicit_camera_pose_file'
    copied = validate_camera_pose(T)
    copied[0, 3] = 99
    assert T[0, 3] == .4


def test_pose_crosses_worker_process_without_axis_or_unit_changes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'worker.py'
    script.write_text(f'''import sys, argparse, numpy as np
sys.path.insert(0, {str(root)!r})
from efg.worker_protocol import serve
ap = argparse.ArgumentParser()
ap.add_argument('--work_dir')
args = ap.parse_args()
def predict(rgb, depth, K, *, T_base_cam=None):
    assert T_base_cam is not None
    arr = np.zeros((1,17))
    arr[0,:4] = [.8,.05,.02,.01]
    arr[0,4:13] = T_base_cam[:3,:3].reshape(9)
    arr[0,13:16] = T_base_cam[:3,3]
    return arr
serve(args.work_dir, predict, poll_interval=.005)
''')
    det = WorkerDetector({'python': sys.executable, 'script': str(script),
                          'poll_interval_s': .005}, name='gfla')
    rgb, depth, K, T = frame()
    try:
        with pytest.raises(ValueError, match='requires capture-time'):
            det.detect(rgb, depth, K)
        output = det.detect(rgb, depth, K, T_base_cam=T)
        np.testing.assert_array_equal(output.rotation_matrices[0], T[:3, :3])
        np.testing.assert_array_equal(output.translations[0], T[:3, 3])
    finally:
        det.close()
    assert det.proc.poll() is not None


def test_manifest_discloses_additional_calibration_and_native_width_policy(tmp_path):
    cfg = load_config('configs/benchmark.yaml', ['detector.name=gfla'])
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'test weights')
    cfg['detector']['gfla']['args'] = ['--checkpoint', str(weights)]
    det = Mock(runtime_metadata={'native_width_add_m': .016},
               describe=lambda: 'gfla test')
    write_manifest(cfg, tmp_path / 'manifest.json', detector=det)
    metadata = json.loads((tmp_path / 'manifest.json').read_text())
    assert metadata['additional_detector_input']['assumption'] == 'table frame equals robot base'
    assert metadata['execution_open_multiplier'] == 1.2
    assert metadata['width_semantics'] == 'required_opening_m'
    assert metadata['worker_runtime']['native_width_add_m'] == .016


def test_foundgrasp_manifest_records_loaded_recipe_and_native_sources(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    import efg.recording as recording

    vendor = tmp_path / 'FoundGrasp'
    vendor.mkdir()
    entrypoint = vendor / 'grasp_detector.py'
    entrypoint.write_text('# native inference entry point\n')
    loaded_source = vendor / 'loaded_backbone.py'
    loaded_source.write_text('# actual imported model module\n')
    monkeypatch.setattr(recording, 'FOUNDGRASP_ROOT', vendor)
    monkeypatch.setitem(sys.modules, '_manifest_test_backbone',
                        SimpleNamespace(__file__=str(loaded_source)))
    base = vendor / 'base.yaml'
    base.write_text('inference: {width_multiplier: 1.2}\n')
    config = vendor / 'model.yaml'
    config.write_text('__base__: base.yaml\ninference: {max_grasps: 4096}\n')
    checkpoint = tmp_path / 'weights.pth'
    checkpoint.write_bytes(b'model')
    refiner = tmp_path / 'refiner.pt'
    refiner.write_bytes(b'refiner')
    cfg = load_config('configs/benchmark.yaml', ['detector.name=foundgrasp'])
    cfg['detector']['foundgrasp'].update(config=str(config), checkpoint=str(checkpoint))
    cfg['detector']['foundgrasp']['refiner']['weights'] = str(refiner)
    # Simulate merged runtime defaults and a shell override absent from YAML.
    monkeypatch.setenv('VERIFIER_NONET', '1')
    merged = {'VERIFIER_PF_EXP': '3', 'VERIFIER_BODY': 'center'}
    loaded_config = OmegaConf.create({'inference': {'width_multiplier': 1.2,
                                                   'max_grasps': 4096}})
    det = SimpleNamespace(
        describe=lambda: 'foundgrasp test', impl=SimpleNamespace(config=loaded_config),
        refiner=object(), refiner_env=merged, final_nms=True, top_k=50,
        device='cpu', torch=SimpleNamespace(__version__='test', version=SimpleNamespace(cuda=None)))
    output = tmp_path / 'manifest.json'
    write_manifest(cfg, output, detector=det)
    metadata = json.loads(output.read_text())
    runtime = metadata['foundgrasp_runtime']
    assert runtime['effective_model_config'] == OmegaConf.to_container(loaded_config)
    assert runtime['refiner_env'] == merged
    assert runtime['refiner_process_env']['VERIFIER_NONET'] == '1'
    assert runtime['final_nms'] is True and runtime['top_k'] == 50
    assert runtime['runtime'] == {'python': metadata['driver_python'], 'torch': 'test',
                                  'cuda': None, 'device': 'cpu', 'gpu': None}
    assert runtime['model_config_sha256'] == {str(config): file_digest(config),
                                             str(base): file_digest(base)}
    assert runtime['native_source_sha256'] == {
        'grasp_detector.py': file_digest(entrypoint),
        'loaded_backbone.py': file_digest(loaded_source)}
    assert metadata['checkpoint_sha256'] == file_digest(checkpoint)
    assert metadata['refiner_sha256'] == file_digest(refiner)
    loaded_source.write_text('# changed native model module\n')
    write_manifest(cfg, output, detector=det)
    updated = json.loads(output.read_text())['foundgrasp_runtime']
    assert updated['native_source_sha256']['loaded_backbone.py'] != runtime['native_source_sha256']['loaded_backbone.py']


def test_pose_aware_warmup_uses_fresh_capture_and_actual_calibration():
    from efg.demo import _warmup_detector
    rgb, depth, K, T = frame()
    stream = Mock(snapshot=Mock(return_value=(rgb, depth, K)))
    det = Mock(requires_camera_pose=True)
    _warmup_detector(det, stream, load_config(), [1, 1, 3, 3], T_base_cam=T)
    stream.snapshot.assert_called_once_with(fresh=True)
    np.testing.assert_array_equal(det.detect.call_args.kwargs['T_base_cam'], T)
    assert np.count_nonzero(det.detect.call_args.args[1]) == 4


@pytest.mark.parametrize('stop_during_warmup', [None, 'interrupt', 'preview_quit'])
def test_live_pose_aware_startup_homes_before_warmup_and_gripper(
        monkeypatch, tmp_path, stop_during_warmup):
    from efg import demo, gripper as gripper_module

    cfg = load_config()
    cfg['output'].update(dir=str(tmp_path), save_cycles=False)
    cfg['scan_roi_px'] = [1, 1, 3, 3]
    rgb, depth, K, T_flange = frame()
    T_hand_eye = np.eye(4)
    T_hand_eye[:3, 3] = [.03, -.01, .08]
    order = []
    detector = Mock(requires_camera_pose=True)
    stream = Mock(quit_requested=False)
    stop = Mock(requested=False)
    robot = Mock()
    gripper = Mock()

    def capture(*, fresh):
        assert fresh is True
        order.append('capture')
        return rgb, depth, K

    def infer(rgb, depth, K, *, T_base_cam):
        order.append('warmup')
        np.testing.assert_allclose(T_base_cam, T_flange @ T_hand_eye)
        assert np.count_nonzero(depth) == 4
        if stop_during_warmup == 'interrupt':
            stop.requested = True
        elif stop_during_warmup == 'preview_quit':
            stream.quit_requested = True

    def initialize_gripper():
        order.append('gripper')
        raise RuntimeError('end startup test')

    stream.snapshot.side_effect = capture
    robot.get_flange_pose.side_effect = lambda: (order.append('pose') or T_flange, None)
    detector.detect.side_effect = infer
    gripper.connect.side_effect = initialize_gripper
    monkeypatch.setattr(demo, 'CameraStream', lambda *a, **kw: stream)
    monkeypatch.setattr(demo, 'build_detector', lambda cfg: detector)
    monkeypatch.setattr(demo, 'connect_robot', lambda *a, **kw: robot)
    monkeypatch.setattr(demo, 'go_home', lambda *a: order.append('home'))
    monkeypatch.setattr(demo, 'load_hand_eye', lambda *a: T_hand_eye)
    monkeypatch.setattr(demo, 'load_payload_check', lambda *a, **kw: None)
    monkeypatch.setattr(demo, 'RerunViz', Mock())
    monkeypatch.setattr(demo, 'StopRequest', lambda: stop)
    monkeypatch.setattr(gripper_module, 'GripperCtl', lambda cfg: gripper)
    if stop_during_warmup:
        assert demo.run(cfg)['cycles'] == 0
        assert order == ['home', 'pose', 'capture', 'warmup']
        gripper.connect.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match='end startup test'):
            demo.run(cfg)
        assert order == ['home', 'pose', 'capture', 'warmup', 'gripper']
    detector.close.assert_called_once()
    robot.disconnect.assert_called_once()
