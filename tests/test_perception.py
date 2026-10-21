import json
from unittest.mock import Mock

import numpy as np
import pytest

from efg.perception import detect_frame
from efg.paths import import_grasp_group
from efg import postprocess
from efg.config import load_config
from efg.recording import save_detection, write_manifest


def frame():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    K = np.array([[10., 0, 2], [0, 10., 2], [0, 0, 1]])
    arr = np.zeros((3, 17))
    arr[:, 0] = [.8, .4, .2]
    arr[:, 1:4] = [.06, .03, .02]
    arr[:, 4:13] = np.eye(3).reshape(9)
    arr[:, 15] = 1.
    arr[1, 1] = .12  # too wide
    arr[2, 13] = 10.  # outside ROI
    return rgb, depth, K, arr


def test_shared_pipeline_uses_masked_input_and_preserves_opening(tmp_path):
    rgb, depth, K, arr = frame()
    detector = Mock(last_inference_ms=12.5)
    detector.detect.return_value = import_grasp_group()(arr)
    cfg = load_config('configs/benchmark.yaml')
    cfg['postprocess']['collision']['enabled'] = False
    result = detect_frame(detector, rgb, depth, K, cfg, [1, 1, 3, 3])
    assert result.counts == {'detected': 3, 'score_ok': 3, 'width_ok': 2,
                             'collision_free': 2, 'roi': 1}
    passed_depth = detector.detect.call_args.args[1]
    assert np.count_nonzero(passed_depth) == 4
    assert np.count_nonzero(depth) == 16
    assert result.grasps.widths[0] == pytest.approx(.06)
    save_detection(tmp_path / 'frame', rgb, depth, K, result)
    with np.load(tmp_path / 'frame_frame.npz') as z:
        np.testing.assert_array_equal(z['depth_m'], depth)
    metrics = json.loads((tmp_path / 'frame_metrics.json').read_text())
    assert metrics['inference_ms'] == 12.5
    assert np.load(tmp_path / 'frame_raw_grasps.npy').shape == (3, 17)


@pytest.mark.parametrize('invalid', ['nan', 'rotation', 'width'])
def test_malformed_detector_output_cannot_reach_shared_filters(invalid):
    rgb, depth, K, arr = frame()
    if invalid == 'nan':
        arr[0, 13] = np.nan
    elif invalid == 'rotation':
        arr[0, 4:13] = 0
    else:
        arr[0, 1] = -.1
    detector = Mock()
    detector.detect.return_value = import_grasp_group()(arr)
    with pytest.raises(ValueError, match='Detector returned'):
        detect_frame(detector, rgb, depth, K, load_config())


def test_common_execution_enables_collision_for_both_methods(monkeypatch):
    _, depth, K, arr = frame()
    cfg = load_config('configs/benchmark.yaml')
    # Reaching backproject proves the legacy FoundGrasp bypass is disabled.
    sentinel = Mock(side_effect=RuntimeError('shared CD reached'))
    monkeypatch.setattr(postprocess, 'backproject', sentinel)
    for name in ['foundgrasp', 'anygrasp']:
        cfg['detector']['name'] = name
        with pytest.raises(RuntimeError, match='shared CD reached'):
            postprocess.shared_filter(import_grasp_group()(arr.copy()), depth, K, cfg)
        assert postprocess.open_width_m(.06, cfg) == pytest.approx(
            .06 if name == 'foundgrasp' else .072)
    assert sentinel.call_count == 2


def test_manifest_records_effective_settings_and_content_hash(tmp_path):
    checkpoint = tmp_path / 'weights'
    checkpoint.write_bytes(b'weights')
    cfg = load_config('configs/benchmark.yaml')
    cfg['detector']['anygrasp']['args'] = ['--checkpoint', str(checkpoint)]
    write_manifest(cfg, tmp_path / 'manifest.json')
    info = json.loads((tmp_path / 'manifest.json').read_text())
    assert info['config']['postprocess']['protocol'] == 'common'
    assert len(info['checkpoint_sha256']) == 64
    assert info['width_semantics'] == 'required_opening_m'
    assert 'efg/perception.py' in info['source_sha256']
