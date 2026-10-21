"""SDK boundary tests; no license, GPU or robot required."""
import importlib.machinery
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from workers.anygrasp_worker import points_from_depth, predict, sdk_runtime
from efg.paths import import_grasp_group
from efg.postprocess import open_width_m


def test_projection_uses_supplied_intrinsics_and_drops_invalid_depth():
    depth = np.array([[1, np.nan, np.inf], [0, -1, 2]], dtype=np.float32)
    K = np.array([[2, 0, 1], [0, 4, 1], [0, 0, 1]])
    points = points_from_depth(depth, K)
    np.testing.assert_allclose(points, [[-.5, -.25, 1]])
    assert points.dtype == np.float32 and points.flags.c_contiguous


def test_empty_scene_does_not_call_sdk():
    sdk = Mock()
    assert predict(sdk, np.zeros((2, 2), dtype=np.float32), np.eye(3)).shape == (0, 17)
    sdk.get_grasp.assert_not_called()


def test_sdk_width_is_preserved_and_execution_margin_applied_once():
    rows = np.zeros((1, 17))
    rows[0, :4] = [.8, .06, .03, .02]
    rows[0, 4:13] = np.eye(3).reshape(9)
    sdk = Mock()
    sdk.get_grasp.return_value = import_grasp_group()(rows)
    out = predict(sdk, np.ones((2, 2), dtype=np.float32), np.eye(3), nms=False)
    np.testing.assert_array_equal(out, rows)
    sdk.get_grasp.assert_called_once()
    assert sdk.get_grasp.call_args.args[1] == {'dense_grasp': False, 'collision_detection': True}
    cfg = {'detector': {'name': 'anygrasp'},
           'postprocess': {'max_open_m': .1, 'open_multiplier': 1.2}}
    assert open_width_m(out[0, 1], cfg) == pytest.approx(.072)


@pytest.mark.parametrize('profile', ['configs/default.yaml', 'configs/benchmark.yaml'])
def test_execution_width_policy_in_both_profiles(profile):
    from efg.config import load_config
    cfg = load_config(profile)
    cfg['detector']['name'] = 'anygrasp'
    assert open_width_m(.06, cfg) == pytest.approx(.072)
    assert open_width_m(.09, cfg) == pytest.approx(.1)
    cfg['postprocess']['open_multiplier'] = 1.1
    assert open_width_m(.06, cfg) == pytest.approx(.066)
    cfg['detector']['name'] = 'foundgrasp'
    assert open_width_m(.06, cfg) == pytest.approx(.06)


def test_license_layout_is_private_and_source_checkout_unchanged(tmp_path):
    sdk = tmp_path / 'sdk'
    versions = sdk / 'grasp_detection/gsnet_versions'
    license_dir = sdk / 'license_registration/license'
    versions.mkdir(parents=True)
    license_dir.mkdir(parents=True)
    binary = versions / ('gsnet' + importlib.machinery.EXTENSION_SUFFIXES[0])
    binary.write_bytes(b'test binary')
    (license_dir / 'licenseCfg.json').write_text('{}')
    runtime = sdk_runtime(sdk, tmp_path / 'worker')
    assert (runtime / 'gsnet.so').resolve() == binary
    assert (runtime / 'license').resolve() == license_dir
    assert not (versions / 'license').exists()
    assert sdk_runtime(sdk, tmp_path / 'worker') == runtime


def test_missing_license_fails_before_sdk_import(tmp_path):
    with pytest.raises(RuntimeError, match='license directory missing'):
        sdk_runtime(tmp_path, tmp_path / 'worker')
