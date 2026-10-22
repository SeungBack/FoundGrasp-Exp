"""Lens correction tests without a camera, model or robot."""
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
import yaml

from efg.rgbd_geometry import RGBDCalibration
from efg.camera import D435, load_frame_npz, save_frame_npz


@pytest.fixture
def scene(tmp_path):
    K = np.array([[60., 0, 40.], [0, 62., 30.], [0, 0, 1.]])
    D = np.array([.2, .05, .01, -.002, 0.])
    data = dict(K=K.tolist(), D=D.tolist(), width=80, height=60,
                distortion_model='opencv_brown_conrady')
    path = tmp_path / 'intrinsics.yaml'
    path.write_text(yaml.safe_dump(data))
    calib = dict(intrinsics=str(path), undistort=True)
    return SimpleNamespace(K=K, D=D, data=data, path=path, calib=calib)


def test_pinhole_rays_map_to_calibrated_distorted_pixels(scene):
    s = scene
    geometry = RGBDCalibration(s.calib)
    rgb = np.zeros((60, 80, 3), np.uint8)
    depth = np.ones((60, 80), np.float32)
    _, _, K = geometry.process(rgb, depth, np.eye(3))
    pixels = np.array([[10., 10.], [60., 40.], [40., 30.]])
    rays = np.column_stack(((pixels - s.K[:2, 2]) / np.diag(s.K)[:2], np.ones(3)))
    expected = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), s.K, s.D)[0][:, 0]
    mx, my = geometry._maps
    actual = np.array([[mx[int(v), int(u)], my[int(v), int(u)]] for u, v in pixels])
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    np.testing.assert_array_equal(K, s.K)


def test_latest_hand_eye_supplies_transform_and_intrinsics_on_restart(scene):
    from efg.config import load_config
    from efg.transforms import load_hand_eye
    from efg.camera import resolve_intrinsics
    latest = scene.path.parent / 'hand_eye_latest.yaml'
    config = scene.path.parent / 'config.yaml'
    config.write_text(yaml.safe_dump({'calib': {
        'hand_eye': str(latest), 'intrinsics': 'hand_eye', 'undistort': True}}))
    for z, k1 in [(0.1, 0.2), (0.2, 0.3)]:
        T = np.eye(4)
        T[2, 3] = z
        intrinsics = dict(scene.data, D=[k1, 0, 0, 0, 0])
        latest.write_text(yaml.safe_dump({
            'T_flange_cam': {'matrix': T.tolist()}, 'camera_intrinsics': intrinsics}))
        cfg = load_config(config)
        assert cfg['calib']['hand_eye'] == cfg['calib']['intrinsics'] == str(latest)
        np.testing.assert_array_equal(load_hand_eye(cfg['calib']['hand_eye']), T)
        geometry = RGBDCalibration(cfg['calib'])
        np.testing.assert_array_equal(geometry.K, scene.K)
        np.testing.assert_array_equal(geometry.D, intrinsics['D'])
        np.testing.assert_array_equal(resolve_intrinsics(np.eye(3), cfg), scene.K)
    latest.unlink()
    with pytest.raises(ValueError, match='requires a calibrated intrinsics file'):
        RGBDCalibration(load_config(config)['calib'])


def test_rgb_and_depth_use_same_map_without_blending_depth_edges(scene):
    geometry = RGBDCalibration(scene.calib)
    y, x = np.indices((60, 80))
    rgb = np.stack([x, y, np.zeros_like(x)], axis=-1).astype(np.uint8)
    depth = np.where(x < 40, .5, 1.).astype(np.float32)
    depth[20:30, 20:30] = 0
    out_rgb, out_depth, _ = geometry.process(rgb, depth, scene.K)
    assert set(np.unique(out_depth)) <= {0, .5, 1.}
    mx, my = geometry._maps
    valid = (mx >= 1) & (mx < 78) & (my >= 1) & (my < 58)
    np.testing.assert_allclose(out_rgb[..., 0][valid], mx[valid], atol=.6)
    np.testing.assert_allclose(out_rgb[..., 1][valid], my[valid], atol=.6)
    xi, yi = np.rint(mx[valid]).astype(int), np.rint(my[valid]).astype(int)
    np.testing.assert_array_equal(out_depth[valid], depth[yi, xi])
    assert (out_depth[~valid] == 0).any()
    maps = geometry._maps
    geometry.process(rgb, depth, scene.K)
    assert geometry._maps is maps  # cached across frames


def test_zero_distortion_is_identity(scene):
    scene.data['D'] = [0.] * 5
    scene.path.write_text(yaml.safe_dump(scene.data))
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, (60, 80, 3), np.uint8)
    depth = rng.random((60, 80), dtype=np.float32)
    a, b, _ = RGBDCalibration(scene.calib).process(rgb, depth, scene.K)
    np.testing.assert_array_equal(a, rgb)
    np.testing.assert_array_equal(b, depth)


def test_old_npz_corrected_once_and_recorded_k_preserved(scene, tmp_path):
    rgb = np.arange(60*80*3, dtype=np.uint8).reshape(60, 80, 3)
    depth = np.ones((60, 80), np.float32)
    old = tmp_path / 'old.npz'
    np.savez(old, rgb=rgb, depth_m=depth, K=np.eye(3))  # historical no marker
    cfg = {'calib': scene.calib}
    a, b, K, metadata = load_frame_npz(old, cfg, with_metadata=True)
    assert metadata == {'rectified': True}
    path = tmp_path / 'rectified.npz'
    save_frame_npz(path, a, b, K, **metadata)
    # A changed/missing calibration must not change already rectified pixels/K.
    replay = load_frame_npz(path, {'calib': {'undistort': True}}, with_metadata=True)
    for actual, expected in zip(replay[:3], (a, b, K)):
        np.testing.assert_array_equal(actual, expected)
    assert replay[3]['rectified'] is True


def test_calibrate_intrinsics_source_overrides_stale_factory_model(scene):
    scene.data.update(source='calibrate_intrinsics.py (session ChArUco images, factory guess + FIX_K3)',
                      distortion_model='distortion.inverse_brown_conrady')
    scene.path.write_text(yaml.safe_dump(scene.data))
    assert RGBDCalibration(scene.calib).rectified
    scene.data.pop('source')
    scene.path.write_text(yaml.safe_dump(scene.data))
    with pytest.raises(ValueError, match='Unsupported'):
        RGBDCalibration(scene.calib)


def test_wrong_resolution_rejected(scene):
    with pytest.raises(ValueError, match='size'):
        RGBDCalibration(scene.calib).process(np.zeros((30, 40, 3), np.uint8),
                                            np.ones((30, 40), np.float32), scene.K)


def test_enabled_correction_never_silently_falls_back():
    with pytest.raises(ValueError, match='requires'):
        RGBDCalibration({'undistort': True})


def test_live_get_frame_uses_same_geometry_as_offline(scene):
    cam = object.__new__(D435)
    cam.geometry = RGBDCalibration(scene.calib)
    cam.K = np.eye(3)
    cam.depth_scale = .001
    bgr = np.zeros((60, 80, 3), np.uint8)
    bgr[..., 0] = np.arange(80)[None, :]
    z = np.full((60, 80), 500, np.uint16)
    cam._raw_frame = lambda: (SimpleNamespace(get_data=lambda: bgr),
                             SimpleNamespace(get_data=lambda: z))
    actual = cam.get_frame()
    expected = RGBDCalibration(scene.calib).process(bgr[..., ::-1].copy(), z.astype(np.float32)*.001, cam.K)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
