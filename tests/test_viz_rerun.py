import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

rr = pytest.importorskip("rerun")
from efg.viz_rerun import RerunViz, _project_segments


@pytest.fixture
def viz(tmp_path):
    v = RerunViz({"rerun": {"viewer": "none"}}, tmp_path.name, out_dir=tmp_path)
    yield v
    v.close()


def test_robot_worker_and_cycle_share_timeline(viz, monkeypatch):
    times = []
    poses = []
    original_time, original_log = rr.set_time, rr.log
    sampled = threading.Event()

    def set_time(name, **kwargs):
        times.append((threading.current_thread().name, name))
        original_time(name, **kwargs)

    def log(path, *args, **kwargs):
        if path == "world/robot/link7":
            poses.append(args[0].translation.as_arrow_array().to_pylist())
            sampled.set()
        original_log(path, *args, **kwargs)

    monkeypatch.setattr(rr, "set_time", set_time)
    monkeypatch.setattr(rr, "log", log)
    viz.set_cycle(1)
    arm = SimpleNamespace(connected=True, angles=[0.] * 7)
    robot = SimpleNamespace(arm=arm, T_flange_tcp=np.eye(4))
    robot.T_flange_tcp[2, 3] = .223
    viz.start_robot_stream(robot, hz=50)
    assert sampled.wait(2)
    sampled.clear()
    arm.angles = [.5, 0, 0, 0, 0, 0, 0]
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and np.allclose(poses[0], poses[-1]):
        sampled.wait(.05)
        sampled.clear()
    viz.close()
    assert not np.allclose(poses[0], poses[-1])
    assert {name for _, name in times} == {"elapsed"}
    assert len({thread for thread, _ in times}) == 2


def test_dryrun_tcp_uses_configured_offset(viz, monkeypatch):
    poses = {}
    original = rr.log

    def log(path, *args, **kwargs):
        if path == "world/robot/tcp":
            poses[path] = args[0].translation.as_arrow_array().to_pylist()[0]
        original(path, *args, **kwargs)

    monkeypatch.setattr(rr, "log", log)
    viz.log_robot(np.zeros(7))
    expected = viz.model.flange_pose(np.zeros(7)) @ np.array([0, 0, .215, 1])
    np.testing.assert_allclose(poses["world/robot/tcp"], expected[:3], atol=1e-7)


def test_visual_tcp_prefers_robot_geometry_and_converts_degrees_without_mutation():
    from efg.se3 import xarm_pose_to_se3
    offset = [2., -3., 203., 10., -20., 30.]
    cfg = {"robot": {"tcp_offset_mm_deg": offset},
           "gripper": {"tip_length_mm": 80},
           "rerun": {"enabled": False, "tcp_offset_m": .215}}
    v = RerunViz(cfg, "configured-tool")
    expected = xarm_pose_to_se3(offset[:3] + np.radians(offset[3:]).tolist())
    np.testing.assert_allclose(v._tcp_offset, expected)
    assert v._tip_length_m == .08
    assert offset == [2., -3., 203., 10., -20., 30.]


def test_live_robot_offset_overrides_visual_config(viz, monkeypatch):
    actual = np.eye(4)
    actual[2, 3] = .203
    monkeypatch.setattr(rr, "log", lambda *a, **kw: None)
    viz.log_robot(np.zeros(7), actual)
    np.testing.assert_array_equal(viz._tcp_offset, actual)


def test_robot_stream_error_is_visible(viz, capsys):
    robot = SimpleNamespace(arm=SimpleNamespace(connected=False), T_flange_tcp=np.eye(4))
    robot.T_flange_tcp[2, 3] = .223
    viz.start_robot_stream(robot, hz=50)
    time.sleep(.1)
    viz.close()
    output = capsys.readouterr().out
    assert output.count("로봇 자세 시각화 갱신 실패") == 1


def test_projection_rejects_points_behind_camera():
    seg = np.array([[[1., 0., 2.], [0., 1., 2.]]])
    np.testing.assert_allclose(_project_segments(seg, np.eye(3)), [[[.5, 0], [0, .5]]])
    seg[0, 1, 2] = -1
    assert _project_segments(seg, np.eye(3)) is None


def test_score_colors_green_is_high_and_equal_scores_are_neutral():
    from efg.viz import score_colors
    colors = score_colors([.1, .3, .5])
    assert colors[0, 0] > colors[0, 1]
    assert colors[-1, 1] > colors[-1, 0]
    equal = score_colors([.2, .2])
    np.testing.assert_array_equal(equal[0], equal[1])
    assert equal[0, 0] > 200 and equal[0, 1] > 150


def test_outcome_does_not_emit_success_metric_or_grasp_colors(viz, monkeypatch):
    paths = []
    original = rr.log

    def log(path, *args, **kwargs):
        paths.append(path)
        original(path, *args, **kwargs)

    monkeypatch.setattr(rr, "log", log)
    for outcome in ("success", "fail", "abort"):
        viz.log_outcome(None, None, outcome)
    assert paths == ["status"] * 3


def test_gripper_tips_contact_at_zero_width_and_end_at_tcp():
    from efg.gripper_visual import gripper_parts
    parts = {name: mesh for name, mesh, _ in gripper_parts(.223)}
    left, right = parts['left/tip'], parts['right/tip']
    assert left.bounds[0, 1] == pytest.approx(0)
    assert right.bounds[1, 1] == pytest.approx(0)
    assert left.bounds[1, 2] == pytest.approx(0)
    assert all(m.is_watertight for m in parts.values())


@pytest.mark.parametrize("tip_length", [.080, .100])
def test_gripper_measured_finger_length_does_not_double_count_contact_pad(tip_length):
    from efg.gripper_visual import gripper_parts
    parts = {name: mesh for name, mesh, _ in gripper_parts(.203, tip_length)}
    assert parts['mount'].bounds[0, 2] == pytest.approx(-.203)
    for side in ('left', 'right'):
        finger, pad = parts[f'{side}/finger'], parts[f'{side}/tip']
        assert finger.bounds[0, 2] == pytest.approx(-tip_length)
        assert pad.bounds[1, 2] == pytest.approx(0)
        assert finger.bounds[1, 2] >= pad.bounds[0, 2]
    assert all(np.isfinite(m.vertices).all() and m.is_watertight for m in parts.values())
    assert max(m.bounds[1, 2] for m in parts.values()) == pytest.approx(0)
    assert min(m.bounds[0, 2] for m in parts.values()) == pytest.approx(-.203)


def test_display_keeps_overlapping_candidates_in_score_order_without_mutation():
    from efg.viz import display_candidate_indices
    scores = np.array([.3, .9, .5, .4])
    positions = np.array([[0,0,0], [.001,0,0], [.05,0,0], [.1,0,0]])
    group = SimpleNamespace(scores=scores, translations=positions)
    np.testing.assert_array_equal(display_candidate_indices(group, 2), [1, 2])
    np.testing.assert_array_equal(display_candidate_indices(group, 4), [1, 2, 3, 0])
    np.testing.assert_array_equal(scores, [.3, .9, .5, .4])


def test_live_cloud_has_its_own_capture_frame_and_does_not_move_scan(viz, monkeypatch):
    """Moving the live camera must leave the detector image/cloud frame fixed."""
    viz._live_cloud_history = 2
    calls = []
    monkeypatch.setattr(rr, "log", lambda path, *data, **kw: calls.append((path, data)))
    rgb = np.full((2, 2, 3), 128, np.uint8)
    depth = np.array([[1., 0.], [np.nan, np.inf]], dtype=np.float32)
    first, second = np.eye(4), np.eye(4)
    first[:3, 3], second[:3, 3] = [.1, .2, .3], [.5, -.1, .2]
    for slot, T in enumerate((first, second)):
        viz.log_live_cloud(rgb, depth, np.eye(3), T, slot, time.monotonic())
    clouds = [(path, data) for path, data in calls if path.startswith("world/")]
    assert [path for path, _ in clouds] == ["world/live_clouds/frame_0", "world/live_clouds/frame_1"]
    for (_, (transform, points)), T in zip(clouds, (first, second)):
        np.testing.assert_allclose(transform.translation.as_arrow_array().to_pylist()[0], T[:3, 3])
        np.testing.assert_allclose(points.positions.as_arrow_array().to_pylist(), [[0., 0., 1.]])
        assert len(points.colors.as_arrow_array()) == 1
    calls.clear()
    viz.log_live_cloud(rgb, np.zeros((2, 2)), np.eye(3), second, 0, time.monotonic())
    assert len(calls[0][1][1].positions.as_arrow_array()) == 0  # overwrite old slot with empty cloud
    calls.clear()
    viz.clear_live_clouds()
    assert [path for path, _ in calls] == ["world/live_clouds"]


def test_live_cloud_disabled_does_not_access_robot_or_camera(viz):
    from unittest.mock import Mock
    robot, camera = Mock(), Mock()
    viz.start_live_cloud_stream(camera, robot, np.eye(4))
    assert not robot.mock_calls and not camera.mock_calls


def test_live_cloud_cleanup_failure_does_not_block_other_shutdown(viz, capsys):
    from unittest.mock import Mock
    worker = Mock()
    worker.close.side_effect = RuntimeError("release callback failed")
    viz._live_cloud_stream = worker
    viz.close()
    assert viz._stop.is_set()
    assert viz._live_cloud_stream is None
    assert "라이브 점군 종료 실패" in capsys.readouterr().out
