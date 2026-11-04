"""Live preview uses synthetic reports/frames, without robot or camera hardware."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from efg.live_pointcloud import LivePointCloudStream, _interpolate_tcp
from efg.se3 import xarm_pose_to_se3


def frame(stamp, sequence):
    return (np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.float32),
            np.eye(3), stamp, sequence)


@pytest.fixture
def scene():
    tcp = np.eye(4)
    tcp[2, 3] = .203
    camera_offset = np.eye(4)
    camera_offset[:3, 3] = [.12, -.03, -.025]
    arm = SimpleNamespace(connected=True, register_report_callback=Mock(return_value=True),
                          release_report_callback=Mock())
    robot = SimpleNamespace(arm=arm, T_flange_tcp=tcp)
    viz = SimpleNamespace(log_live_cloud=Mock(), clear_live_clouds=Mock(), log_text=Mock())
    camera = SimpleNamespace(preview_snapshot=Mock(return_value=None))
    stream = LivePointCloudStream(viz, camera, robot, camera_offset,
                                  history=3, hz=5, stride=4)
    stream._last_logged_at = 10.
    return SimpleNamespace(stream=stream, viz=viz, camera=camera, robot=robot,
                           camera_offset=camera_offset)


def test_interpolation_uses_short_rotation_across_angle_wrap():
    a = [0., 0., 0., 0., 0., np.radians(179)]
    b = [100., 20., 40., 0., 0., np.radians(-179)]
    T = _interpolate_tcp([(1., a), (1.2, b)], 1.1)
    np.testing.assert_allclose(T[:3, 3], [.05, .01, .02])
    np.testing.assert_allclose(T[:3, :3], np.diag([-1., -1., 1.]), atol=1e-12)


@pytest.mark.parametrize('stamp', [.9, 1.3])
def test_pose_is_not_extrapolated(stamp):
    assert _interpolate_tcp([(1., [0.] * 6), (1.2, [0.] * 6)], stamp) is None


def test_missing_reports_do_not_bridge_large_time_gap():
    assert _interpolate_tcp([(1., [0.] * 6), (1.5, [0.] * 6)], 1.2) is None


def test_live_frame_keeps_capture_pose_and_removes_tcp_offset(scene):
    s = scene
    first = [300., 0., 500., np.pi, 0., 0.]
    last = [400., 20., 600., np.pi, 0., 0.]
    s.stream._reports.extend([(10., first), (10.2, last)])
    captured = frame(10.1, 1)
    s.camera.preview_snapshot.return_value = captured
    s.stream._tick(10.2)
    call = s.viz.log_live_cloud.call_args.args
    expected_tcp = xarm_pose_to_se3([350., 10., 550., np.pi, 0., 0.])
    tcp_inverse = np.eye(4)
    tcp_inverse[2, 3] = -.203
    np.testing.assert_allclose(call[3], expected_tcp @ tcp_inverse @ s.camera_offset)
    assert call[4:] == (0, 10.1)
    assert call[0] is captured[0]
    s.camera.preview_snapshot.assert_called_once_with(after_sequence=0, stride=4)


def test_pending_frame_waits_for_next_report_instead_of_using_current_pose(scene):
    s = scene
    s.stream._reports.append((10., [0.] * 6))
    s.camera.preview_snapshot.return_value = frame(10.1, 1)
    s.stream._tick(10.1)
    s.viz.log_live_cloud.assert_not_called()
    s.stream._tick(10.15)
    assert s.camera.preview_snapshot.call_count == 1
    s.stream._reports.append((10.2, [100., 0., 0., 0., 0., 0.]))
    s.stream._tick(10.2)
    assert s.viz.log_live_cloud.call_count == 1
    assert s.viz.log_live_cloud.call_args.args[-1] == 10.1


def test_frame_published_during_snapshot_copy_waits_until_next_poll(scene):
    s = scene
    s.stream._reports.extend([(10., [0.] * 6), (10.2, [0.] * 6)])
    s.camera.preview_snapshot.return_value = frame(10.101, 1)
    s.stream._tick(10.1)
    s.viz.log_live_cloud.assert_not_called()
    assert s.stream._pending is not None
    s.stream._tick(10.15)
    s.viz.log_live_cloud.assert_called_once()


def test_ring_overwrites_only_one_slot_per_new_frame_and_skips_duplicates(scene):
    s = scene
    for index in range(7):
        stamp = 10. + index * .25
        s.stream._reports.append((stamp, [0.] * 6))
        s.camera.preview_snapshot.return_value = frame(stamp, index + 1)
        s.stream._tick(stamp)
    assert [call.args[4] for call in s.viz.log_live_cloud.call_args_list] == [0, 1, 2, 0, 1, 2, 0]
    s.stream._tick(11.8)
    assert s.viz.log_live_cloud.call_count == 7
    assert s.camera.preview_snapshot.call_args.kwargs['after_sequence'] == 7


def test_stale_frames_clear_live_history_once_and_do_not_replace_fixed_scan(scene):
    s = scene
    s.stream._reports.append((10., [0.] * 6))
    s.camera.preview_snapshot.return_value = frame(10., 1)
    s.stream._tick(10.)
    s.camera.preview_snapshot.return_value = frame(10., 2)
    s.stream._tick(10.7)
    s.stream._tick(10.9)
    assert s.viz.log_live_cloud.call_count == 1
    s.viz.clear_live_clouds.assert_called_once_with()
    assert s.viz.log_text.call_count == 1


def test_closed_stream_does_not_log_or_read_frames_and_releases_only_own_callback(scene):
    s = scene
    s.stream.start()
    s.stream.close()
    s.stream.close()
    assert not s.stream._thread.is_alive()
    s.robot.arm.release_report_callback.assert_called_once_with(s.stream._report)
    s.viz.log_live_cloud.reset_mock()
    s.camera.preview_snapshot.reset_mock()
    s.stream._tick(11.)
    s.stream._report({'cartesian': [0.] * 6})
    s.viz.log_live_cloud.assert_not_called()
    s.camera.preview_snapshot.assert_not_called()
    assert not s.stream._reports


def test_callback_copies_only_finite_pose_and_retains_bounded_history(scene):
    s = scene
    pose = [0.] * 6
    s.stream._report({'cartesian': pose})
    pose[0] = 99.
    assert s.stream._reports[-1][1][0] == 0.
    for invalid in ({}, {'cartesian': [1.]}, {'cartesian': [np.nan] * 6}):
        s.stream._report(invalid)
    assert len(s.stream._reports) == 1
    for _ in range(300):
        s.stream._report({'cartesian': pose})
    assert len(s.stream._reports) == 256


def test_callback_registration_failure_does_not_start_worker(scene):
    scene.robot.arm.register_report_callback.return_value = False
    with pytest.raises(RuntimeError, match='보고 등록 실패'):
        scene.stream.start()
    assert scene.stream._thread is None
    scene.stream.close()
    scene.robot.arm.release_report_callback.assert_not_called()


def test_failed_thread_start_can_be_closed_and_releases_callback(scene, monkeypatch):
    monkeypatch.setattr('efg.live_pointcloud.threading.Thread.start',
                        Mock(side_effect=RuntimeError('thread unavailable')))
    with pytest.raises(RuntimeError, match='thread unavailable'):
        scene.stream.start()
    scene.stream.close()
    scene.robot.arm.release_report_callback.assert_called_once_with(scene.stream._report)
    assert scene.stream._thread is None


def test_disconnect_clears_history_without_hardware_queries(scene):
    s = scene
    s.stream._visible = True
    s.robot.arm.connected = False
    s.stream._tick(10.1)
    s.viz.clear_live_clouds.assert_called_once_with()
    s.camera.preview_snapshot.assert_not_called()


def test_worker_camera_failure_is_contained_and_cleanup_still_releases_callback(scene):
    import threading
    s = scene
    warned = threading.Event()
    s.camera.preview_snapshot.side_effect = RuntimeError('preview copy failed')
    s.viz.log_text.side_effect = lambda *_a, **_k: warned.set()
    s.stream._visible = True
    s.stream.start()
    try:
        assert warned.wait(1)
        assert s.stream._thread.is_alive()
    finally:
        s.stream.close()
    s.viz.clear_live_clouds.assert_called_once_with()
    assert s.viz.log_text.call_count == 1
    s.robot.arm.release_report_callback.assert_called_once_with(s.stream._report)
