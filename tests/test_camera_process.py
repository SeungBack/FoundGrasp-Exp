"""실제 spawn/IPC로 SDK 멈춤·비정상 종료를 모의한다. 하드웨어는 사용하지 않는다."""
import os
import time

import numpy as np
import pytest

from efg.camera import CameraError
from efg.camera_process import ProcessCameraStream


class FakeCamera:
    def __init__(self, cfg):
        self.mode = cfg["camera"].get("test_mode", "normal")
        self.n = 0
        if self.mode == "init_error":
            raise RuntimeError("device start failed")
        if self.mode == "init_hang":
            time.sleep(60)

    def get_frame(self):
        self.n += 1
        if self.n > 1:
            if self.mode == "read_hang":
                time.sleep(60)
            if self.mode == "disconnect":
                time.sleep(0.1)
                raise RuntimeError("USB disconnected")
            if self.mode == "native_crash":
                time.sleep(0.1)
                os._exit(23)
        time.sleep(0.01)
        v = self.n % 200
        return (np.full((2, 3, 3), v, np.uint8),
                np.full((2, 3), v, np.float32), np.eye(3, dtype=np.float32))

    def close(self):
        if self.mode == "close_hang":
            time.sleep(60)


def make_camera(cfg):
    return FakeCamera(cfg)


@pytest.fixture
def make_stream():
    streams = []

    def make(mode="normal"):
        cfg = {"camera": {"width": 3, "height": 2, "max_frame_age_s": 0.3,
                          "shutdown_timeout_s": 0.2, "test_mode": mode}}
        stream = ProcessCameraStream(cfg, live=False, camera_factory=make_camera)
        streams.append(stream)
        stream.start()
        return stream

    yield make
    for stream in streams:
        stream.stop()


def test_latest_frame_has_consistent_owned_buffers(make_stream):
    stream = make_stream()
    rgb, depth, K = stream.snapshot()
    np.testing.assert_array_equal(rgb[:, :, 0], depth)
    np.testing.assert_array_equal(K, np.eye(3))
    old = rgb.copy()
    time.sleep(0.08)
    rgb2, depth2, _ = stream.snapshot()
    assert rgb2[0, 0, 0] != rgb[0, 0, 0]
    np.testing.assert_array_equal(rgb[:, :, 0], old[:, :, 0])
    np.testing.assert_array_equal(rgb2[:, :, 0], depth2)
    stream.stop()
    assert stream._process.exitcode == 0
    assert not stream.forced_shutdown
    with pytest.raises(CameraError):
        stream.snapshot()


def test_startup_exception_reaches_parent(make_stream):
    stream = make_stream("init_error")
    with pytest.raises(CameraError, match="device start failed"):
        stream.snapshot()


def test_disconnect_rejects_cached_frame(make_stream):
    stream = make_stream("disconnect")
    stream.snapshot()
    time.sleep(0.25)
    with pytest.raises(CameraError, match="USB disconnected"):
        stream.snapshot()


def test_native_crash_is_contained_in_child(make_stream):
    stream = make_stream("native_crash")
    stream.snapshot()
    stream._process.join(timeout=2)
    assert stream._process.exitcode == 23
    with pytest.raises(CameraError):
        stream.raise_if_failed()


def test_read_hang_rejects_stale_data_and_shutdown_is_bounded(make_stream):
    stream = make_stream("read_hang")
    stream.snapshot()
    time.sleep(0.4)
    with pytest.raises(CameraError, match="갱신 중단"):
        stream.raise_if_failed()
    start = time.monotonic()
    stream.stop()
    assert time.monotonic() - start < 2
    assert stream.forced_shutdown
    assert not stream._process.is_alive()
    stream.stop()  # 반복 종료도 안전하다.


def test_close_hang_cannot_keep_parent_alive(make_stream):
    stream = make_stream("close_hang")
    stream.snapshot()
    start = time.monotonic()
    stream.stop()
    assert time.monotonic() - start < 2
    assert stream.forced_shutdown
    assert not stream._process.is_alive()


def test_startup_hang_has_bounded_wait_and_stop(make_stream):
    stream = make_stream("init_hang")
    with pytest.raises(CameraError, match="최초 프레임"):
        stream.snapshot(timeout=0.05)
    start = time.monotonic()
    stream.stop()
    assert time.monotonic() - start < 2
    assert not stream._process.is_alive()


def test_shutdown_failure_still_closes_ipc():
    # SIGKILL 직후에도 살아 있는 커널 I/O 대기 상황을 모의한다.
    from unittest.mock import Mock

    stream = ProcessCameraStream({"camera": {"width": 3, "height": 2}}, live=False)
    stream._process = Mock(pid=1234)
    stream._process.is_alive.return_value = True
    stream._started = True
    with pytest.raises(CameraError, match="pid=1234"):
        stream.stop()
    stream._process.kill.assert_called_once()
    assert stream._errors.closed
    assert stream._error_sender.closed
    assert stream.forced_shutdown


def test_camera_check_reports_capture_and_shutdown_failures(monkeypatch, capsys):
    from scripts import check_camera
    from unittest.mock import Mock

    stream = Mock()
    stream.snapshot.side_effect = CameraError("capture failed")
    stream.stop.side_effect = CameraError("shutdown failed")
    monkeypatch.setattr(check_camera, "ProcessCameraStream", lambda *a, **k: stream)
    monkeypatch.setattr("sys.argv", ["check_camera.py"])
    assert check_camera.main() == 1
    output = capsys.readouterr().out
    assert "capture failed" in output
    assert "shutdown failed" in output
    assert "강제 종료했습니다" not in output


def test_fresh_snapshot_waits_for_new_capture(make_stream):
    stream = make_stream()
    rgb, _, _ = stream.snapshot()
    newer, _, _ = stream.snapshot(fresh=True)
    assert int(newer[0, 0, 0]) > int(rgb[0, 0, 0])
