"""카메라 연결 단절·정리·프레임 유효성 회귀 테스트 — 실제 장비 미사용."""
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import efg.camera as camera


class FakeCamera:
    def __init__(self, frames=(), close_error=None):
        self.frames = iter(frames)
        self.close_error = close_error
        self.close_calls = 0
        self.read_calls = 0

    def get_frame(self):
        self.read_calls += 1
        frame = next(self.frames)
        if isinstance(frame, Exception):
            raise frame
        return frame

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def frame():
    return (np.zeros((2, 3, 3), dtype=np.uint8),
            np.ones((2, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32))


def stream_with(monkeypatch, fake, **camera_cfg):
    monkeypatch.setattr(camera.D435, "from_config", lambda cfg: fake)
    return camera.CameraStream({"camera": camera_cfg}, live=False)


def test_failure_before_first_frame_reaches_snapshot(monkeypatch):
    failure = RuntimeError("Device disconnected")
    fake = FakeCamera([failure])
    stream = stream_with(monkeypatch, fake)

    stream.run()

    assert stream._finished.is_set()
    assert fake.close_calls == 1
    with pytest.raises(camera.CameraError, match="Device disconnected"):
        stream.snapshot(timeout=0)


def test_disconnect_invalidates_previously_captured_frame(monkeypatch):
    fake = FakeCamera([frame(), RuntimeError("USB disconnected")])
    stream = stream_with(monkeypatch, fake)

    stream.run()

    assert fake.read_calls == 2
    assert stream._latest is None
    with pytest.raises(camera.CameraError, match="USB disconnected"):
        stream.snapshot(timeout=0)


def test_cleanup_failure_preserves_capture_failure(monkeypatch):
    failure = RuntimeError("capture failed first")
    fake = FakeCamera([failure], close_error=RuntimeError("cleanup failed later"))
    stream = stream_with(monkeypatch, fake)

    stream.run()

    assert stream._finished.is_set()
    assert stream._error is failure
    with pytest.raises(camera.CameraError, match="capture failed first"):
        stream.raise_if_failed()


def test_non_runtime_capture_failure_also_propagates(monkeypatch):
    fake = FakeCamera([ValueError("bad image buffer")])
    stream = stream_with(monkeypatch, fake)

    stream.run()

    assert fake.close_calls == 1
    with pytest.raises(camera.CameraError, match="bad image buffer"):
        stream.raise_if_failed()


def test_normal_stop_cleans_up_without_error_output(monkeypatch, capsys):
    fake = FakeCamera()
    stream = stream_with(monkeypatch, fake)
    stream._stop_evt.set()

    stream.run()

    assert fake.read_calls == 0
    assert fake.close_calls == 1
    assert stream._error is None
    assert stream._finished.is_set()
    assert "오류" not in capsys.readouterr().out
    with pytest.raises(camera.CameraError):
        stream.snapshot(timeout=0)


@pytest.mark.parametrize("flag", ["_stop_evt", "_finished"])
def test_stopped_stream_rejects_cached_frame(monkeypatch, flag):
    stream = stream_with(monkeypatch, FakeCamera())
    stream._latest = frame()
    stream._latest_at = time.monotonic()
    getattr(stream, flag).set()

    with pytest.raises(camera.CameraError):
        stream.snapshot(timeout=0)


def test_snapshot_initial_wait_is_bounded(monkeypatch):
    stream = stream_with(monkeypatch, FakeCamera())
    started = time.monotonic()

    with pytest.raises(camera.CameraError):
        stream.snapshot(timeout=0.01)

    assert time.monotonic() - started < 0.5


def test_snapshot_returns_independent_array_copies(monkeypatch):
    stream = stream_with(monkeypatch, FakeCamera())
    originals = frame()
    stream._latest = originals
    stream._latest_at = time.monotonic()

    copied = stream.snapshot(timeout=0)

    for original, result in zip(originals, copied):
        np.testing.assert_array_equal(result, original)
        assert not np.shares_memory(result, original)


def test_stale_frame_is_rejected_before_sdk_reports_failure(monkeypatch):
    stream = stream_with(monkeypatch, FakeCamera(), max_frame_age_s=0.25)
    stream._latest = frame()
    stream._latest_at = time.monotonic() - 1.0

    # SDK 내부 재연결이 아직 반환하지 않아도 오래된 입력으로 움직이면 안 된다.
    assert stream._error is None
    assert not stream._finished.is_set()
    with pytest.raises(camera.CameraError):
        stream.raise_if_failed()
    with pytest.raises(camera.CameraError):
        stream.snapshot(timeout=0)


class FakePipeline:
    def __init__(self, stop_error=None):
        self.stop_error = stop_error
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error


@pytest.mark.parametrize("stop_error", [
    None,
    RuntimeError("stop() cannot be called before start()"),
])
def test_camera_close_is_idempotent_after_disconnect(stop_error):
    device = camera.D435.__new__(camera.D435)
    device.pipeline = FakePipeline(stop_error)
    device._started = True

    device.close()
    device.close()

    assert not device._started
    assert device.pipeline.stop_calls == 1


def test_camera_close_preserves_unrelated_runtime_error():
    device = camera.D435.__new__(camera.D435)
    device.pipeline = FakePipeline(RuntimeError("unexpected driver failure"))
    device._started = True

    with pytest.raises(RuntimeError, match="unexpected driver failure"):
        device.close()


def test_failed_warmup_closes_pipeline_without_masking_error(monkeypatch):
    pipeline = FakePipeline(RuntimeError("cleanup also failed"))
    intrinsics = SimpleNamespace(fx=1, fy=1, ppx=0, ppy=0)
    video_profile = SimpleNamespace(get_intrinsics=lambda: intrinsics)
    profile = SimpleNamespace(
        get_device=lambda: SimpleNamespace(
            first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001)),
        get_stream=lambda stream: SimpleNamespace(
            as_video_stream_profile=lambda: video_profile),
    )
    pipeline.start = lambda cfg: profile
    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=lambda: SimpleNamespace(enable_stream=lambda *args: None),
        stream=SimpleNamespace(color=1, depth=2),
        format=SimpleNamespace(bgr8=1, z16=2),
        align=lambda stream: object(),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    def fail_warmup(self):
        raise RuntimeError("warmup device disconnected")

    monkeypatch.setattr(camera.D435, "_raw_frame", fail_warmup)

    with pytest.raises(RuntimeError, match="warmup device disconnected"):
        camera.D435(postprocess=False, warmup_frames=1)

    assert pipeline.stop_calls == 1


def test_frame_wait_times_out_without_sdk_auto_reconnect():
    device = camera.D435.__new__(camera.D435)
    device.frame_timeout_s = 0.02
    device.pipeline = SimpleNamespace(poll_for_frames=lambda: None)
    start = time.monotonic()
    with pytest.raises(camera.CameraError, match="수신 시간 초과"):
        device._raw_frame()
    assert time.monotonic() - start < 0.5


def test_poll_wait_delivers_frame_to_alignment_and_filters():
    device = camera.D435.__new__(camera.D435)
    device.frame_timeout_s = 0.5
    frames = SimpleNamespace(get_depth_frame=lambda: "depth", get_color_frame=lambda: "color")
    incoming = iter([None, frames])
    device.pipeline = SimpleNamespace(poll_for_frames=lambda: next(incoming))
    device.align = SimpleNamespace(process=lambda f: f)
    device.filters = [SimpleNamespace(process=lambda d: d + "_filtered")]
    assert device._raw_frame() == ("color", "depth_filtered")
