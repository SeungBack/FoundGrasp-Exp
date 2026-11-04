"""Bounded live point-cloud preview, independent of grasp capture and control.

Camera frame availability and robot report reception share the host monotonic
clock. Interpolation aligns those timestamps, not sensor exposure times; motion
can therefore leave some registration error. This is visualization only.
"""
from collections import deque
import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .se3 import inv_se3, xarm_pose_to_se3


MAX_REPORT_GAP_S = .3
MAX_FRAME_AGE_S = .6


def _interpolate_tcp(reports, stamp):
    """Interpolate bracketing reports; never extrapolate through missing data."""
    if not reports or stamp < reports[0][0] or stamp > reports[-1][0]:
        return None
    for index, (at, pose) in enumerate(reports):
        if at == stamp:
            return xarm_pose_to_se3(pose)
        if at > stamp:
            before, pose_before = reports[index - 1]
            if at - before > MAX_REPORT_GAP_S:
                return None
            fraction = (stamp - before) / (at - before)
            first, last = xarm_pose_to_se3(pose_before), xarm_pose_to_se3(pose)
            result = np.eye(4)
            result[:3, 3] = first[:3, 3] * (1 - fraction) + last[:3, 3] * fraction
            result[:3, :3] = Slerp(
                [0., 1.], Rotation.from_matrix(np.stack([first[:3, :3], last[:3, :3]]))
            )(fraction).as_matrix()
            return result
    return None


class LivePointCloudStream:
    """Observe reports and an existing camera buffer; send no hardware commands."""

    def __init__(self, viz, camera, robot, T_flange_cam, *, history, hz, stride):
        if (isinstance(history, bool) or not isinstance(history, int) or history < 1
                or isinstance(stride, bool) or not isinstance(stride, int) or stride < 1
                or not math.isfinite(hz) or hz <= 0):
            raise ValueError("라이브 점군 history/stride는 양의 정수, hz는 유한한 양수여야 합니다")
        self.viz, self.camera, self.robot = viz, camera, robot
        self.history, self.hz, self.stride = history, float(hz), stride
        self._tcp_to_cam = inv_se3(np.asarray(robot.T_flange_tcp)) @ np.asarray(T_flange_cam)
        self._reports = deque(maxlen=256)
        self._report_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._registered = False
        self._pending = None
        self._sequence = 0
        self._slot = 0
        self._next_sample = 0.
        self._started_at = time.monotonic()
        self._last_logged_at = self._started_at
        self._visible = False
        self._last_error = None

    def _report(self, data):
        # Keep the SDK report thread free of array work, Rerun and camera locks.
        if self._stop.is_set():
            return
        try:
            pose = tuple(float(value) for value in data["cartesian"])
            if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
                return
            with self._report_lock:
                self._reports.append((time.monotonic(), pose))
        except (KeyError, TypeError, ValueError):
            return

    def start(self):
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("라이브 점군 스트림은 재시작할 수 없습니다")
        self._registered = bool(self.robot.arm.register_report_callback(
            self._report, report_cartesian=True, report_joints=False,
            report_state=False, report_error_code=False, report_warn_code=False,
            report_mtable=False, report_mtbrake=False, report_cmd_num=False))
        if not self._registered:
            raise RuntimeError("라이브 점군용 로봇 자세 보고 등록 실패")
        self._started_at = self._last_logged_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="live-pointcloud", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            try:
                self.robot.arm.release_report_callback(self._report)
            finally:
                self._registered = False
            raise

    def _unavailable(self, message):
        if self._stop.is_set():
            return
        if self._visible:
            self.viz.clear_live_clouds()
            self._visible = False
        if message != self._last_error:
            self.viz.log_text(f"라이브 점군 갱신 대기: {message}", level="WARN")
            self._last_error = message

    def _tick(self, now):
        if self._stop.is_set():
            return
        if not self.robot.arm.connected:
            self._pending = None
            self._unavailable("로봇 연결 끊김")
            return
        if self._pending is None and now >= self._next_sample:
            self._next_sample = now + 1. / self.hz
            frame = self.camera.preview_snapshot(after_sequence=self._sequence, stride=self.stride)
            if frame is not None:
                sequence = int(frame[4])
                if sequence > self._sequence:
                    self._pending = frame
                    self._sequence = sequence
        if self._pending is not None:
            rgb, depth, K, stamp, _ = self._pending
            if not math.isfinite(stamp) or abs(now - stamp) > MAX_FRAME_AGE_S:
                self._pending = None
            elif stamp <= now:
                with self._report_lock:
                    reports = tuple(self._reports)
                tcp = _interpolate_tcp(reports, stamp)
                if tcp is not None and not self._stop.is_set():
                    self.viz.log_live_cloud(rgb, depth, K, tcp @ self._tcp_to_cam,
                                            self._slot, stamp)
                    self._slot = (self._slot + 1) % self.history
                    self._last_logged_at = stamp
                    self._visible = True
                    self._last_error = None
                    self._pending = None
        if now - self._last_logged_at > MAX_FRAME_AGE_S:
            self._unavailable("최신 카메라 프레임과 로봇 자세 보고의 시각을 맞출 수 없음")

    def _run(self):
        while not self._stop.is_set():
            try:
                self._tick(time.monotonic())
            except Exception as exc:
                self._pending = None
                try:
                    self._unavailable(f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass  # Rendering failure must not affect robot control.
            self._stop.wait(min(.05, 1. / self.hz))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.)
        if self._registered:
            try:
                self.robot.arm.release_report_callback(self._report)
            finally:
                self._registered = False
