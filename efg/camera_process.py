"""RealSense 전용 프로세스: SDK/GIL/종료를 추론 프로세스에서 격리한다.

spawn으로 SDK 객체를 자식 안에서 생성한다. 최신 RGBD 한 장만 공유 메모리에
보관하고 생산 시각으로 freshness를 검사한다. USB 오류 후 자동 재개하지 않는다.
"""
import multiprocessing as mp
import queue
import time

import numpy as np

from .camera import CameraError, D435


def _arrays(buffers, height, width):
    rgb, depth, K = buffers
    return (np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3),
            np.frombuffer(depth, dtype=np.float32).reshape(height, width),
            np.frombuffer(K, dtype=np.float32).reshape(3, 3))


def _capture(cfg, live, buffers, lock, updated, sequence, stop, quit_flag,
             errors, controls, camera_factory):
    cam = None
    cv2 = None
    failed = False

    def report(exc):
        nonlocal failed
        if not failed:
            failed = True
            try:
                errors.send(f"{type(exc).__name__}: {exc}"[:2000])
            except (BrokenPipeError, OSError):
                pass

    try:
        # 이 프로세스는 torch/Open3D/Rerun을 import하지 않는다.
        cam = camera_factory(cfg)
        c = cfg["camera"]
        target = _arrays(buffers, c["height"], c["width"])
        status, marks = "INIT", []
        if live:
            import cv2
            from .roi import draw_overlay
        while not stop.is_set():
            frame = cam.get_frame()
            captured_at = time.monotonic()
            with lock:
                for dst, src in zip(target, frame):
                    np.copyto(dst, src)
                updated.value = captured_at
                sequence.value += 1
            if live:
                try:
                    status, marks = controls.get_nowait()
                except queue.Empty:
                    pass
                bgr = np.ascontiguousarray(frame[0][:, :, ::-1])
                draw_overlay(bgr, cfg.get("scan_roi_px"), status, marks)
                cv2.imshow("exp_foundgrasp live", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    quit_flag.value = True
    except BaseException as exc:
        report(exc)  # close()가 멈추더라도 부모에게 먼저 오류를 전달한다.
    finally:
        if cam is not None:
            try:
                cam.close()
            except BaseException as exc:
                report(exc)
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        errors.close()


class ProcessCameraStream:
    """CameraStream의 데모 인터페이스를 유지하는 프로세스 기반 최신 프레임 공급자."""

    def __init__(self, cfg, live=True, *, camera_factory=None):
        self.cfg, self.live = cfg, live
        c = cfg["camera"]
        self._max_age = float(c.get("max_frame_age_s", 2.0))
        self._shutdown_timeout = float(c.get("shutdown_timeout_s", 10.0))
        if self._max_age <= 0 or self._shutdown_timeout < 0:
            raise ValueError("camera timeout 설정이 유효하지 않습니다")
        ctx = mp.get_context("spawn")
        height, width = int(c["height"]), int(c["width"])
        self._buffers = (ctx.RawArray("B", height * width * 3),
                         ctx.RawArray("f", height * width), ctx.RawArray("f", 9))
        self._arrays = _arrays(self._buffers, height, width)
        self._lock = ctx.Lock()
        self._updated = ctx.RawValue("d", 0)
        self._sequence = ctx.RawValue("Q", 0)
        self._quit = ctx.RawValue("b", False)
        self._stop = ctx.Event()
        self._errors, error_sender = ctx.Pipe(duplex=False)
        self._error_sender = error_sender
        self._controls = ctx.Queue(maxsize=1)
        self._status, self._marks = "INIT", []
        self._error = None
        self._started = self._closed = False
        self.forced_shutdown = False
        self._process = ctx.Process(
            name="realsense-capture", target=_capture,
            args=(cfg, live, self._buffers, self._lock, self._updated,
                  self._sequence, self._stop, self._quit, error_sender,
                  self._controls, camera_factory or D435.from_config), daemon=True)

    @property
    def quit_requested(self):
        return bool(self._quit.value)

    def start(self):
        if self._started or self._closed:
            raise CameraError("카메라 프로세스는 재시작할 수 없습니다")
        self._process.start()
        self._started = True
        self._error_sender.close()

    def _check(self):
        if self._closed or not self._started:
            raise CameraError("카메라 프로세스가 실행 중이 아닙니다")
        if self._error is None and self._errors.poll():
            try:
                self._error = self._errors.recv()
            except EOFError:
                self._error = "카메라 프로세스 연결이 종료되었습니다"
        if self._error is not None:
            raise CameraError(f"카메라 스트림 실패: {self._error}")
        if not self._process.is_alive():
            raise CameraError(f"카메라 프로세스 종료 (exit={self._process.exitcode})")
        if self._sequence.value:
            age = time.monotonic() - self._updated.value
            if age > self._max_age:
                raise CameraError(f"카메라 프레임 갱신 중단 ({age:.1f}초)")

    def raise_if_failed(self):
        self._check()

    def snapshot(self, timeout=5.0, fresh=False):
        deadline = time.monotonic() + timeout
        after_sequence = self._sequence.value if fresh else 0
        while True:
            self._check()
            if self._sequence.value > after_sequence:
                # 자식이 버퍼를 쓰다가 죽어도 부모가 락에서 영원히 대기하지 않는다.
                if not self._lock.acquire(timeout=0.25):
                    self._check()
                    raise CameraError("카메라 공유 프레임 읽기 시간 초과")
                try:
                    self._check()
                    result = tuple(a.copy() for a in self._arrays)
                finally:
                    self._lock.release()
                self._check()
                return result
            if time.monotonic() >= deadline:
                raise CameraError(f"카메라 최초 프레임 대기 시간 초과 ({timeout:g}초)")
            time.sleep(0.02)

    def _send_overlay(self):
        if not self.live or self._closed:
            return
        try:
            self._controls.get_nowait()
        except queue.Empty:
            pass
        try:
            self._controls.put_nowait((self._status, self._marks))
        except queue.Full:
            pass  # 미리보기 갱신이 캡처/제어 흐름을 막지 않도록 한다.

    def set_status(self, text):
        self._status = text
        self._send_overlay()

    def set_marks(self, marks):
        self._marks = list(marks)
        self._send_overlay()

    def stop(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._started:
                self._stop.set()
                self._process.join(timeout=self._shutdown_timeout)
                if self._process.is_alive():
                    self.forced_shutdown = True
                    self._process.terminate()
                    self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1)
                if self._process.is_alive():
                    raise CameraError(
                        f"카메라 프로세스 종료 대기 시간 초과 (pid={self._process.pid}, "
                        "SIGKILL 전달 후에도 생존). 커널/USB 응답 대기 여부를 확인하세요.")
        finally:
            self._error_sender.close()
            self._errors.close()
            self._controls.cancel_join_thread()
            self._controls.close()
