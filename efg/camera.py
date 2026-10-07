"""RealSense D435 캡처 + 라이브 스트림 스레드.

반환 규격: color RGB uint8 (H,W,3) / depth float32 meters (H,W) /
K (3,3) color 스트림 intrinsics. depth는 color에 정렬(align)된다.
depth 후처리(camera.postprocess): disparity→spatial→temporal→depth 체인,
파라미터는 pyrealsense2 기본값 (temporal Holes Fill=3 포함 — 정지 직후
캡처는 잔상이 섞일 수 있어 스캔 전 짧은 대기를 둔다).
"""
import threading
import time

import numpy as np


class D435:
    def __init__(self, width=1280, height=720, fps=30,
                 postprocess=True, warmup_frames=15):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = (profile.get_device().first_depth_sensor()
                            .get_depth_scale())

        self.filters = []
        if postprocess:
            self.filters = [
                rs.disparity_transform(True),
                rs.spatial_filter(),
                rs.temporal_filter(),
                rs.disparity_transform(False),
            ]

        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]], dtype=np.float32)

        for _ in range(warmup_frames):
            self._raw_frame()

    @classmethod
    def from_config(cls, cfg, warmup_frames=None):
        c = cfg["camera"]
        return cls(c["width"], c["height"], c["fps"],
                   postprocess=c["postprocess"],
                   warmup_frames=(c["warmup_frames"] if warmup_frames is None
                                  else warmup_frames))

    def _raw_frame(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth = frames.get_depth_frame()
        for f in self.filters:
            depth = f.process(depth)
        return frames.get_color_frame(), depth

    def get_frame(self):
        color, depth = self._raw_frame()
        bgr = np.asanyarray(color.get_data())
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        depth_m = (np.asanyarray(depth.get_data()).astype(np.float32)
                   * self.depth_scale)
        return rgb, depth_m, self.K

    def close(self):
        self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def resolve_intrinsics(K_stream, cfg):
    """calib.intrinsics(캘리브레이션 K)가 있으면 그것을, 없으면 스트림 K.

    hand-eye 캘리브레이션과 같은 K를 써야 역투영 좌표가 일관된다.
    """
    from pathlib import Path

    import yaml

    p = cfg.get("calib", {}).get("intrinsics")
    if not p:
        return K_stream
    path = Path(p)
    if not path.exists():
        print(f"경고: calib.intrinsics 파일 없음 ({path}) — 스트림 K 사용")
        return K_stream
    with open(path) as f:
        K = np.asarray(yaml.safe_load(f)["K"], dtype=np.float32)
    print(f"intrinsics: 캘리브레이션 K 사용 (fx {K[0, 0]:.1f}, {path.name})")
    return K


def save_frame_npz(path, rgb, depth_m, K):
    np.savez_compressed(path, rgb=rgb, depth_m=depth_m, K=K)


def load_frame_npz(path):
    d = np.load(path)
    return d["rgb"], d["depth_m"], d["K"]


class CameraStream(threading.Thread):
    """카메라 스트림 스레드 + 라이브 뷰(OpenCV).

    카메라는 이 스레드가 단독 소유한다 (pyrealsense2 다중 스레드 접근 회피).
    메인 루프는 snapshot()으로 최신 프레임을 받아 추론에 쓰고, set_status()/
    set_marks()로 오버레이를 갱신한다. 라이브 창은 로봇 이동 중에도 갱신된다.
    """
    WINDOW = "exp_foundgrasp live"

    def __init__(self, cfg, live=True):
        super().__init__(daemon=True)
        self.cam = D435.from_config(cfg)
        self.live = live
        self.roi_px = cfg.get("scan_roi_px")
        self._lock = threading.Lock()
        self._latest = None
        self._status = "INIT"
        self._marks = []
        self._stop_evt = threading.Event()  # 이름 주의: Thread._stop() 가림 방지
        self.quit_requested = False  # 라이브 창에서 q 입력 시 True

    def run(self):
        import cv2

        from .roi import draw_overlay

        while not self._stop_evt.is_set():
            try:
                rgb, depth_m, K = self.cam.get_frame()
            except RuntimeError as e:
                print(f"카메라 프레임 오류: {e}")
                break
            with self._lock:
                self._latest = (rgb, depth_m, K)
                status, marks = self._status, list(self._marks)
            if self.live:
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                draw_overlay(bgr, self.roi_px, status, marks)
                cv2.imshow(self.WINDOW, bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self.quit_requested = True
        self.cam.close()
        if self.live:
            cv2.destroyAllWindows()

    def snapshot(self):
        """최신 프레임 (rgb, depth_m, K) 사본. 아직 없으면 대기."""
        while True:
            with self._lock:
                if self._latest is not None:
                    rgb, d, K = self._latest
                    return rgb.copy(), d.copy(), K.copy()
            time.sleep(0.05)

    def set_status(self, text):
        with self._lock:
            self._status = text

    def set_marks(self, marks):
        with self._lock:
            self._marks = list(marks)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)
