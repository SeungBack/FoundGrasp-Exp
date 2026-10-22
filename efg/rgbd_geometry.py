"""Calibrated RGB-D -> pinhole image geometry, before inference or display.

The aligned depth stores axial Z, so rectification with R=I only resamples its
pixels. Nearest-neighbour depth avoids synthesising surfaces across depth edges.
The output keeps the calibrated K and image size, with zero distortion.
"""
from pathlib import Path

import cv2
import numpy as np
import yaml

from .paths import resolve


class RGBDCalibration:
    def __init__(self, calib):
        self.rectified = bool(calib.get('undistort', False))
        self.K = self.D = self.size = None
        self._maps = None
        path = resolve(calib.get('intrinsics'))
        if path is None or not Path(path).is_file():
            if self.rectified:
                raise ValueError('calib.undistort=true requires a calibrated intrinsics file')
            return
        data = yaml.safe_load(Path(path).read_text())
        if 'T_flange_cam' in data:
            data = data['camera_intrinsics']
        self.K = np.asarray(data['K'], dtype=np.float64)
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or min(self.K[0, 0], self.K[1, 1]) <= 0
                or not np.allclose(self.K[2], [0, 0, 1])):
            raise ValueError('Invalid calibrated K')
        if not self.rectified:
            return
        self.D = np.asarray(data['D'], dtype=np.float64).reshape(-1)
        if len(self.D) not in (4, 5, 8, 12, 14) or not np.isfinite(self.D).all():
            raise ValueError('Invalid OpenCV distortion coefficients')
        # calibrate_intrinsics.py uses cv2.calibrateCamera, but copies the old
        # factory distortion_model label. Its fitted coefficients are OpenCV's.
        model = data.get('projection_model', data.get('distortion_model', ''))
        if str(data.get('source', '')).startswith('calibrate_intrinsics.py'):
            model = 'opencv_brown_conrady'
        if model not in ('opencv_brown_conrady', 'brown_conrady', 'plumb_bob'):
            raise ValueError(f'Unsupported calibrated distortion model: {model!r}')
        self.size = (int(data['width']), int(data['height']))
        if min(self.size) <= 0:
            raise ValueError('Invalid calibration image size')

    def process(self, rgb, depth, K_stream):
        K = np.asarray(self.K if self.K is not None else K_stream, dtype=np.float32)
        if not self.rectified:
            return rgb, depth, K
        if rgb.shape[:2] != depth.shape or rgb.shape[1::-1] != self.size:
            raise ValueError(f'RGB-D size does not match calibration {self.size}')
        if self._maps is None:
            self._maps = cv2.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, self.size, cv2.CV_32FC1)
        rgb = cv2.remap(rgb, *self._maps, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        depth = cv2.remap(depth, *self._maps, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.ascontiguousarray(rgb), np.ascontiguousarray(depth), K
