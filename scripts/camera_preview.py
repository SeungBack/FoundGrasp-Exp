#!/usr/bin/env python3
"""[점검] D435 라이브 프리뷰 — 컬러 + depth 컬러맵을 OpenCV 창으로 표시.

  python scripts/camera_preview.py
  # q/ESC 종료, s 현재 프레임 npz 저장 (outputs/preview_*.npz → run_inference.py --npz)
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time

import cv2
import numpy as np

from efg.camera import D435, save_frame_npz  # noqa: E402
from efg.config import add_config_args, config_from_args  # noqa: E402


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--depth-max", type=float, default=1.0, help="depth 컬러맵 상한 (m)")
    args = ap.parse_args()
    cfg = config_from_args(args)

    with D435.from_config(cfg, warmup_frames=1) as cam:
        print("카메라 프리뷰 실행 중 — q/ESC 종료, s 프레임 저장")
        t_prev, fps = time.time(), 0.0
        while True:
            rgb, depth_m, K = cam.get_frame()
            t_now = time.time()
            fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            bgr = rgb[:, :, ::-1].copy()
            d8 = np.clip(depth_m / args.depth_max * 255, 0, 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
            depth_vis[depth_m <= 0.05] = 0  # 무효 depth는 검정
            valid = (depth_m > 0.1).mean() * 100
            cv2.putText(bgr, f"{fps:.0f} fps  depth valid {valid:.0f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("D435 preview (color | depth)", np.hstack([bgr, depth_vis]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = Path(cfg["output"]["dir"]) / time.strftime("preview_%Y%m%d_%H%M%S.npz")
                out.parent.mkdir(parents=True, exist_ok=True)
                save_frame_npz(str(out), rgb, depth_m, K)
                print(f"저장: {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
