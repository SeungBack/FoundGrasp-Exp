#!/usr/bin/env python3
"""[사전 준비 1] 스캔 영역(ROI) 지정 — 카메라 프레임에서 마우스 드래그.

물체 박스 내부를 드래그 → Space/Enter 확정, c 취소. 결과는 설정 파일의
scan_roi_px 에 저장된다. 스캔 자세를 바꾸면(2단계) 다시 실행할 것.

  python scripts/set_roi.py
  python scripts/set_roi.py --no-save   # 값 출력만
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from efg.config import add_config_args, config_from_args, set_yaml_line  # noqa: E402


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--no-save", action="store_true", help="설정 파일에 저장하지 않음")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from efg.camera import D435
    print("카메라 캡처 중...")
    with D435.from_config(cfg) as cam:
        rgb, _, _ = cam.get_frame()
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])

    print("물체 박스 내부를 드래그하세요 (Space/Enter 확정, c 취소)")
    x, y, w, h = cv2.selectROI("set_roi (drag inside the box)", bgr,
                               showCrosshair=True)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("취소됨 — ROI 미지정")
        return
    roi = [int(x), int(y), int(x + w), int(y + h)]
    print(f"scan_roi_px: {roi}")

    if args.no_save:
        return
    path = cfg["_config_path"]
    if set_yaml_line(path, "scan_roi_px", roi):
        print(f"저장됨: {path}")
    else:
        print(f"경고: {path}에서 scan_roi_px 라인을 찾지 못함 — 직접 기입하세요")


if __name__ == "__main__":
    main()
