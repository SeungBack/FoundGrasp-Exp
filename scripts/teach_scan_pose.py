#!/usr/bin/env python3
"""[사전 준비 2, 수동] 스캔(홈) 자세 티칭 — 카메라 라이브 화면을 보며 조정.

xArm Studio(또는 프리드라이브)로 로봇을 움직이며 라이브 화면(중앙 빨간 점 =
이미지 센터)으로 시점을 맞추고, 원하는 스캔 자세에서 s 키 — 현재 조인트를
설정 파일의 robot.home_joints_deg 에 저장한다. (자동 정렬: align_scan_pose.py)

저장 후 스캔 시점이 바뀌므로 반드시 set_roi.py 를 다시 실행할 것.

  python scripts/teach_scan_pose.py
  python scripts/teach_scan_pose.py --no-save   # 화면·조인트 확인만
창 키: s = 저장 후 종료 / q(ESC) = 저장 없이 종료
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from efg.config import add_config_args, config_from_args, set_yaml_line  # noqa: E402

WINDOW = "teach_scan_pose (s: save / q: quit)"


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from efg.camera import D435
    from efg.motion import connect_robot

    robot = connect_robot(cfg)
    joints = None
    try:
        print("로봇을 움직여 화면으로 스캔 시점을 맞추세요 (xArm Studio 조그/프리드라이브).")
        print("창에서 s = 저장, q/ESC = 취소")
        with D435.from_config(cfg) as cam:
            while True:
                rgb, _, _ = cam.get_frame()
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                h, w = bgr.shape[:2]
                cv2.circle(bgr, (w // 2, h // 2), 4, (0, 0, 255), -1)  # 센터 점
                cv2.imshow(WINDOW, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("취소됨 — 저장하지 않음")
                    return
                if key == ord("s"):
                    break
        code, angles = robot.arm.get_servo_angle(is_radian=False)
        if code != 0:
            raise SystemExit(f"조인트 읽기 실패 (code={code})")
        joints = [round(float(a), 2) for a in angles[:7]]
        print(f"스캔 조인트: {joints}")
    finally:
        cv2.destroyAllWindows()
        robot.disconnect()

    if args.no_save or joints is None:
        return
    path = cfg["_config_path"]
    if set_yaml_line(path, "home_joints_deg", joints):
        print(f"저장됨: {path}")
        print("주의: 스캔 시점이 바뀌었으니 set_roi.py 를 다시 실행하세요.")
    else:
        print(f"경고: {path}에서 home_joints_deg 라인을 찾지 못함 — 직접 기입하세요")


if __name__ == "__main__":
    main()
