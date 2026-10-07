#!/usr/bin/env python3
"""[사전 준비 2, 자동] 스캔(홈) 자세 정렬 — 카메라를 박스 중심 상방에 수직으로.

스캔 자세가 박스에 대해 x,y 평면상 틀어져 있으면 그립 자세의 수직성에 영향을
준다. 이 스크립트는:
  1) 현재 홈에서 캡처해 ROI(박스) 중심의 base 좌표를 구하고
  2) 카메라 광축이 정확히 수직(base -z)이고 박스 중심 바로 위(현재 높이 유지)에
     오는 자세를 계산해 (yaw는 현재와 최소 변화)
  3) IK 확인 후 이동, 실제 도달 자세의 조인트를 robot.home_joints_deg 에 저장한다.

저장 후 이 자세에서 보는 이미지가 달라지므로 반드시 set_roi.py 를 다시 실행할 것.

  python scripts/align_scan_pose.py
  python scripts/align_scan_pose.py --no-save   # 이동·수치 확인만
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from efg.config import add_config_args, config_from_args, require, set_yaml_line  # noqa: E402


def vertical_cam_pose(T_base_cam_now, box_center_base):
    """목표 T_base_cam: 광축 수직 하향, 박스 중심 상방(현재 카메라 높이 유지).

    카메라 광학 프레임: z=광축(전방). z_new = [0,0,-1](수직 아래),
    x_new = 현재 카메라 x축의 수평 성분 (yaw 최소 변화).
    """
    z = np.array([0.0, 0.0, -1.0])
    x = T_base_cam_now[:3, 0].copy()
    x[2] = 0.0
    n = np.linalg.norm(x)
    x = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else x / n
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = [box_center_base[0], box_center_base[1], T_base_cam_now[2, 3]]
    return T


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    roi = require(cfg, "scan_roi_px", "scripts/set_roi.py 로 스캔 영역을 지정하세요")

    from efg.camera import D435, resolve_intrinsics
    from efg.motion import connect_robot, go_home
    from efg.se3 import inv_se3
    from efg.transforms import load_hand_eye

    T_flange_cam = load_hand_eye(cfg["calib"]["hand_eye"])
    robot = connect_robot(cfg)
    try:
        print("현재 스캔 자세(홈)로 이동 후 캡처...")
        go_home(robot, cfg)
        T_base_flange, _ = robot.get_flange_pose()
        T_base_cam = T_base_flange @ T_flange_cam

        with D435.from_config(cfg) as cam:
            _, depth_m, K = cam.get_frame()
        K = resolve_intrinsics(K, cfg)

        # ROI 중심 픽셀 + ROI 유효 depth 중앙값 → 박스 중심(base)
        x1, y1, x2, y2 = [int(v) for v in roi]
        d_roi = depth_m[y1:y2, x1:x2]
        d_valid = d_roi[d_roi > 0]
        if d_valid.size < 100:
            raise SystemExit("ROI 내 유효 depth가 부족합니다 — 장면/카메라 확인")
        z_med = float(np.median(d_valid))
        u_c, v_c = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        p_cam = z_med * np.linalg.solve(K, np.array([u_c, v_c, 1.0]))
        p_base = T_base_cam[:3, :3] @ p_cam + T_base_cam[:3, 3]
        print(f"박스 중심(base): {np.round(p_base * 1000, 1).tolist()} mm "
              f"(ROI depth 중앙값 {z_med * 1000:.0f}mm)")

        tilt_now = np.degrees(np.arccos(np.clip(-T_base_cam[:3, 2] @ [0, 0, 1], -1, 1)))
        off_now = np.linalg.norm(T_base_cam[:2, 3] - p_base[:2]) * 1000
        print(f"현재: 광축 기울임 {tilt_now:.1f}°, 중심 수평 오프셋 {off_now:.0f}mm")

        T_cam_new = vertical_cam_pose(T_base_cam, p_base)
        T_flange_new = T_cam_new @ inv_se3(T_flange_cam)
        if not robot.check_flange_ik(T_flange_new):
            raise SystemExit("목표 정렬 자세 IK 불가 — 박스/로봇 배치 확인")

        print("정렬 자세로 이동...")
        if not robot.move_flange(T_flange_new, speed_mm_s=cfg["robot"]["travel_speed_mm_s"]):
            raise SystemExit("정렬 자세 이동 실패")

        T_bf, _ = robot.get_flange_pose()
        T_bc = T_bf @ T_flange_cam
        tilt = np.degrees(np.arccos(np.clip(-T_bc[:3, 2] @ [0, 0, 1], -1, 1)))
        off = np.linalg.norm(T_bc[:2, 3] - p_base[:2]) * 1000
        print(f"도달: 광축 기울임 {tilt:.2f}°, 중심 수평 오프셋 {off:.0f}mm")

        code, angles = robot.arm.get_servo_angle(is_radian=False)
        if code != 0:
            raise SystemExit(f"조인트 읽기 실패 (code={code})")
        joints = [round(float(a), 2) for a in angles[:7]]
        print(f"정렬 스캔 조인트: {joints}")
    finally:
        robot.disconnect()

    if args.no_save:
        return
    path = cfg["_config_path"]
    if set_yaml_line(path, "home_joints_deg", joints):
        print(f"저장됨: {path}")
        print("주의: 스캔 시점이 바뀌었으니 set_roi.py 를 다시 실행하세요 "
              "(로봇이 정렬 자세에 있는 지금 바로 실행하면 됨).")
    else:
        print(f"경고: {path}에서 home_joints_deg 라인을 찾지 못함 — 직접 기입하세요")


if __name__ == "__main__":
    main()
