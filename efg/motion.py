"""로봇 연결·기본 이동 헬퍼 (robot.XArm7 위의 얇은 계층)."""
import numpy as np

from .robot import XArm7
from .se3 import inv_se3


def connect_robot(cfg):
    robot = XArm7(cfg["robot"]["ip"])
    payload = cfg.get("robot", {}).get("tcp_payload") or {}
    if payload.get("weight_kg"):
        robot.set_payload(payload["weight_kg"],
                          payload.get("com_mm", (0.0, 0.0, 0.0)))
    return robot


def go_home(robot, cfg):
    r = cfg["robot"]
    robot.enable_motion()
    robot.move_joints(r["home_joints_deg"],
                      speed_deg_s=r["home_joint_speed_deg_s"])


def move_tcp(robot, T_base_tcp, speed_mm_s, arrive_tol_mm=10.0):
    """툴(TCP) 목표 SE(3)로 직선 이동 + 도착 검증.

    이동 호출이 성공을 반환해도 실제 플랜지 자세를 읽어 목표와 비교한다
    (컨트롤러 상태 이상으로 이동 없이 반환되는 경우 차단).
    """
    T_flange = T_base_tcp @ inv_se3(robot.T_flange_tcp)
    if not robot.move_flange(T_flange, speed_mm_s=speed_mm_s):
        return False
    T_actual, _ = robot.get_flange_pose()
    err_mm = float(np.linalg.norm(
        (T_actual[:3, 3] - T_flange[:3, 3]) * 1000.0))
    if err_mm > arrive_tol_mm:
        print(f"경고: 목표 미도달 (오차 {err_mm:.1f}mm > {arrive_tol_mm}mm) — "
              "이동 실패로 처리")
        return False
    return True
