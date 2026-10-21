"""xArm7 wrapper (xArm-Python-SDK, non-ROS).

Calibration records the FLANGE pose: the SDK reports the TCP pose including
any configured TCP offset, so the offset is read once and removed. This makes
the result (T_flange_cam) independent of whatever tool is mounted later.
"""

import time

import numpy as np

from .se3 import inv_se3, pose_delta, xarm_pose_to_se3


class XArm7:
    def __init__(self, ip):
        from xarm.wrapper import XArmAPI

        self.arm = XArmAPI(ip, is_radian=True)
        self.arm.clean_warn()
        self.arm.clean_error()

        # tcp_offset property: [x,y,z (mm), r,p,y (rad, is_radian=True 기준)]
        self.tcp_offset = [float(v) for v in self.arm.tcp_offset]
        self.T_flange_tcp = xarm_pose_to_se3(self.tcp_offset)
        if np.linalg.norm(self.tcp_offset) > 1e-6:
            print(f"[robot] TCP offset active {self.tcp_offset} — flange pose로 자동 보정합니다.")

    def get_tcp_pose(self):
        """T_base_tcp (4x4, meters). TCP = flange if no offset is configured."""
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed (code={code})")
        return xarm_pose_to_se3(pose), [float(v) for v in pose]

    def get_flange_pose(self):
        """T_base_flange (4x4, meters), with any TCP offset removed."""
        T_base_tcp, raw = self.get_tcp_pose()
        return T_base_tcp @ inv_se3(self.T_flange_tcp), raw

    def is_stationary(self, interval_s=0.3, max_pos_delta_mm=0.2, max_rot_delta_deg=0.05):
        T_a, _ = self.get_flange_pose()
        time.sleep(interval_s)
        T_b, _ = self.get_flange_pose()
        d_pos, d_rot = pose_delta(T_a, T_b)
        return d_pos * 1000.0 <= max_pos_delta_mm and d_rot <= max_rot_delta_deg

    # ------------------------------------------------------------------
    # Guarded motion (verify_touch.py / collect_samples_auto.py 전용)
    # ------------------------------------------------------------------
    def set_payload(self, weight_kg, com_mm=(0.0, 0.0, 0.0)):
        """TCP 페이로드(질량 kg / 무게중심 mm, 플랜지 기준)를 컨트롤러에 설정.

        충돌 감지(C31)의 이론 토크 계산에 쓰인다. Studio 설정과 별개로
        매 실행 시 적용되므로 서드파티 그리퍼도 안전하게 쓸 수 있다.
        """
        code = self.arm.set_tcp_load(float(weight_kg),
                                     [float(v) for v in com_mm])
        if code != 0:
            print(f"[robot] set_tcp_load 실패 (code={code}) — Studio에서 설정하세요")
        else:
            print(f"[robot] TCP payload 설정: {weight_kg}kg, CoM {list(com_mm)}mm")
        return code == 0

    def enable_motion(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)

    def move_tcp_linear(self, pose_mm_rad, speed_mm_s=30):
        """Slow straight-line TCP move. pose = [x,y,z (mm), r,p,y (rad)]."""
        code = self.arm.set_position(
            x=pose_mm_rad[0], y=pose_mm_rad[1], z=pose_mm_rad[2],
            roll=pose_mm_rad[3], pitch=pose_mm_rad[4], yaw=pose_mm_rad[5],
            speed=speed_mm_s, is_radian=True, wait=True,
        )
        if code != 0:
            raise RuntimeError(f"set_position failed (code={code})")

    def move_tcp_linear_async(self, pose_mm_rad, speed_mm_s=30):
        """비차단 직선 TCP 이동. is_moving()으로 폴링, stop_motion()으로 중단."""
        code = self.arm.set_position(
            x=pose_mm_rad[0], y=pose_mm_rad[1], z=pose_mm_rad[2],
            roll=pose_mm_rad[3], pitch=pose_mm_rad[4], yaw=pose_mm_rad[5],
            speed=speed_mm_s, is_radian=True, wait=False,
        )
        if code != 0:
            print(f"[robot] set_position 거부 (code={code}, "
                  f"controller error={self.arm.error_code}) — 에러 정리")
            self.arm.clean_error()
            self.arm.set_state(0)
        return code == 0

    def move_joints(self, angles_deg, speed_deg_s=20, acc_deg_s2=50):
        """조인트 목표(deg)로 차단 이동. 실패 시 예외.

        낮은 기본 가속도(50 deg/s^2): 페이로드 관성에 의한 토크 스파이크로
        충돌 감지(C31)가 오작동하는 것을 줄인다.
        """
        code = self.arm.set_servo_angle(
            angle=[float(a) for a in angles_deg],
            speed=float(speed_deg_s), mvacc=float(acc_deg_s2),
            is_radian=False, wait=True,
        )
        if code != 0:
            raise RuntimeError(f"set_servo_angle failed (code={code}, "
                               f"controller error={self.arm.error_code})")

    def _flange_to_tcp_pose(self, T_base_flange):
        """set_position은 TCP 기준이므로 플랜지 목표를 TCP 목표로 변환한다."""
        from .se3 import se3_to_xarm_pose

        return se3_to_xarm_pose(T_base_flange @ self.T_flange_tcp)

    def check_flange_ik(self, T_base_flange):
        """목표 플랜지 자세의 IK 가능 여부 (SDK 미지원 시 True로 통과)."""
        pose = self._flange_to_tcp_pose(T_base_flange)
        try:
            code, _ = self.arm.get_inverse_kinematics(
                pose, input_is_radian=True, return_is_radian=True)
        except AttributeError:
            return True
        return code == 0

    def flange_ik_joints_deg(self, T_base_flange):
        """목표 플랜지 자세의 IK 관절해 (deg 리스트). 실패/미지원 시 None.

        관절 이동량 비교(예: 등가 자세 중 손목 회전 최소 선택)에 사용한다.
        시드는 현재 관절 자세(SDK 내부)이다.
        """
        pose = self._flange_to_tcp_pose(T_base_flange)
        try:
            code, angles = self.arm.get_inverse_kinematics(
                pose, input_is_radian=True, return_is_radian=True)
        except AttributeError:
            return None
        if code != 0 or angles is None:
            return None
        return [float(np.degrees(a)) for a in angles[:7]]

    def move_flange(self, T_base_flange, speed_mm_s=60, acc_mm_s2=None):
        """플랜지를 목표 SE(3)로 직선 이동. 실패 시 예외 대신 False 반환."""
        pose = self._flange_to_tcp_pose(T_base_flange)
        code = self.arm.set_position(
            x=pose[0], y=pose[1], z=pose[2],
            roll=pose[3], pitch=pose[4], yaw=pose[5],
            speed=speed_mm_s, is_radian=True, wait=True,
            **({"mvacc": float(acc_mm_s2)} if acc_mm_s2 is not None else {}),
        )
        if code != 0:
            self.arm.clean_error()
            self.arm.set_state(0)
        return code == 0

    def move_flange_async(self, T_base_flange, speed_mm_s=60):
        """비차단 이동 시작. is_moving()으로 완료를 폴링하고,
        stop_motion()으로 언제든 즉시 중단할 수 있다."""
        pose = self._flange_to_tcp_pose(T_base_flange)
        code = self.arm.set_position(
            x=pose[0], y=pose[1], z=pose[2],
            roll=pose[3], pitch=pose[4], yaw=pose[5],
            speed=speed_mm_s, is_radian=True, wait=False,
        )
        if code != 0:
            print(f"[robot] set_position 거부 (code={code}, "
                  f"controller error={self.arm.error_code}) — 에러 정리")
            self.arm.clean_error()
            self.arm.set_state(0)
        return code == 0

    def move_joints_async(self, angles_deg, speed_deg_s=20, acc_deg_s2=50):
        """조인트 목표(deg)로 비차단 이동. is_moving()으로 폴링, stop_motion()으로 중단."""
        code = self.arm.set_servo_angle(
            angle=[float(a) for a in angles_deg],
            speed=float(speed_deg_s), mvacc=float(acc_deg_s2),
            is_radian=False, wait=False,
        )
        if code != 0:
            print(f"[robot] set_servo_angle 거부 (code={code}, "
                  f"controller error={self.arm.error_code}) — 에러 정리")
            self.arm.clean_error()
            self.arm.set_state(0)
        return code == 0

    def is_moving(self):
        code, state = self.arm.get_state()
        return code == 0 and state == 1  # 1 = IN_MOTION

    def stop_motion(self):
        """모션 즉시 정지 + 컨트롤러 명령 큐 제거.
        이후 다시 움직이려면 resume_after_stop() 을 호출해야 한다."""
        try:
            self.arm.set_state(4)  # 4 = STOP
        except Exception as e:  # 정지 요청은 어떤 상황에서도 예외로 죽으면 안 됨
            print(f"[robot] stop_motion 실패: {e} — 비상정지 버튼을 사용하세요")

    def resume_after_stop(self):
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_state(0)
        time.sleep(0.5)

    def disconnect(self):
        self.arm.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.disconnect()
