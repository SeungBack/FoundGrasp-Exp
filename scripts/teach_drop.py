#!/usr/bin/env python3
"""[사전 준비 3] 릴리즈(놓는 박스 상방) 자세 티칭.

xArm Studio(또는 프리드라이브)로 로봇을 릴리즈 자세로 옮긴 뒤 Enter — 현재
조인트 각도를 설정 파일의 drop.joints_deg 에 저장한다.
payload_check 를 쓰는 경우, 이 자세를 바꾸면 calib_payload.py 를 재실행할 것.

  python scripts/teach_drop.py
  python scripts/teach_drop.py --no-save   # 값 출력만
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from efg.config import add_config_args, config_from_args, set_yaml_line  # noqa: E402


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from efg.motion import connect_robot
    robot = connect_robot(cfg)
    try:
        input("로봇을 릴리즈 자세(놓는 박스 상방, 물체를 놓을 높이)로 옮긴 뒤 Enter... ")
        code, angles = robot.arm.get_servo_angle(is_radian=False)
        if code != 0:
            raise SystemExit(f"조인트 읽기 실패 (code={code})")
        joints = [round(float(a), 2) for a in angles[:7]]
        print(f"drop joints_deg: {joints}")
    finally:
        robot.disconnect()

    if args.no_save:
        return
    path = cfg["_config_path"]
    if set_yaml_line(path, "joints_deg", joints):
        print(f"저장됨: {path}")
    else:
        print(f"경고: {path}에서 joints_deg 라인을 찾지 못함 — 직접 기입하세요")


if __name__ == "__main__":
    main()
