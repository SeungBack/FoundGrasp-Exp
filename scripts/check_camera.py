#!/usr/bin/env python3
"""카메라 전용 프로세스의 연속 캡처 확인. 로봇/그리퍼/추론기/Rerun 미사용.

  python scripts/check_camera.py --seconds 60
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efg.camera import CameraError
from efg.camera_process import ProcessCameraStream
from efg.config import add_config_args, config_from_args


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds는 양수여야 합니다")
    stream = ProcessCameraStream(config_from_args(args), live=False)
    result = 0
    try:
        stream.start()
        rgb, depth, _ = stream.snapshot(timeout=10)
        print(f"카메라 연결: RGB {rgb.shape}, depth {depth.shape}", flush=True)
        start = time.monotonic()
        next_report = 5.0
        while time.monotonic() - start < args.seconds:
            stream.snapshot()
            elapsed = time.monotonic() - start
            if elapsed >= next_report:
                print(f"프레임 갱신 정상: {elapsed:.0f}초", flush=True)
                next_report += 5
            time.sleep(0.1)
        print(f"연속 캡처 {args.seconds:g}초 통과", flush=True)
    except CameraError as exc:
        print(f"카메라 실패: {exc}", flush=True)
        result = 1
    except KeyboardInterrupt:
        result = 130
    finally:
        stop_started = time.monotonic()
        try:
            stream.stop()
        except CameraError as exc:
            print(f"카메라 종료 실패: {exc}", flush=True)
            result = 1
        else:
            if stream.forced_shutdown:
                print("SDK 응답 지연으로 카메라 프로세스를 강제 종료했습니다.", flush=True)
            else:
                print(f"카메라 정상 종료: {time.monotonic() - stop_started:.2f}초", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
