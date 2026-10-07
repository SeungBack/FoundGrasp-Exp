#!/usr/bin/env python3
"""[사전 준비, 선택] 빈 박스 스캔 → 하강 제한 높이맵 생성 (floor_limit).

물체 박스를 비운 상태에서 실행 — 스캔 자세(홈)로 이동해 여러 프레임의 depth
중앙값으로 ROI 영역을 base 좌표 2.5D 높이맵(셀별 최고 표면 z)으로 변환해
저장한다. 실행 시 floor_limit.enabled: true 면 이 맵 기준으로 그리퍼가 박스
바닥·벽 아래로 내려가지 않도록 삽입량을 제한한다. 경사진 벽도 셀별 최고 z로
보수적으로 반영된다.

박스 위치·스캔 자세·ROI가 바뀌면 다시 실행할 것.

  python scripts/scan_box_floor.py
  python scripts/scan_box_floor.py --frames 25
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from efg.config import add_config_args, config_from_args, require  # noqa: E402

CELL_M = 0.005  # 높이맵 격자 (5mm)


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--frames", type=int, default=15, help="중앙값용 프레임 수")
    args = ap.parse_args()
    cfg = config_from_args(args)
    roi = require(cfg, "scan_roi_px", "scripts/set_roi.py 로 스캔 영역을 지정하세요")

    from efg.camera import D435, resolve_intrinsics
    from efg.motion import connect_robot, go_home
    from efg.transforms import load_hand_eye

    input("물체 박스를 비운 상태인지 확인 후 Enter (로봇이 스캔 자세로 이동)... ")
    robot = connect_robot(cfg)
    try:
        go_home(robot, cfg)
        T_base_flange, _ = robot.get_flange_pose()
    finally:
        robot.disconnect()
    T_base_cam = T_base_flange @ load_hand_eye(cfg["calib"]["hand_eye"])

    print(f"{args.frames} 프레임 캡처 중...")
    depths = []
    with D435.from_config(cfg) as cam:
        for _ in range(args.frames):
            _, d, K = cam.get_frame()
            depths.append(d)
    K = resolve_intrinsics(K, cfg)

    # 픽셀별 유효 depth 중앙값 (전 프레임 무효면 0)
    stack = np.stack(depths)
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    med = np.nan_to_num(med, nan=0.0)

    x1, y1, x2, y2 = [int(v) for v in roi]
    z_px = med[y1:y2, x1:x2]
    valid = z_px > 0
    print(f"ROI 유효 depth: {valid.mean() * 100:.1f}%")
    if valid.mean() < 0.5:
        raise SystemExit("ROI 유효 depth 50% 미만 — 카메라/장면 확인")

    # ROI 픽셀 → base 좌표 점군
    us, vs = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    z = z_px[valid]
    u, v = us[valid], vs[valid]
    x_c = (u - K[0, 2]) * z / K[0, 0]
    y_c = (v - K[1, 2]) * z / K[1, 1]
    p_base = (np.stack([x_c, y_c, z], axis=1) @ T_base_cam[:3, :3].T
              + T_base_cam[:3, 3])

    # xy 격자 — 셀별 최고 표면 z (벽·바닥의 보수적 상한)
    x0, y0 = p_base[:, 0].min(), p_base[:, 1].min()
    nx = int(np.ceil((p_base[:, 0].max() - x0) / CELL_M)) + 1
    ny = int(np.ceil((p_base[:, 1].max() - y0) / CELL_M)) + 1
    grid = np.full((ny, nx), -np.inf)
    j = np.clip(((p_base[:, 0] - x0) / CELL_M).round().astype(int), 0, nx - 1)
    i = np.clip(((p_base[:, 1] - y0) / CELL_M).round().astype(int), 0, ny - 1)
    np.maximum.at(grid, (i, j), p_base[:, 2])

    # 빈 셀 채움: 이웃 최고값 전파(보수적) → 남으면 전체 최고값
    empty0 = ~np.isfinite(grid)
    for _ in range(5):
        if not (~np.isfinite(grid)).any():
            break
        pad = np.pad(grid, 1, constant_values=-np.inf)
        neigh = np.max(np.stack([
            pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:],
        ]), axis=0)
        grid = np.where(np.isfinite(grid), grid, neigh)
    grid[~np.isfinite(grid)] = grid[np.isfinite(grid)].max()

    floor_z = float(np.percentile(grid, 10))
    print(f"높이맵 {ny}x{nx} (셀 {CELL_M * 1000:.0f}mm), 빈 셀 {empty0.mean() * 100:.1f}% 채움")
    print(f"바닥 추정 z(10퍼센타일): {floor_z * 1000:.1f}mm, "
          f"최고(벽 상단) z: {grid.max() * 1000:.1f}mm")

    out = Path(cfg["floor_limit"]["map_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, z=grid.astype(np.float32), x0=x0, y0=y0,
                        cell=CELL_M, roi=np.array(roi))
    print(f"저장됨: {out}")
    print("설정에서 floor_limit.enabled: true 로 활성화하세요 "
          "(또는 run.py --set floor_limit.enabled=true).")


if __name__ == "__main__":
    main()
