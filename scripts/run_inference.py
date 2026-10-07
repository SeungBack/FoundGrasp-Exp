#!/usr/bin/env python3
"""[점검] 검출만 실행 — 저장 프레임(npz) 또는 카메라 즉시 캡처 (로봇 불필요).

검출기 동작·설치 확인, 비교군 백엔드 오프라인 검증에 쓴다.
출력: 후보 수·상위 score, (옵션) 그리퍼 와이어프레임 오버레이 jpg, npy.

  python scripts/run_inference.py --npz tests/data/sample_frame.npz --vis outputs/grasps.jpg
  python scripts/run_inference.py --live --save-npz outputs/frame.npz
  python scripts/run_inference.py --npz ... --set detector.name=<other>   # 비교군
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from efg.config import add_config_args, config_from_args  # noqa: E402


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--npz", help="camera_preview.py(s) 저장본 또는 --save-npz 결과")
    ap.add_argument("--live", action="store_true", help="D435 즉시 캡처")
    ap.add_argument("--save-npz", help="캡처 프레임 저장 경로 (--live 시)")
    ap.add_argument("--out", default=None, help="후보 저장 npy (기본 output.dir/grasps.npy)")
    ap.add_argument("--vis", help="2D 오버레이 저장 경로 (jpg)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--roi", action="store_true", help="설정 scan_roi_px 로 ROI 마스킹·필터 적용")
    args = ap.parse_args()
    cfg = config_from_args(args)

    from efg.camera import load_frame_npz, resolve_intrinsics, save_frame_npz
    from efg.detectors import build_detector

    if args.live:
        from efg.camera import D435
        with D435.from_config(cfg) as cam:
            rgb, depth_m, K = cam.get_frame()
        if args.save_npz:
            Path(args.save_npz).parent.mkdir(parents=True, exist_ok=True)
            save_frame_npz(args.save_npz, rgb, depth_m, K)
            print(f"프레임 저장: {args.save_npz}")
    elif args.npz:
        rgb, depth_m, K = load_frame_npz(args.npz)
    else:
        ap.error("--npz 또는 --live 중 하나가 필요합니다")
    K = resolve_intrinsics(K, cfg)

    roi = cfg.get("scan_roi_px") if args.roi else None
    if roi:
        x1, y1, x2, y2 = [int(v) for v in roi]
        d = np.zeros_like(depth_m)
        d[y1:y2, x1:x2] = depth_m[y1:y2, x1:x2]
        depth_m = d

    det = build_detector(cfg)
    gg = det.detect(rgb, depth_m, K, top_k=args.top_k)
    if roi:
        from efg.roi import filter_roi
        n_all = len(gg)
        gg = filter_roi(gg, K, roi)
        print(f"ROI 필터: {n_all} → {len(gg)}")
    top = [round(float(gg[i].score), 3) for i in range(min(5, len(gg)))]
    print(f"[{det.name}] 그립 후보 {len(gg)}개 (score 내림차순). 상위 5: {top}")

    out = Path(args.out) if args.out else Path(cfg["output"]["dir"]) / "grasps.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    gg.save_npy(str(out))
    print(f"저장: {out}")
    if args.vis:
        import cv2

        from efg.roi import draw_overlay
        from efg.viz import draw_grippers
        vis = np.ascontiguousarray(rgb[:, :, ::-1]).copy()
        draw_overlay(vis, roi)
        draw_grippers(vis, gg, K, color=(0, 165, 255))
        if len(gg):
            draw_grippers(vis, gg[:1], K, color=(0, 0, 255), thickness=2)
        Path(args.vis).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.vis, vis)
        print(f"오버레이: {args.vis}")


if __name__ == "__main__":
    main()
