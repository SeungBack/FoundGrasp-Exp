#!/usr/bin/env python3
"""[점검] 검출만 실행 — 저장 프레임(npz) 또는 카메라 즉시 캡처 (로봇 불필요).

검출기 동작·설치 확인, 비교군 백엔드 오프라인 검증에 쓴다. 실행 루프와 같은
입력(ROI depth 마스킹)·공유 후처리(폭 한계·충돌필터·정렬)·ROI 필터를 적용한다.
출력: 단계별 후보 수·상위 score, (옵션) 그리퍼 와이어프레임 오버레이 jpg, npy.

  python scripts/run_inference.py --npz tests/data/sample_frame.npz --vis outputs/grasps.jpg
  python scripts/run_inference.py --live --save-npz outputs/frame.npz
  python scripts/run_inference.py --npz ... --set detector.name=<other>   # 비교군
  python scripts/run_inference.py --npz ... --set detector.foundgrasp.refiner.enabled=false
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
    ap.add_argument("--no-roi", action="store_true", help="ROI 마스킹·필터 생략 (전체 프레임)")
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

    import time

    from efg.postprocess import mask_depth_to_roi, shared_filter
    from efg.roi import filter_roi

    roi = None if args.no_roi else cfg.get("scan_roi_px")
    depth_in = mask_depth_to_roi(depth_m, roi) if cfg["postprocess"].get("roi_depth_mask", True) else depth_m
    det = build_detector(cfg)
    try:
        det.detect(rgb, depth_in, K)  # 워밍업 (CUDA 초기화·캐시) — 시간 측정에서 제외
        t0 = time.perf_counter()
        gg = det.detect(rgb, depth_in, K)
        ms = (time.perf_counter() - t0) * 1000.0
    finally:
        det.close()
    post_t0 = time.perf_counter()
    gg, counts = shared_filter(gg, depth_m, K, cfg)
    if roi:
        gg = filter_roi(gg, K, roi)
        counts["roi"] = len(gg)
    post_ms = (time.perf_counter() - post_t0) * 1000.0
    top = [round(float(gg[i].score), 3) for i in range(min(5, len(gg)))]
    print(f"[{det.describe()}] 검출 {ms:.0f}ms + 후처리 {post_ms:.0f}ms"
          f" = {ms + post_ms:.0f}ms | 단계별 후보 {counts}")
    print(f"  상위 5 score: {top}")
    if len(gg):
        w = gg.widths
        print(f"  폭(raw) mm: 중앙값 {np.median(w) * 1000:.0f}, 범위 {w.min() * 1000:.0f}–{w.max() * 1000:.0f}")

    out = Path(args.out) if args.out else Path(cfg["output"]["dir"]) / "grasps.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    gg.save_npy(str(out))
    print(f"저장: {out}")
    if args.vis:
        import cv2

        from efg.roi import draw_overlay
        from efg.viz import draw_score_grippers
        vis = np.ascontiguousarray(rgb[:, :, ::-1]).copy()
        draw_overlay(vis, roi)
        draw_score_grippers(vis, gg, K, max_n=int(cfg.get("rerun", {}).get("max_candidates", 25)))
        Path(args.vis).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.vis, vis)
        print(f"오버레이: {args.vis}")


if __name__ == "__main__":
    main()
