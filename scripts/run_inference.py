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
    ap.add_argument("--repeat", type=int, default=1, help="Measured repetitions on the same frame")
    ap.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup calls")
    ap.add_argument("--record-dir", help="Save effective config, checkpoint hash, frame, candidates, timings")
    args = ap.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        ap.error("--repeat >= 1 and --warmup >= 0 required")
    cfg = config_from_args(args)
    if args.no_roi:
        cfg['scan_roi_px'] = None
    cfg['_offline'] = {'frame': args.npz, 'warmup': args.warmup, 'repeat': args.repeat}

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

    from efg.perception import detect_frame
    from efg.recording import write_manifest, save_detection

    roi = None if args.no_roi else cfg.get("scan_roi_px")
    det = build_detector(cfg)
    results = []
    try:
        if args.record_dir:
            write_manifest(cfg, Path(args.record_dir) / "manifest.json", detector=det)
        for _ in range(args.warmup):
            detect_frame(det, rgb, depth_m, K, cfg, roi)
        for index in range(args.repeat):
            result = detect_frame(det, rgb, depth_m, K, cfg, roi)
            results.append(result)
            if args.record_dir:
                save_detection(Path(args.record_dir) / f"frame{index:03d}", rgb, depth_m, K, result)
    finally:
        det.close()
    gg, counts = result.grasps, result.counts
    ms, post_ms = result.detect_ms, result.post_ms
    for field, label in (("detect_ms", "검출(통신 포함)"), ("inference_ms", "워커 내부 추론"), ("post_ms", "공통 후처리")):
        values = [getattr(r, field) for r in results if getattr(r, field) is not None]
        if values:
            print(f"  {label}: median {np.median(values):.1f} ms · min {min(values):.1f} · max {max(values):.1f} · n={len(values)}")
    top = [round(float(gg[i].score), 3) for i in range(min(5, len(gg)))]
    print(f"[{det.describe()}] 검출 {ms:.0f}ms + 후처리 {post_ms:.0f}ms"
          f" = {ms + post_ms:.0f}ms | 단계별 후보 {counts}")
    print(f"  상위 5 score: {top}")
    if len(gg):
        w = gg.widths
        print(f"  후보 폭 mm: 중앙값 {np.median(w) * 1000:.0f}, 범위 {w.min() * 1000:.0f}–{w.max() * 1000:.0f}")

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
