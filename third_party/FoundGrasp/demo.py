#!/usr/bin/env python3
import argparse
import os
import time
import urllib.parse

import cv2
import numpy as np

from grasp_detector import FoundGraspDetector, estimate_intrinsics
from utils.visualization import log_grasps_to_rerun, visualize_and_save_opencv


def load_depth_anything_v3(model_name="depth-anything/da3metric-large", device="cuda"):
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        print("Depth Anything 3 is not installed.")
        print("Install it with: pip install git+https://github.com/ByteDance-Seed/depth-anything-3")
        raise

    model = DepthAnything3.from_pretrained(model_name)
    return model.to(device=device)


def estimate_depth_da3(model, image_path):
    prediction = model.inference([image_path])
    depth = prediction.depth[0]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32)

    if hasattr(prediction, "intrinsics") and prediction.intrinsics is not None:
        intrinsics = prediction.intrinsics[0]
        if hasattr(intrinsics, "detach"):
            intrinsics = intrinsics.detach().cpu().numpy()
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
    else:
        h, w = depth.shape
        intrinsics = estimate_intrinsics(h, w)

    focal_length = (intrinsics[0, 0] + intrinsics[1, 1]) / 2.0
    depth = focal_length / 300.0 * depth / 1.8
    return depth.astype(np.float32), intrinsics


def load_color(path):
    color_bgr = cv2.imread(path)
    if color_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def load_depth(path):
    depth = np.load(path).astype(np.float32)
    if depth.max() > 10.0:
        depth = depth / 1000.0
    return depth


def resolve_inference_options(args, detector):
    inf_cfg = detector.config.get("inference", {})

    if args.collision_voxel_size is None:
        collision_voxel_size = float(inf_cfg.get("collision_voxel_size", 0.01))
    else:
        collision_voxel_size = args.collision_voxel_size

    return {
        "top_k": args.top_k,
        "nms": args.nms,
        "use_collision_detection": args.use_collision_detection,
        "collision_voxel_size": collision_voxel_size,
    }


def setup_rerun(app_id, web_port=9090, grpc_port=9876):
    import rerun as rr

    try:
        rr.init(app_id, recording_id=app_id)
    except TypeError:
        rr.init(app_id)
    rr.serve_grpc(grpc_port=grpc_port, server_memory_limit="1GB")
    rr.serve_web_viewer(open_browser=False, web_port=web_port)
    server_uri = f"rerun+http://127.0.0.1:{grpc_port}/proxy"
    url = f"http://127.0.0.1:{web_port}/?url={urllib.parse.quote(server_uri, safe='')}"
    print(f"Rerun viewer: {url}")
    return rr


def log_detection_to_rerun(detector, gg, frame_idx):
    import rerun as rr

    rr.set_time("frame_idx", sequence=frame_idx)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    color = detector.last_color
    xyz = detector.last_xyz
    if color is not None:
        rr.log("world/camera/color", rr.Image(color))
    if xyz is not None and color is not None:
        points = xyz.reshape(-1, 3)
        colors = color.reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1)
        if valid.any():
            rr.log("world/point_cloud", rr.Points3D(points[valid], colors=colors[valid], radii=0.001))
    log_grasps_to_rerun(gg)


def save_outputs(detector, gg, image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    vis_result = None
    if detector.last_vis_sample is not None:
        vis_result = visualize_and_save_opencv(detector.last_vis_sample, save_dir=None, point_size=9)
        stem = os.path.basename(image_path).rsplit(".", 1)[0]
        vis_path = os.path.join(output_dir, f"{stem}_vis.jpg")
        cv2.imwrite(vis_path, vis_result["all_data_grid"])
        print(f"Saved visualization to: {vis_path}")

    stem = os.path.basename(image_path).rsplit(".", 1)[0]
    grasp_path = os.path.join(output_dir, f"{stem}_grasps.npy")
    gg.save_npy(grasp_path)
    print(f"Saved grasps to: {grasp_path}")
    return vis_result


def log_realsense_detection(args, detector, gg, frame_idx):
    if args.rerun:
        log_detection_to_rerun(detector, gg, frame_idx)


def wait_for_rerun_if_needed(args):
    if args.rerun:
        input("Press Enter to exit and close the Rerun server...")


def run_image(args, detector):
    color = load_color(args.image)

    if args.depth:
        depth = load_depth(args.depth)
        intrinsics = estimate_intrinsics(depth.shape[0], depth.shape[1])
    else:
        depth_model = load_depth_anything_v3(args.depth_model, args.device)
        depth, intrinsics = estimate_depth_da3(depth_model, args.image)

    print(f"Depth range: {depth.min():.3f} - {depth.max():.3f} m")
    gg = detector.detect(color, depth, intrinsics, **resolve_inference_options(args, detector))
    print(f"Found {len(gg)} grasps")

    save_outputs(detector, gg, args.image, args.output)

    if args.rerun:
        setup_rerun("FoundGrasp_Demo", args.rerun_web_port, args.rerun_grpc_port)
        log_detection_to_rerun(detector, gg, 0)
        input("Press Enter to exit...")


def _set_rs_filter_option(filter_obj, option, value):
    try:
        filter_obj.set_option(option, value)
    except RuntimeError:
        pass


def make_realsense_postprocess_filters(rs):
    spatial = rs.spatial_filter()
    _set_rs_filter_option(spatial, rs.option.filter_magnitude, 2)
    _set_rs_filter_option(spatial, rs.option.filter_smooth_alpha, 0.5)
    _set_rs_filter_option(spatial, rs.option.filter_smooth_delta, 20)
    holes_fill_option = getattr(rs.option, "holes_fill", None)
    if holes_fill_option is not None:
        _set_rs_filter_option(spatial, holes_fill_option, 0)

    temporal = rs.temporal_filter()
    _set_rs_filter_option(temporal, rs.option.filter_smooth_alpha, 0.4)
    _set_rs_filter_option(temporal, rs.option.filter_smooth_delta, 20)

    return (
        rs.disparity_transform(True),
        spatial,
        temporal,
        rs.disparity_transform(False),
    )


def apply_realsense_postprocess(depth_frame, postprocess_filters):
    if postprocess_filters is None:
        return depth_frame

    for postprocess_filter in postprocess_filters:
        depth_frame = postprocess_filter.process(depth_frame)
    return depth_frame


def start_realsense(width, height, fps, use_postprocess=True):
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 is not installed. Install it with: pip install pyrealsense2")
        raise

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    postprocess_filters = make_realsense_postprocess_filters(rs) if use_postprocess else None
    return rs, pipeline, align, depth_scale, postprocess_filters


def get_realsense_frame(pipeline, align, depth_scale, postprocess_filters=None):
    frames = align.process(pipeline.wait_for_frames())
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None

    depth_frame = apply_realsense_postprocess(depth_frame, postprocess_filters)
    color = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    intrinsics = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return color, depth, intrinsics


def run_realsense_inference(detector, frame, inference_options):
    color, depth, intrinsics = frame
    t0 = time.time()
    gg = detector.detect(color, depth, intrinsics, **inference_options)
    return gg, time.time() - t0


def format_realsense_status(mode, frame_idx, gg, dt, start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    avg_fps = (frame_idx + 1) / elapsed
    return (
        f"rs {mode:<6} | frame {frame_idx:05d} | grasps {len(gg):03d} | "
        f"infer {dt * 1000.0:5.1f} ms | avg {avg_fps:4.1f} fps"
    )


def print_status_line(message):
    print(f"\r{message}\033[K", end="", flush=True)


def should_update_status(last_status_time, interval):
    return interval <= 0.0 or time.time() - last_status_time >= interval


def run_realsense_live(args, detector, pipeline, align, depth_scale, postprocess_filters, inference_options):
    print("Running RealSense live inference. Press Ctrl+C to stop.")
    frame_idx = 0
    start_time = time.time()
    last_status_time = 0.0

    try:
        while args.num_frames <= 0 or frame_idx < args.num_frames:
            frame = get_realsense_frame(pipeline, align, depth_scale, postprocess_filters)
            if frame is None:
                continue

            gg, dt = run_realsense_inference(detector, frame, inference_options)

            log_realsense_detection(args, detector, gg, frame_idx)

            is_last_frame = args.num_frames > 0 and frame_idx + 1 >= args.num_frames
            if is_last_frame or should_update_status(last_status_time, args.status_interval):
                print_status_line(format_realsense_status("live", frame_idx, gg, dt, start_time))
                last_status_time = time.time()

            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        print()


def run_realsense_manual(args, detector, pipeline, align, depth_scale, postprocess_filters, inference_options):
    print("Running RealSense manual inference. Press Enter to infer, or type q then Enter to stop.")
    frame_idx = 0
    start_time = time.time()

    try:
        while args.num_frames <= 0 or frame_idx < args.num_frames:
            command = input("> ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break

            frame = get_realsense_frame(pipeline, align, depth_scale, postprocess_filters)
            if frame is None:
                print("No RealSense frame received.")
                continue

            gg, dt = run_realsense_inference(detector, frame, inference_options)
            print(format_realsense_status("manual", frame_idx, gg, dt, start_time))

            log_realsense_detection(args, detector, gg, frame_idx)

            frame_idx += 1
    except KeyboardInterrupt:
        pass


def run_realsense_once(args, detector, pipeline, align, depth_scale, postprocess_filters, inference_options):
    print("Capturing one RealSense frame...")
    start_time = time.time()

    while True:
        frame = get_realsense_frame(pipeline, align, depth_scale, postprocess_filters)
        if frame is not None:
            break

    gg, dt = run_realsense_inference(detector, frame, inference_options)
    print(format_realsense_status("once", 0, gg, dt, start_time))

    log_realsense_detection(args, detector, gg, 0)
    wait_for_rerun_if_needed(args)


def run_realsense(args, detector):
    _, pipeline, align, depth_scale, postprocess_filters = start_realsense(
        args.rs_width,
        args.rs_height,
        args.rs_fps,
        use_postprocess=args.rs_postprocess,
    )
    inference_options = resolve_inference_options(args, detector)
    if args.rerun:
        setup_rerun("FoundGrasp_RealSense", args.rerun_web_port, args.rerun_grpc_port)

    print(f"RealSense depth postprocessing: {'on' if args.rs_postprocess else 'off'}")
    try:
        if args.rs_mode == "live":
            run_realsense_live(args, detector, pipeline, align, depth_scale, postprocess_filters, inference_options)
        elif args.rs_mode == "manual":
            run_realsense_manual(args, detector, pipeline, align, depth_scale, postprocess_filters, inference_options)
        elif args.rs_mode == "once":
            run_realsense_once(args, detector, pipeline, align, depth_scale, postprocess_filters, inference_options)
        else:
            raise ValueError(f"Unknown RealSense mode: {args.rs_mode}")
    finally:
        pipeline.stop()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="FoundGrasp demo")
    parser.add_argument("image", nargs="?", help="Path to RGB image. Omit when using --realsense.")
    parser.add_argument("--realsense", action="store_true", help="Run live RealSense RGB-D inference.")
    parser.add_argument("--config", default="configs/final/foundgrasp_base.yaml", help="Model config path.")
    parser.add_argument(
        "--checkpoint",
        default="logs/final/foundgrasp_base/checkpoints/final_model.pth",
        help="Model checkpoint path.",
    )
    parser.add_argument("--output", default="demo_output", help="Output directory for image mode.")
    parser.add_argument("--depth", default=None, help="Optional .npy depth map in meters or millimeters.")
    parser.add_argument("--depth-model", default="depth-anything/da3metric-large", help="Depth Anything 3 model.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--top-k", type=int, default=100, help="Maximum grasps to return.")
    parser.add_argument(
        "--no-nms",
        dest="nms",
        action="store_false",
        default=True,
        help="Disable final GraspGroup NMS; to_grasp_group voxel candidate reduction still applies.",
    )
    parser.add_argument(
        "--use-collision-detection",
        "--collision-detection",
        dest="use_collision_detection",
        action="store_true",
        default=True,
        help="Enable collision detection; enabled by default.",
    )
    parser.add_argument(
        "--no-collision-detection",
        "--skip-cd",
        dest="use_collision_detection",
        action="store_false",
        help="Disable collision detection.",
    )
    parser.add_argument(
        "--collision-voxel-size",
        type=float,
        default=None,
        help="Voxel size for collision detection; defaults to config inference.collision_voxel_size or 0.01.",
    )
    parser.add_argument("--rerun", action="store_true", help="Log results to rerun.")
    parser.add_argument("--rerun-web-port", type=int, default=9090)
    parser.add_argument("--rerun-grpc-port", type=int, default=9876)
    parser.add_argument("--rs-width", type=int, default=1280)
    parser.add_argument("--rs-height", type=int, default=720)
    parser.add_argument("--rs-fps", type=int, default=30)
    parser.add_argument(
        "--no-rs-postprocess",
        dest="rs_postprocess",
        action="store_false",
        default=True,
        help="Disable RealSense depth postprocessing; enabled by default.",
    )
    parser.add_argument(
        "--rs-mode",
        choices=("live", "manual", "once"),
        default="live",
        help="RealSense inference mode: live runs continuously, manual runs on Enter, once runs one frame.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=0.5,
        help="Seconds between live status-line refreshes; 0 updates every inference.",
    )
    parser.add_argument("--num-frames", type=int, default=0, help="RealSense inferences to process; 0 means forever.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not args.realsense and not args.image:
        raise SystemExit("Provide an image path or use --realsense.")

    detector = FoundGraspDetector(args.config, args.checkpoint, device=args.device)
    if args.realsense:
        run_realsense(args, detector)
    else:
        run_image(args, detector)


if __name__ == "__main__":
    main()
