import os
# Fix distutils/setuptools conflict with triton
os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'stdlib')

import json
import torch
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets.graspnet1b import GraspNet1BDataset, graspnet_collate_fn
from datasets.graspclutter6d import GraspClutter6DDataset
from model.found_grasp import FoundGrasp

import open3d as o3d
from utils.collision_detector import ModelFreeCollisionDetectorGPU
from utils.visualization import log_to_rerun, visualize_and_save_opencv
from utils.config import load_config_with_base, resolve_config_path, config_name_from_path
import cv2

from graspnetAPI import GraspNetEval
from graspclutter6dAPI import GraspClutter6DEval
import numpy as np


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _flatten_valid_xyz_points(xyz):
    """Convert an xyz map to a valid (N, 3) point cloud."""
    if xyz.ndim == 3 and xyz.shape[0] == 3:
        points = xyz.permute(1, 2, 0).reshape(-1, 3)
    elif xyz.ndim == 3 and xyz.shape[-1] == 3:
        points = xyz.reshape(-1, 3)
    else:
        points = xyz.reshape(-1, 3)
    valid_mask = torch.isfinite(points).all(dim=1) & (points[:, 2] > 0.1)
    return points[valid_mask]


def resolve_checkpoint_path(checkpoint_arg, base_log_dir):
    if checkpoint_arg:
        return checkpoint_arg

    checkpoint_dir = os.path.join(base_log_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"No checkpoint directory found at '{checkpoint_dir}'. "
            "Pass --checkpoint explicitly or train the config first."
        )

    latest_epoch = -1
    latest_path = None
    prefix = "checkpoint_epoch_"
    suffix = ".pth"

    for name in os.listdir(checkpoint_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        epoch_str = name[len(prefix):-len(suffix)]
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = os.path.join(checkpoint_dir, name)

    if latest_path is None:
        raise FileNotFoundError(
            f"No epoch checkpoints found under '{checkpoint_dir}'. "
            "Expected files like checkpoint_epoch_10.pth."
        )

    print(f"No --checkpoint provided. Using latest epoch checkpoint: {latest_path}")
    return latest_path


def evaluate(data, output, valid_mask):
    """
    Evaluation function to compute metrics.

    Args:
        data: dict containing ground tru
        
        th data
        output: dict containing model predictions
        valid_mask: (B, 1, H, W) binary mask for valid pixels

    Returns:
        dict: evaluation metrics
    """
    # TODO: Implement your evaluation metrics here
    # Example structure:
    metrics = {
        'graspness_mae': 0.0,
        'score_accuracy': 0.0,
        'approach_angle_error': 0.0,
        'gripper_angle_error': 0.0,
        'width_mae': 0.0,
        'depth_accuracy': 0.0,
    }

    return metrics


G1B_CAMERA_ALIASES = {'realsense-d435': 'realsense', 'azure-kinect': 'kinect'}


def inference(config,
         batch_size,
         num_workers,
         camera,
         split,
         checkpoint_path, log_dir, save_visualization=True, use_rerun=False,
         use_collision_detection=False,  ann_ids_path=None):
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Initialize rerun if requested
    if use_rerun:
        try:
            import rerun as rr
            import rerun.blueprint as rrb
            import urllib.parse
            grpc_port = 9876
            web_port = 9090
            blueprint = rrb.Blueprint(auto_views=True, auto_layout=True)
            rr.init("ViTGrasp_Evaluation", default_blueprint=blueprint)
            server_uri = rr.serve_grpc(grpc_port=grpc_port, server_memory_limit='1GB')
            rr.serve_web_viewer(open_browser=False, web_port=web_port)
            encoded_uri = urllib.parse.quote(server_uri, safe='')
            url = f"http://127.0.0.1:{web_port}/?url={encoded_uri}"
            print(f"\n{'='*60}")
            print(f"VSCode Ports 패널에서 {web_port}, {grpc_port} 두 포트 모두 forward 후 아래 URL 접속:")
            print(f"{url}")
            print(f"{'='*60}\n")
        except ImportError:
            print("Warning: rerun not installed. Disabling rerun visualization.")
            use_rerun = False


    # Setup dataset
    scene_ann_ids = None
    if split == 'test_mini_mixed':
        if ann_ids_path is None:
            raise ValueError("--ann_ids_path is required for split=test_mini_mixed. "
                             "Generate it first with GraspNetEval.generate_mini_mixed_ann_ids().")
        with open(ann_ids_path, 'r') as f:
            _raw = json.load(f)
        scene_ann_ids = {int(k): v for k, v in _raw.items()}

    if config.dataset.name == 'GraspNet1B':
        dataset = GraspNet1BDataset(
            root=config.dataset.root,
            camera=camera,
            split=split,
            image_size=tuple(config.dataset.image_size),
            backbone_name=config.model.backbone.name,
            depth_corruption=None,
            scene_ann_ids=scene_ann_ids,
        )
    elif config.dataset.name == 'GraspClutter6D':
        dataset = GraspClutter6DDataset(
            root=config.dataset.root,
            camera=camera,
            split=split,
            image_size=tuple(config.dataset.image_size),
            backbone_name=config.model.backbone.name,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.name}")

    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if save_visualization or use_rerun else False,
        num_workers=num_workers,
        collate_fn=graspnet_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Total batches: {len(dataset_loader)}")

    # Setup model
    model = FoundGrasp(
        config=config.model
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Assume it's a direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")

    model = model.cuda()
    model.eval()
    if save_visualization:
        vis_dir = os.path.join(log_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

    print(f"Total parameters (M): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}")

    # Evaluation loop
    all_metrics = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataset_loader, desc="Evaluating")):
            # Move data to GPU
            for key in data:
                if key in ['og_color', 'scene_id', 'im_id', 'gt_sparse', 'aug_params', 'crop_params']:
                    continue
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda()

            # Forward pass

            with torch.cuda.amp.autocast(enabled=True):
                predictions = model(data)
            predictions = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in predictions.items()}

            batch_size = data['pixel_values'].shape[0]

            for b in range(batch_size):
                scene_id = data['scene_id'][b]
                im_id = data['im_id'][b]
                output = {k: v[b] for k, v in predictions.items()} # Single item batch
                # Get inference config with defaults
                inf_cfg = config.get('inference', {})
                gg = model.to_grasp_group(
                    predictions=output,
                    xyz=data['xyz'][b],
                    graspness_threshold=inf_cfg.get('graspness_threshold', 0.01),
                    objectness_threshold=inf_cfg.get('objectness_threshold', 0.5),
                    graspness_exp=inf_cfg.get('graspness_exp', 0.1),
                    grasp_score_exp=inf_cfg.get('grasp_score_exp', 2.0),
                    total_score_threshold=inf_cfg.get('total_score_threshold', None),
                    approach_kappa_risk_weight=inf_cfg.get('approach_kappa_risk_weight', 0.5),
                    gripper_kappa_risk_weight=inf_cfg.get('gripper_kappa_risk_weight', 1.0),
                    width_multiplier=inf_cfg.get('width_multiplier', 1.2),
                    z_max=inf_cfg.get('z_max', 1.5),
                    empty_grasp_min_points=inf_cfg.get('empty_grasp_min_points', 0),
                    segm_erosion_k=inf_cfg.get('segm_erosion_k', 0),
                    max_grasps=inf_cfg.get('max_grasps', 2048),
                    voxel_size=inf_cfg.get('voxel_size', 0.004),
                )

                if use_collision_detection:
                    points_gpu = _flatten_valid_xyz_points(data['xyz'][b])
                    mfcdetector = ModelFreeCollisionDetectorGPU(points_gpu, voxel_size=0.01)
                    gg_array = torch.from_numpy(gg.grasp_group_array).cuda()
                    no_collision_mask = mfcdetector.detect(gg_array)
                    gg = gg[no_collision_mask]

                # Visualize and save if needed
                if use_rerun or save_visualization:
                    vis_sample = {}
                    for key in data:
                        if isinstance(data[key], torch.Tensor):
                            vis_sample[key] = data[key][b]  # Extract first sample from batch
                    for key in output:
                        vis_sample[f'pred_{key}'] = output[key]
                    vis_result = visualize_and_save_opencv(vis_sample, save_dir=None, point_size=9)

                # Visualize with rerun if enabled
                if use_rerun:
                    points = _flatten_valid_xyz_points(data['xyz'][b]).detach().cpu().numpy()
                    log_to_rerun(vis_sample, vis_result, points, gg.nms(translation_thresh=0.03, rotation_thresh=30.0 / 180.0 * np.pi
                                                                        ).sort_by_score()[:50], batch_idx)
                    input("Press Enter to continue...")  # 일시정지하여 사용자가 확인할 수 있도록 함
                if save_visualization:
                    cv2.imwrite(os.path.join(vis_dir, f'{camera}_{scene_id}_{im_id}_vis.jpg'), vis_result['all_data_grid'])

                pred_subdir = 'grasp_predictions_cd' if use_collision_detection else 'grasp_predictions'
                gg_log_dir = os.path.join(log_dir, pred_subdir, f'{scene_id}', camera)
                if not os.path.exists(gg_log_dir):
                    os.makedirs(gg_log_dir, exist_ok=True)
                if config.dataset.name == 'GraspClutter6D':
                    gg_save_path = os.path.join(gg_log_dir, '{:06d}.npy'.format(im_id))
                else:
                    gg_save_path = os.path.join(gg_log_dir, '{:04d}.npy'.format(im_id))
                gg.save_npy(gg_save_path)


    # Aggregate metrics across all batches
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)


    print("\nEvaluation completed!")

def eval(config, log_dir, camera, split, num_workers=4, ann_ids_path=None, use_collision_detection=False):
    pred_subdir = 'grasp_predictions_cd' if use_collision_detection else 'grasp_predictions'
    gg_log_dir = os.path.join(log_dir, pred_subdir)

    if config.dataset.name == 'GraspClutter6D':
        ge = GraspClutter6DEval(root=config.dataset.root, camera=camera, split='test')
        res, ap = ge.eval_all(gg_log_dir, proc=num_workers)
        return {'ap': float(ap[0]), 'ap_0.4': float(ap[1]), 'ap_0.8': float(ap[2])}

    # GraspNet1B: normalize camera alias (realsense-d435 → realsense, azure-kinect → kinect)
    camera = G1B_CAMERA_ALIASES.get(camera, camera)

    if split == 'test_mini_mixed':
        if ann_ids_path is None:
            raise ValueError("--ann_ids_path is required for split=test_mini_mixed.")
        ge = GraspNetEval(root=config.dataset.root, split='test', camera=camera)
        return ge.eval_mini_mixed(gg_log_dir, ann_ids_path=ann_ids_path, proc=num_workers)
    elif split == 'test_mini':
        print("Using mini split for quick evaluation.")
        ge = GraspNetEval(root=config.dataset.root, split='test', camera=camera)
        results = ge.eval_mini(gg_log_dir, proc=num_workers)
        print(results)
        return results
    elif split in ('test_mini_seen', 'test_mini_similar', 'test_mini_novel'):
        scene_info = {
            'test_mini_seen': list(range(100, 130)),
            'test_mini_similar': list(range(130, 160)),
            'test_mini_novel': list(range(160, 190)),
        }
        ge = GraspNetEval(root=config.dataset.root, split='test', camera=camera)
        scene_ids = scene_info[split]
        print(f"Using mini split for quick evaluation: {split}")
        res = np.array(ge.parallel_eval_scenes(scene_ids=scene_ids, dump_folder=gg_log_dir, proc=num_workers, mini=True))
        ap = np.mean(res)
        res_reshaped = np.array(res).reshape(-1, 6)
        res_mean = np.mean(res_reshaped, axis=0)
        result = {
            split: {
                'res': res_mean,
                'ap': ap,
            }
        }
        print(result)
        return result

    else:
        ge = GraspNetEval(root=config.dataset.root, split=split, camera=camera)
        if split == 'test':
            return ge.eval_all(gg_log_dir, proc=num_workers)
        elif split == 'test_seen':
            return ge.eval_seen(gg_log_dir, proc=num_workers)
        elif split == 'test_similar':
            return ge.eval_similar(gg_log_dir, proc=num_workers)
        elif split == 'test_novel':
            return ge.eval_novel(gg_log_dir, proc=num_workers)
        else:
            raise ValueError(f"Unknown split: {split}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base.yaml',
                        help='Config path or name (e.g., exp8/foo, exp8/foo.yaml, configs/exp8/foo.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file. If omitted, use the latest epoch checkpoint for this config.')
    parser.add_argument('--save-vis', action='store_true', default=False,
                        help='Save visualizations as images (default: True, recommended for headless servers)')
    parser.add_argument('--vis-dir', type=str, default='visualizations',
                        help='Directory to save visualizations (default: visualizations)')
    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Use rerun for interactive 3D visualization (works on headless servers)')
    parser.add_argument('--ann_ids_path', type=str, default='mini_mixed_ann_ids.json',
                        help='Path to JSON file with per-scene ann_ids for test_mini_mixed split. '
                             'Generate with GraspNetEval.generate_mini_mixed_ann_ids().')
    parser.add_argument('--skip-cd', dest='skip_cd', action='store_true', default=False,
                        help='Skip collision detection (overrides config)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers (default: 4)')
    parser.add_argument('--split', type=str, default='test_similar',
                        help='Dataset split to use (default: test_similar)')
    parser.add_argument('--camera', type=str, default='realsense',
                        help='Camera type to use (default: realsense)')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='Run inference n')
    parser.add_argument('--eval', action='store_true', default=True,
                        help='Run evaluation ')
    parser.add_argument('--eval_mode', type=str, default='3d',)
    # Inference hyperparameters (override config)
    parser.add_argument('--graspness_exp', type=float, default=None,
                        help='Graspness exponent (overrides config)')
    parser.add_argument('--grasp_score_exp', type=float, default=None,
                        help='Grasp score exponent (overrides config)')
    parser.add_argument('--graspness_threshold', type=float, default=None,
                        help='Graspness threshold (overrides config)')
    parser.add_argument('--objectness_threshold', type=float, default=None,
                        help='Objectness threshold (overrides config)')
    parser.add_argument('--total_score_threshold', type=float, default=None,
                        help='Final ranking score threshold for NMS (overrides config)')
    parser.add_argument('--approach_kappa_risk_weight', type=float, default=None,
                        help='Risk weight for approach expected angular error (overrides config)')
    parser.add_argument('--gripper_kappa_risk_weight', type=float, default=None,
                        help='Risk weight for gripper expected angular error (overrides config)')
    parser.add_argument('--z_max', type=float, default=None,
                        help='Maximum grasp-center depth in meters (overrides config)')
    parser.add_argument('--empty_grasp_min_points', type=int, default=None,
                        help='Minimum voxelized points inside the evaluator-style inner grasp region')
    parser.add_argument('--log-suffix', type=str, default=None,
                        help='Optional subdirectory under logs/<config_name>/ for this run.')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['GraspNet1B', 'GraspClutter6D'],
                        help='Dataset to evaluate on (overrides config.dataset.name)')
    args = parser.parse_args()

    # Load config
    config_path = resolve_config_path(args.config)
    config = load_config_with_base(config_path)
    if args.dataset is not None:
        config.dataset.name = args.dataset
    ds_cfg = config.datasets.get(config.dataset.name)
    if ds_cfg is None:
        raise ValueError(f"No entry for '{config.dataset.name}' in config.datasets. Add it to _base.yaml.")
    config.dataset.root = ds_cfg.root

    # Override inference config with command line arguments
    if 'inference' not in config:
        config.inference = {}
    if args.graspness_exp is not None:
        config.inference.graspness_exp = args.graspness_exp
    if args.grasp_score_exp is not None:
        config.inference.grasp_score_exp = args.grasp_score_exp
    if args.graspness_threshold is not None:
        config.inference.graspness_threshold = args.graspness_threshold
    if args.objectness_threshold is not None:
        config.inference.objectness_threshold = args.objectness_threshold
    if args.total_score_threshold is not None:
        config.inference.total_score_threshold = args.total_score_threshold
    if args.approach_kappa_risk_weight is not None:
        config.inference.approach_kappa_risk_weight = args.approach_kappa_risk_weight
    if args.gripper_kappa_risk_weight is not None:
        config.inference.gripper_kappa_risk_weight = args.gripper_kappa_risk_weight
    if args.z_max is not None:
        config.inference.z_max = args.z_max
    if args.empty_grasp_min_points is not None:
        config.inference.empty_grasp_min_points = args.empty_grasp_min_points
    print(OmegaConf.to_yaml(config))

    config_name = config_name_from_path(config_path)
    base_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', config_name)
    log_dir = base_log_dir
    if args.log_suffix:
        log_dir = os.path.join(log_dir, args.log_suffix)
    os.makedirs(log_dir, exist_ok=True)

    use_collision_detection = bool(config.inference.get('use_collision_detection', True)) and not args.skip_cd

    checkpoint_path = None
    if args.infer:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint, base_log_dir)

    if args.infer:
        inference(config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            camera=args.camera,
            split=args.split,
            checkpoint_path=checkpoint_path,
            save_visualization=args.save_vis,
            log_dir=log_dir,
            use_rerun=args.rerun,
            use_collision_detection=use_collision_detection,
            ann_ids_path=args.ann_ids_path)
    if args.eval:
        eval_results = eval(config, log_dir, camera=args.camera, split=args.split, num_workers=args.num_workers, ann_ids_path=args.ann_ids_path, use_collision_detection=use_collision_detection)
        if eval_results is not None:
            cd_tag = '_cd' if use_collision_detection else ''
            eval_path = os.path.join(log_dir, f'eval_results{cd_tag}.json')
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(_to_serializable(eval_results), f, indent=2)
            print(f"Saved evaluation results to: {eval_path}")
