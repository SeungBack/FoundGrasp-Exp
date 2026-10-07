import os
# Fix distutils/setuptools conflict with triton
os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'stdlib')

# comet_ml must be imported before torch
try:
    import comet_ml  # noqa: F401
except ImportError:
    pass

import json
import re
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from utils.logger import logger
import argparse
from omegaconf import OmegaConf
from datetime import datetime, timedelta
import numpy as np
import time
import random


from datasets.graspnet1b import GraspNet1BDataset, graspnet_collate_fn
from datasets.graspclutter6d import GraspClutter6DDataset
from utils.config import load_config_with_base, resolve_config_path, config_name_from_path
import matplotlib.pyplot as plt

from model.found_grasp import FoundGrasp
import torch.nn.functional as F

from utils.visualization import visualize_and_save_opencv
from model.loss import compute_all_losses, compute_all_metrics
from utils.collision_detector import ModelFreeCollisionDetectorGPU
from utils.optimizer import build_optimizer
from graspnetAPI import GraspNetEval
from graspclutter6dAPI import GraspClutter6DEval


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_dataloader_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def maybe_barrier():
    if is_distributed():
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_peft_log_stats(model):
    peft = getattr(unwrap_model(model), 'peft', None)
    if peft is None or not hasattr(peft, 'get_compat_stats'):
        return {}
    return peft.get_compat_stats()


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def setup_distributed(config):
    ddp_config = config.training.get('ddp', {})
    ddp_requested = bool(ddp_config.get('enabled', False))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    if world_size > 1 and not ddp_requested:
        raise ValueError(
            "Launched with multiple processes but `training.ddp.enabled` is false. "
            "Either run `python train.py ...` or enable DDP in the config and launch with `torchrun`."
        )

    if not ddp_requested:
        return torch.device('cuda'), False, 0, 1, 0

    if world_size <= 1:
        print(
            "Warning: `training.ddp.enabled=true` but WORLD_SIZE<=1. "
            "Falling back to single-GPU training. Use `torchrun --nproc_per_node=N ...` to enable DDP."
        )
        return torch.device('cuda'), False, 0, 1, 0

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    backend = ddp_config.get('backend', 'nccl')

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method='env://', timeout=timedelta(hours=5))

    return torch.device(f'cuda:{local_rank}'), True, rank, world_size, local_rank


def move_batch_to_device(batch, device, skip_keys):
    for key in batch:
        if key in skip_keys:
            continue
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)
    return batch


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


def reduce_scalar_dict(values, device, average=True):
    reduced = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.numel() != 1:
                raise ValueError(f"Expected scalar tensor for key '{key}', got shape {tuple(tensor.shape)}")
            tensor = tensor.to(device=device, dtype=torch.float64)
        else:
            tensor = torch.tensor(float(value), device=device, dtype=torch.float64)

        if is_distributed():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if average:
                tensor /= get_world_size()

        reduced[key] = tensor.item()

    return reduced


def load_scene_ann_ids(split, ann_ids_path):
    if split != 'test_mini_mixed':
        return None

    if ann_ids_path is None:
        raise ValueError(
            "`ann_ids_path` is required when split='test_mini_mixed'. "
            "Generate it first with GraspNetEval.generate_mini_mixed_ann_ids()."
        )

    with open(ann_ids_path, 'r') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def find_latest_epoch_checkpoint(checkpoint_dir):
    """Return the latest checkpoint_epoch_*.pth path, or None if absent."""
    if not os.path.isdir(checkpoint_dir):
        return None

    latest = None
    pattern = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")
    for name in os.listdir(checkpoint_dir):
        match = pattern.match(name)
        if match is None:
            continue
        epoch = int(match.group(1))
        path = os.path.join(checkpoint_dir, name)
        if latest is None or epoch > latest[0]:
            latest = (epoch, path)
    return latest[1] if latest is not None else None


def torch_load_checkpoint(path, device):
    """Load a local training checkpoint, keeping compatibility across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def infer_best_val_loss(checkpoint_dir, device):
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.isfile(best_path):
        return float('inf')
    try:
        best_ckpt = torch_load_checkpoint(best_path, device)
        return float(best_ckpt.get('best_val_loss', best_ckpt.get('val_loss', float('inf'))))
    except Exception:
        return float('inf')


def resume_from_latest_checkpoint(
    checkpoint_dir,
    base_model,
    optimizer,
    scheduler,
    scaler,
    train_generator,
    device,
    log,
):
    checkpoint_path = find_latest_epoch_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"--resume was requested, but no checkpoint_epoch_*.pth was found in {checkpoint_dir}"
        )

    checkpoint = torch_load_checkpoint(checkpoint_path, device)
    if 'model_state_dict' not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a training checkpoint "
            "(missing 'model_state_dict')."
        )

    base_model.load_state_dict(checkpoint['model_state_dict'])

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        log(f"Warning: {checkpoint_path} has no optimizer_state_dict; optimizer starts fresh.")

    if scheduler is not None:
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            log(f"Warning: {checkpoint_path} has no scheduler_state_dict; scheduler starts fresh.")

    if scaler is not None:
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            log(f"Warning: {checkpoint_path} has no scaler_state_dict; GradScaler starts fresh.")

    set_rng_state(checkpoint.get('rng_state', None))
    if train_generator is not None and 'dataloader_generator_state' in checkpoint:
        train_generator.set_state(checkpoint['dataloader_generator_state'])

    start_epoch = int(checkpoint.get('epoch', 0))
    global_step = int(checkpoint.get('global_step', 0))
    best_val_loss = float(
        checkpoint.get(
            'best_val_loss',
            checkpoint.get('val_loss', infer_best_val_loss(checkpoint_dir, device)),
        )
    )

    log(
        f"Resumed from {checkpoint_path}: "
        f"completed_epoch={start_epoch}, global_step={global_step}, "
        f"best_val_loss={best_val_loss if np.isfinite(best_val_loss) else 'inf'}"
    )
    return start_epoch, global_step, best_val_loss, checkpoint_path


def build_dataloader_kwargs(config, num_workers=None):
    if num_workers is None:
        num_workers = config.training.num_workers
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': config.training.get('pin_memory', True),
        'collate_fn': graspnet_collate_fn,
        'worker_init_fn': worker_init_fn,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = config.training.get('persistent_workers', True)
        kwargs['prefetch_factor'] = config.training.get('prefetch_factor', 4)
    return kwargs


def maybe_apply_train_subset(dataset, dataset_cfg, default_seed):
    """Optionally subsample the training set for low-data scaling experiments."""
    train_fraction = float(dataset_cfg.get('train_fraction', 1.0))
    max_samples = dataset_cfg.get('train_max_samples', None)

    if train_fraction <= 0.0 or train_fraction > 1.0:
        raise ValueError(f"`dataset.train_fraction` must be in (0, 1], got {train_fraction}")

    total_samples = len(dataset)
    requested_samples = total_samples
    if train_fraction < 1.0:
        requested_samples = min(
            requested_samples,
            max(1, int(round(total_samples * train_fraction))),
        )
    if max_samples is not None:
        requested_samples = min(requested_samples, int(max_samples))

    if requested_samples >= total_samples:
        return dataset, total_samples, total_samples

    subset_seed = int(dataset_cfg.get('train_subset_seed', default_seed))
    rng = np.random.default_rng(subset_seed)
    selected_indices = np.sort(rng.choice(total_samples, size=requested_samples, replace=False))
    subset = Subset(dataset, selected_indices.tolist())
    return subset, requested_samples, total_samples


def _run_gc6d_final_test(base_model, config, log_dir, device, global_step, batch_size, num_workers, use_amp):
    """Run inference + GraspClutter6DEval on the GraspClutter6D test split."""
    GC6D_CAMERA = 'realsense-d435'

    gc6d_root = config.datasets.GraspClutter6D.root
    dataset = GraspClutter6DDataset(
        root=gc6d_root,
        camera=GC6D_CAMERA,
        split='test',
        image_size=tuple(config.dataset.image_size),
        backbone_name=config.model.backbone.name,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=graspnet_collate_fn,
        pin_memory=config.training.get('pin_memory', True),
        worker_init_fn=worker_init_fn,
        **(({'persistent_workers': True, 'prefetch_factor': 4}) if num_workers > 0 else {}),
    )

    model = unwrap_model(base_model)
    model.eval()

    inf_cfg = config.get('inference', {})
    use_cd = bool(inf_cfg.get('use_collision_detection', False))
    pred_subdir = 'grasp_predictions_cd' if use_cd else 'grasp_predictions'
    grasp_pred_dir = os.path.join(log_dir, 'final_test_gc6d', pred_subdir)

    print(f"\n{'='*80}")
    print(f"Running Final Test Evaluation on GraspClutter6D (camera={GC6D_CAMERA})...")
    print(f"  Total samples: {len(dataset)}")
    print(f"{'='*80}")

    from tqdm import tqdm
    with torch.no_grad():
        for data in tqdm(loader, desc="GC6D Inference"):
            move_batch_to_device(data, device, {'og_color', 'scene_id', 'im_id'})

            if use_amp:
                with torch.amp.autocast('cuda', enabled=True):
                    output = model(data)
                output = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
            else:
                output = model(data)

            for b in range(data['pixel_values'].shape[0]):
                scene_id = data['scene_id'][b]   # already '000123'
                im_id = int(data['im_id'][b])

                sample_output = {k: v[b] for k, v in output.items()}
                gg = model.to_grasp_group(
                    predictions=sample_output,
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
                    empty_grasp_min_points=inf_cfg.get('empty_grasp_min_points', 10),
                    segm_erosion_k=inf_cfg.get('segm_erosion_k', 0),
                    max_grasps=inf_cfg.get('max_grasps', 2048),
                    voxel_size=inf_cfg.get('voxel_size', 0.004),
                )

                if use_cd and len(gg) > 0:
                    points_gpu = _flatten_valid_xyz_points(data['xyz'][b])
                    mfcdetector = ModelFreeCollisionDetectorGPU(
                        points_gpu, voxel_size=0.01, device=points_gpu.device
                    )
                    gg_array = torch.from_numpy(gg.grasp_group_array).to(points_gpu.device)
                    gg = gg[mfcdetector.detect(gg_array)]

                gg_dir = os.path.join(grasp_pred_dir, scene_id, GC6D_CAMERA)
                os.makedirs(gg_dir, exist_ok=True)
                gg.save_npy(os.path.join(gg_dir, f'{im_id:06d}.npy'))
                del gg, sample_output

            del output
            for key in list(data.keys()):
                if isinstance(data[key], torch.Tensor):
                    del data[key]

    # GraspClutter6D evaluation
    print(f"\n  Running GraspClutter6DEval...")
    ge = GraspClutter6DEval(root=gc6d_root, camera=GC6D_CAMERA, split='test')
    _, ap = ge.eval_all(grasp_pred_dir, proc=num_workers)

    gc6d_metrics = {
        'test/gc6d/ap':     float(ap[0]) * 100,
        'test/gc6d/ap@0.4': float(ap[1]) * 100,
        'test/gc6d/ap@0.8': float(ap[2]) * 100,
    }
    logger.log({**gc6d_metrics, 'epoch': config.training.num_epochs, 'global_step': global_step}, step=global_step)
    print(f"    GC6D AP={gc6d_metrics['test/gc6d/ap']:.2f}  "
          f"AP@0.4={gc6d_metrics['test/gc6d/ap@0.4']:.2f}  "
          f"AP@0.8={gc6d_metrics['test/gc6d/ap@0.8']:.2f}")
    return gc6d_metrics


def evaluate(
    model,
    val_loader,
    config,
    epoch,
    device,
    log_dir=None,
    global_step=0,
    eval_graspnet=False,
    collision_thresh=0.0,
    stage_name='Validation',
    log_prefix='val',
    visualization_subdir='validation',
    prediction_subdir='grasp_predictions',
    graspnet_log_prefix='graspnet',
    save_visualizations=True,
    use_amp=False,
    ann_ids_path=None,
):
    """
    Evaluate the model on validation set.
    Returns average loss and metrics.

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        config: Configuration object
        epoch: Current epoch number
        log_dir: Directory to save visualizations (optional)
        global_step: Current global step for naming (optional)
        eval_graspnet: Whether to save grasp groups (optional)
        collision_thresh: Collision threshold for grasp filtering (optional)
    """
    model = unwrap_model(model)
    model.eval()
    if config.model.backbone.freeze:
        model.backbone.eval()

    val_loss = 0.0
    val_losses = {
        'segm': 0.0,
        'graspness': 0.0,
        'score': 0.0,
        'approach': 0.0,
        'gripper': 0.0,
        'orientation': 0.0,
        'width': 0.0,
        'depth': 0.0
    }
    val_metrics = {
        'graspness_mae': 0.0,
        'score_mae': 0.0,
        'approach_angle_error': 0.0,
        'gripper_angle_error': 0.0,
        'width_mae': 0.0,
        'depth_mae': 0.0
    }

    num_batches_processed = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            move_batch_to_device(data, device, {'og_color', 'scene_id', 'im_id', 'pad_hw'})

            # Forward pass
            if use_amp:
                with torch.amp.autocast('cuda', enabled=True):
                    output = model(data)
                output = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
            else:
                output = model(data)

            # Compute all losses
            losses, valid_mask = compute_all_losses(output, data, config.training.loss_functions)

            # Accumulate losses
            val_loss += losses['total'].item()
            for key in ['segm', 'graspness', 'score', 'approach', 'gripper', 'orientation', 'width', 'depth']:
                if key in losses:
                    val_losses[key] += losses[key].item()
           
            # Compute all metrics
            metrics = compute_all_metrics(
                output, data, valid_mask,
                pred_width_scaled=losses['_pred_width_scaled'],
                pred_depth_scaled=losses['_pred_depth_scaled']
            )

            # Accumulate metrics
            val_metrics['graspness_mae'] += metrics['graspness_mae'].item()
            val_metrics['score_mae'] += metrics['score_mae'].item()
            val_metrics['approach_angle_error'] += metrics['approach_angle_error'].item()
            val_metrics['gripper_angle_error'] += metrics['gripper_angle_error'].item()
            val_metrics['width_mae'] += metrics['width_mae'].item()
            val_metrics['depth_mae'] += metrics['depth_mae'].item()

            # Save grasp groups if requested
            if eval_graspnet and log_dir is not None:
                batch_size = output['objectness'].shape[0]
                for b in range(batch_size):
                    scene_id = int(data['scene_id'][b].split('_')[-1])
                    im_id = int(data['im_id'][b])

                    # Extract single sample predictions
                    sample_output = {k: v[b] for k, v in output.items()}

                    # Generate grasp group (use inference config with defaults)
                    inf_cfg = config.get('inference', {})
                    gg = model.to_grasp_group(
                        predictions=sample_output,
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
                        empty_grasp_min_points=inf_cfg.get('empty_grasp_min_points', 10),
                        segm_erosion_k=inf_cfg.get('segm_erosion_k', 0),
                        max_grasps=inf_cfg.get('max_grasps', 2048),
                        voxel_size=inf_cfg.get('voxel_size', 0.004),
                    )
                    # Apply collision detection if enabled (same hard-filter path as test.py)
                    use_collision_detection = bool(inf_cfg.get('use_collision_detection', collision_thresh > 0))
                    if use_collision_detection and len(gg) > 0:
                        points_gpu = _flatten_valid_xyz_points(data['xyz'][b])
                        mfcdetector = ModelFreeCollisionDetectorGPU(
                            points_gpu, voxel_size=0.01, device=points_gpu.device
                        )
                        gg_array = torch.from_numpy(gg.grasp_group_array).to(points_gpu.device)
                        no_collision_mask = mfcdetector.detect(gg_array)
                        gg = gg[no_collision_mask]
                    # Save grasp group
                    camera = config.dataset.camera
                    gg_log_dir = os.path.join(log_dir, prediction_subdir, f'scene_{scene_id:04d}', camera)
                    os.makedirs(gg_log_dir, exist_ok=True)
                    gg_save_path = os.path.join(gg_log_dir, f'{im_id:04d}.npy')
                    gg.save_npy(gg_save_path)
                    
                    # Clear grasp group to free memory
                    del gg, sample_output

            # Visualize every 50 images
            if save_visualizations and log_dir is not None and batch_idx % 50 == 0:
                # Create visualization dict from batch data (idx=0)
                vis_sample = {}
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        vis_sample[key] = data[key][0]  # Extract first sample from batch
                    else:
                        vis_sample[key] = data[key]

                # Add predictions to visualization
                for key in output:
                    if isinstance(output[key], torch.Tensor):
                        vis_sample[f'pred_{key}'] = output[key][0]  # Extract first sample from batch

                # Save using OpenCV-based visualization
                viz_dir = os.path.join(log_dir, 'visualizations', visualization_subdir)
                visualize_and_save_opencv(vis_sample, save_dir=viz_dir, prefix=f'step_{global_step}_batch_{batch_idx}_')
                del vis_sample

            num_batches_processed += 1
            
            # Clear batch data to prevent memory accumulation
            del output, losses, metrics
            for key in list(data.keys()):
                if isinstance(data[key], torch.Tensor):
                    del data[key]

    # Compute averages
    num_batches = num_batches_processed if num_batches_processed > 0 else 1
    avg_val_loss = val_loss / num_batches

    # Log metrics
    log_dict = {
        f'{log_prefix}/total_loss': avg_val_loss,
        f'{log_prefix}/segm_loss': val_losses['segm']/num_batches,
        f'{log_prefix}/graspness_loss': val_losses['graspness']/num_batches,
        f'{log_prefix}/score_loss': val_losses['score']/num_batches,
        f'{log_prefix}/width_loss': val_losses['width']/num_batches,
        f'{log_prefix}/depth_loss': val_losses['depth']/num_batches,
        f'{log_prefix}/graspness_mae': val_metrics['graspness_mae']/num_batches,
        f'{log_prefix}/score_mae': val_metrics['score_mae']/num_batches,
        f'{log_prefix}/approach_angle_error': val_metrics['approach_angle_error']/num_batches,
        f'{log_prefix}/gripper_angle_error': val_metrics['gripper_angle_error']/num_batches,
        f'{log_prefix}/width_mae': val_metrics['width_mae']/num_batches,
        f'{log_prefix}/depth_mae': val_metrics['depth_mae']/num_batches,
        'epoch': epoch,
        'global_step': global_step
    }
    if val_losses['orientation'] != 0:
        log_dict[f'{log_prefix}/orientation_loss'] = val_losses['orientation']/num_batches
    else:
        log_dict[f'{log_prefix}/approach_loss'] = val_losses['approach']/num_batches
        log_dict[f'{log_prefix}/gripper_loss'] = val_losses['gripper']/num_batches
    logger.log(log_dict, step=global_step)

    # Print validation summary
    print(f"\n  {stage_name} Results:")
    print(f"    Average Loss: {avg_val_loss:.4f}")
    print(f"    Losses:")
    print(f"      Segmentation: {val_losses['segm']/num_batches:.4f}")
    print(f"      Graspness: {val_losses['graspness']/num_batches:.4f}")
    print(f"      Score: {val_losses['score']/num_batches:.4f}")
    if val_losses['orientation'] != 0:
        print(f"      Joint Orientation: {val_losses['orientation']/num_batches:.4f}")
    else:
        print(f"      Approach: {val_losses['approach']/num_batches:.4f}")
        print(f"      Gripper: {val_losses['gripper']/num_batches:.4f}")
    print(f"      Width: {val_losses['width']/num_batches:.4f}")
    print(f"      Depth: {val_losses['depth']/num_batches:.4f}")
    print(f"    Metrics:")
    print(f"      Graspness MAE: {val_metrics['graspness_mae']/num_batches:.4f}")
    print(f"      Score MAE: {val_metrics['score_mae']/num_batches:.4f}")
    print(f"      Approach Angle Error: {val_metrics['approach_angle_error']/num_batches:.2f}°")
    print(f"      Gripper Angle Error: {val_metrics['gripper_angle_error']/num_batches:.2f}°")
    print(f"      Width MAE: {val_metrics['width_mae']/num_batches:.4f}m")
    print(f"      Depth MAE: {val_metrics['depth_mae']/num_batches:.4f}m")

    # Run GraspNetEval if grasps were saved
    graspnet_metrics = {}
    if eval_graspnet and log_dir is not None:
        print(f"\n  Running {stage_name} GraspNet Evaluation...")
        try:
            # Determine split from val_loader dataset
            val_split = val_loader.dataset.split
            camera = config.dataset.camera

            # Create grasp predictions directory path
            grasp_pred_dir = os.path.join(log_dir, prediction_subdir)

            if val_split in ('test_mini', 'test_mini_mixed'):
                print(f"    Using mini split for quick evaluation: {val_split}")
                ge = GraspNetEval(root=config.dataset.root, split='test', camera=camera)
                if val_split == 'test_mini_mixed':
                    if ann_ids_path is None:
                        raise ValueError("ann_ids_path is required for split='test_mini_mixed'.")
                    results_dict = ge.eval_mini_mixed(grasp_pred_dir, ann_ids_path=ann_ids_path, proc=8)
                else:
                    results_dict = ge.eval_mini(grasp_pred_dir, proc=8)

                # results_dict contains results for each split: 'test_seen', 'test_similar', 'test_novel'
                # Each split contains 'res' (AP at different thresholds) and 'ap' (mean AP)
                graspnet_metrics = {}
                for split_name, split_results in results_dict.items():
                    res_mean = split_results['res']
                    split_prefix = split_name.replace('test_', '')
                    graspnet_metrics[f'{graspnet_log_prefix}/{split_prefix}_ap'] = np.mean(res_mean) * 100
                    graspnet_metrics[f'{graspnet_log_prefix}/{split_prefix}_ap@0.4'] = res_mean[1] * 100
                    graspnet_metrics[f'{graspnet_log_prefix}/{split_prefix}_ap@0.8'] = res_mean[3] * 100
            else:
                ge = GraspNetEval(root=config.dataset.root, split=val_split, camera=camera)
                if val_split == 'test':
                    res_seen, _ = ge.eval_seen(grasp_pred_dir, proc=8)
                    res_similar, _ = ge.eval_similar(grasp_pred_dir, proc=8)
                    res_novel, _ = ge.eval_novel(grasp_pred_dir, proc=8)
                    all_ap = np.mean([np.mean(res_seen), np.mean(res_similar), np.mean(res_novel)])
                    graspnet_metrics = {
                        f'{graspnet_log_prefix}/all_ap': all_ap * 100,
                        f'{graspnet_log_prefix}/seen_ap': np.mean(res_seen) * 100,
                        f'{graspnet_log_prefix}/seen_ap@0.4': res_seen[1] * 100,
                        f'{graspnet_log_prefix}/seen_ap@0.8': res_seen[3] * 100,
                        f'{graspnet_log_prefix}/similar_ap': np.mean(res_similar) * 100,
                        f'{graspnet_log_prefix}/similar_ap@0.4': res_similar[1] * 100,
                        f'{graspnet_log_prefix}/similar_ap@0.8': res_similar[3] * 100,
                        f'{graspnet_log_prefix}/novel_ap': np.mean(res_novel) * 100,
                        f'{graspnet_log_prefix}/novel_ap@0.4': res_novel[1] * 100,
                        f'{graspnet_log_prefix}/novel_ap@0.8': res_novel[3] * 100,
                    }
                elif val_split == 'test_seen':
                    res, ap = ge.eval_seen(grasp_pred_dir, proc=8)
                    # res is shape (6,) containing AP at different thresholds
                    graspnet_metrics = {
                        f'{graspnet_log_prefix}/seen_ap': np.mean(res) * 100,
                        f'{graspnet_log_prefix}/seen_ap@0.4': res[1] * 100,
                        f'{graspnet_log_prefix}/seen_ap@0.8': res[3] * 100,
                    }
                elif val_split == 'test_similar':
                    res, ap = ge.eval_similar(grasp_pred_dir, proc=8)
                    # res is shape (6,) containing AP at different thresholds
                    graspnet_metrics = {
                        f'{graspnet_log_prefix}/similar_ap': np.mean(res) * 100,
                        f'{graspnet_log_prefix}/similar_ap@0.4': res[1] * 100,
                        f'{graspnet_log_prefix}/similar_ap@0.8': res[3] * 100,
                    }
                elif val_split == 'test_novel':
                    res, ap = ge.eval_novel(grasp_pred_dir, proc=8)
                    # res is shape (6,) containing AP at different thresholds
                    graspnet_metrics = {
                        f'{graspnet_log_prefix}/novel_ap': np.mean(res) * 100,
                        f'{graspnet_log_prefix}/novel_ap@0.4': res[1] * 100,
                        f'{graspnet_log_prefix}/novel_ap@0.8': res[3] * 100,
                    }
                else:
                    print(f"    Warning: Unknown split '{val_split}' for GraspNet evaluation")

            # Log GraspNet evaluation results
            if graspnet_metrics:
                logger.log({**graspnet_metrics, 'epoch': epoch, 'global_step': global_step}, step=global_step)
                print(f"    GraspNet evaluation completed and logged!")
            else:
                print(f"    GraspNet evaluation completed!")

        except Exception as e:
            print(f"    Warning: GraspNet evaluation failed: {e}")

    return avg_val_loss, val_metrics, graspnet_metrics


def main(config, config_name='base', resume=False):
    device, use_ddp, rank, world_size, local_rank = setup_distributed(config)

    def log(*args, **kwargs):
        if is_main_process():
            print(*args, **kwargs)

    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    log(f"Random seed set to: {seed}")
    if use_ddp:
        log(f"DDP enabled: world_size={world_size}, local_batch_size={config.training.batch_size}, global_batch_size={config.training.batch_size * world_size}")

    # Setup logs directory
    log_dir = os.path.join('logs', f"{config_name}")
    os.makedirs(log_dir, exist_ok=True)
    log(f"Logs will be saved to: {log_dir}")

    # Setup checkpoint directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    log(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Initialize experiment logger (comet or wandb)
    log_config = config.get('wandb', {})
    backend = log_config.get('backend', 'comet')
    mode = 'disabled' if config_name.startswith('_debug') else log_config.get('mode', 'online')

    if is_main_process():
        logger.init(
            backend=backend,
            project=log_config.get('project', 'FoundGrasp'),
            name=config_name,
            config_dict=OmegaConf.to_container(config, resolve=True),
            log_dir=log_dir,
            mode=mode,
            entity=log_config.get('entity', None),
        )
        if mode == 'disabled':
            log("Experiment logging disabled (debug mode)")
        else:
            log(f"Experiment logging initialized: backend={backend}, project=FoundGrasp, run={config_name}")

    if config.dataset.name == 'GraspNet1B':
        dataset_input_type = config.dataset.get('input_type', config.model.input_type)
        train_dataset = GraspNet1BDataset(
            root=config.dataset.root,
            camera=config.dataset.camera,
            split=config.dataset.split,
            image_size=tuple(config.dataset.image_size),
            backbone_name=config.model.backbone.name,
            augmentation=config.training.augmentation,
            input_type=dataset_input_type,
            depth_corruption=config.training.get('depth_corruption', None),
            color_aug_strength=config.training.get('color_aug_strength', 1.0),
        )
        train_dataset, train_subset_size, train_full_size = maybe_apply_train_subset(
            train_dataset,
            config.dataset,
            seed,
        )

        # Create validation dataset (without augmentation)
        validation_cfg = config.training.get('validation', {})
        val_split = validation_cfg.get('split', 'test_mini')
        val_ann_ids_path = validation_cfg.get('ann_ids_path', None)
        val_scene_ann_ids = load_scene_ann_ids(val_split, val_ann_ids_path)
        val_dataset = GraspNet1BDataset(
            root=config.dataset.root,
            camera=config.dataset.camera,
            split=val_split,
            image_size=tuple(config.dataset.image_size),
            backbone_name=config.model.backbone.name,
            augmentation=None,
            input_type=dataset_input_type,
            depth_corruption=None,
            scene_ann_ids=val_scene_ann_ids,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.name}")

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )

    loader_kwargs = build_dataloader_kwargs(config)
    validation_cfg = config.training.get('validation', {})
    val_batch_size = validation_cfg.get('batch_size', config.training.batch_size)
    val_num_workers = validation_cfg.get('num_workers', None)
    train_generator = make_dataloader_generator(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=train_generator,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        **build_dataloader_kwargs(config, num_workers=val_num_workers)
    )

    base_model = FoundGrasp(
        config=config.model
    )
    base_model = base_model.to(device)
    base_model.train()

    # unfreeze prompt parameters if using prompt tuning
    if config.model.prompt.mode in ['vpt', 'vipt', 'ours'] and base_model.prompt_generator is not None:
        for name, param in base_model.prompt_generator.named_parameters():
            log(f"Unfreezing prompt parameter: {name}")
            param.requires_grad = True

    model = base_model
    if use_ddp:
        ddp_config = config.training.get('ddp', {})
        model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(ddp_config.get('find_unused_parameters', False)),
            broadcast_buffers=bool(ddp_config.get('broadcast_buffers', False)),
        )

    # Setup optimizer (with optional layer-wise learning rate decay)
    optimizer = build_optimizer(base_model, config)

    # Setup AMP GradScaler if enabled
    use_amp = config.training.get('use_amp', False)
    amp_dtype_str = config.training.get('amp_dtype', 'bfloat16') # config에 없으면 기본 bf16 시도
    
    # 하드웨어 지원 확인
    if use_amp and amp_dtype_str == 'bfloat16':
        if torch.cuda.is_bf16_supported():
            log("Using BFloat16 AMP (No GradScaler needed)")
            amp_dtype = torch.bfloat16
            use_scaler = False  # BF16은 Scaler 불필요
        else:
            log("Warning: BFloat16 not supported on this GPU. Falling back to Float16.")
            amp_dtype = torch.float16
            use_scaler = True
    else:
        # 기존 FP16 로직
        amp_dtype = torch.float16
        use_scaler = use_amp

    # Scaler 초기화 (FP16일 때만 생성)
    scaler = GradScaler() if use_scaler else None
    
    if use_amp:
        log(f"AMP Enabled. Dtype: {amp_dtype}, Scaler: {'On' if use_scaler else 'Off'}")
    else:
        log("Using full precision (FP32) training")

    # Setup gradient clipping
    grad_clip = config.training.get('grad_clip', None)
    if grad_clip is not None:
        log(f"Using gradient clipping with max_norm: {grad_clip}")
    else:
        log("Gradient clipping disabled")

    # Setup learning rate warmup
    warmup_config = config.training.get('warmup', {})
    warmup_epochs = warmup_config.get('epochs', 0)
    warmup_start_factor = warmup_config.get('start_factor', 0.01)

    if warmup_epochs > 0:
        log(f"Using LR warmup: {warmup_epochs} epochs, start_factor={warmup_start_factor}")

    # Setup learning rate scheduler
    scheduler_type = config.training.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs - warmup_epochs,
            eta_min=max(config.training.learning_rate * 0.01, 1e-6)
        )
        log("Using Cosine Annealing LR scheduler")
    elif scheduler_type == 'linear':
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=config.training.num_epochs - warmup_epochs
        )
        log("Using Linear LR scheduler")
    elif scheduler_type == 'none':
        main_scheduler = None
        log("No LR scheduler")
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose from 'cosine', 'linear', or 'none'.")

    # Combine warmup with main scheduler
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        if main_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = warmup_scheduler
    else:
        scheduler = main_scheduler

    log(f"Total parameters: {sum(p.numel() for p in base_model.parameters())/1e6:.2f}M")
    log(f"Trainable parameters: {sum(p.numel() for p in base_model.parameters() if p.requires_grad)/1e6:.2f}M")
    # print trainable parameters for each big module
    for name, module in base_model.named_children():
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in module.parameters())
        log(f"  {name}: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M trainable")
    log(f"Train dataset size: {len(train_dataset)}")
    if len(train_dataset) != train_full_size:
        log(
            f"  Using deterministic train subset: {train_subset_size}/{train_full_size} "
            f"({train_subset_size / train_full_size:.1%})"
        )
    log(f"Validation dataset size: {len(val_dataset)}")

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    if resume:
        start_epoch, global_step, best_val_loss, _ = resume_from_latest_checkpoint(
            checkpoint_dir=checkpoint_dir,
            base_model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_generator=train_generator,
            device=device,
            log=log,
        )
        if start_epoch >= config.training.num_epochs:
            log(
                f"Warning: resumed checkpoint already completed {start_epoch} epochs, "
                f"but config.training.num_epochs={config.training.num_epochs}. "
                "No additional training epochs remain."
            )

    if resume:
        # Do not run the optimizer-state warmup after loading optimizer state:
        # that would add an extra update not present in the original run.
        log("Skipping CUDA warmup optimizer step because training is being resumed.")
        if train_sampler is not None:
            train_sampler.set_epoch(start_epoch)
        train_iter = iter(train_loader)
    else:
        # Pre-create epoch 0 iterator first so warmup and training share the same generator state.
        if train_sampler is not None:
            train_sampler.set_epoch(0)
        train_iter = iter(train_loader)

        # Pre-warm CUDA kernels + optimizer state with one real batch before timing starts.
        log("Pre-warming CUDA kernels...")
        warmup_batch = next(train_iter)
        move_batch_to_device(warmup_batch, device, {'og_color', 'scene_id', 'im_id', 'pad_hw'})
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            _warmup_out = model(warmup_batch)
        if use_amp:
            _warmup_out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in _warmup_out.items()}
        _warmup_loss, _ = compute_all_losses(_warmup_out, warmup_batch, config.training.loss_functions)
        _warmup_loss['total'].backward()
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del warmup_batch, _warmup_out, _warmup_loss
        torch.cuda.synchronize()
        log("CUDA warmup done.")

        # Reset epoch 0 iterator so the warmup batch is re-seen during training.
        train_iter = iter(train_loader)

    # Time tracking
    training_start_time = time.time()
    epoch_times = []

    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start_time = time.time()
        # train_iter already points to this epoch (created at end of previous epoch,
        # or above for epoch 0). No need to recreate here.

        log(f"\n{'='*80}")
        log(f"Epoch {epoch+1}/{config.training.num_epochs}")
        log(f"{'='*80}")
        epoch_loss = 0.0
        epoch_losses = {
            'segm': 0.0,
            'graspness': 0.0,
            'score': 0.0,
            'approach': 0.0,
            'gripper': 0.0,
            'orientation': 0.0,
            'width': 0.0,
            'depth': 0.0
        }
        epoch_metrics = {
            'graspness_mae': 0.0,
            'score_mae': 0.0,
            'approach_angle_error': 0.0,
            'gripper_angle_error': 0.0,
            'width_mae': 0.0,
            'depth_mae': 0.0
        }

        model.train()
        if config.model.backbone.freeze:
            base_model.backbone.eval()

        num_batches_processed = 0
        for batch_idx, data in enumerate(train_iter):
            move_batch_to_device(data, device, {'og_color', 'scene_id', 'im_id', 'pad_hw'})

            # Forward pass (AMP)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                output = model(data)

            # Label assignment + loss computation in FP32
            # (prevents AMP-induced loss spikes from argmax-based dynamic label assignment)
            if use_amp:
                output = {k: (v.float() if torch.is_tensor(v) else v) for k, v in output.items()}
            losses, valid_mask = compute_all_losses(output, data, config.training.loss_functions)

            # Backward pass + optimizer step
            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(losses['total']).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses['total'].backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)
                optimizer.step()
            # Accumulate losses
            epoch_loss += losses['total'].item()
            for key in ['segm', 'graspness', 'score', 'approach', 'gripper', 'orientation', 'width', 'depth']:
                if key in losses:
                    epoch_losses[key] += losses[key].item()


            # Accumulate ranking losses if present
            for key in losses:
                if key.startswith('ranking_'):
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += losses[key].item()

            # Metrics
            with torch.no_grad():
                metrics = compute_all_metrics(
                    output, data, valid_mask,
                    pred_width_scaled=losses['_pred_width_scaled'],
                    pred_depth_scaled=losses['_pred_depth_scaled']
                )
                epoch_metrics['graspness_mae'] += metrics['graspness_mae'].item()
                epoch_metrics['score_mae'] += metrics['score_mae'].item()
                epoch_metrics['approach_angle_error'] += metrics['approach_angle_error'].item()
                epoch_metrics['gripper_angle_error'] += metrics['gripper_angle_error'].item()
                epoch_metrics['width_mae'] += metrics['width_mae'].item()
                epoch_metrics['depth_mae'] += metrics['depth_mae'].item()

            # Wandb logging
            if batch_idx % config.training.print_freq == 0:
                batch_log_values = {
                    'loss': losses['total'],
                    'segm_loss': losses['segm'],
                    'graspness_loss': losses['graspness'],
                    'score_loss': losses['score'],
                    'width_loss': losses['width'],
                    'depth_loss': losses['depth'],
                    'graspness_mae': metrics['graspness_mae'],
                    'score_mae': metrics['score_mae'],
                    'approach_angle_error': metrics['approach_angle_error'],
                    'gripper_angle_error': metrics['gripper_angle_error'],
                    'width_mae': metrics['width_mae'],
                    'depth_mae': metrics['depth_mae'],
                }
                if 'orientation' in losses:
                    batch_log_values['orientation_loss'] = losses['orientation']
                else:
                    batch_log_values['approach_loss'] = losses['approach']
                    batch_log_values['gripper_loss'] = losses['gripper']
                for key in losses:
                    if key.startswith('ranking_'):
                        batch_log_values[key] = losses[key]
                peft_log_stats = get_peft_log_stats(model)
                batch_log_values.update(peft_log_stats)
                reduced_batch_logs = reduce_scalar_dict(batch_log_values, device, average=True)

            if batch_idx % config.training.print_freq == 0 and is_main_process():
                log_metrics = {
                    'train/loss': reduced_batch_logs['loss'],
                    'train/segm_loss': reduced_batch_logs['segm_loss'],
                    'train/graspness_loss': reduced_batch_logs['graspness_loss'],
                    'train/score_loss': reduced_batch_logs['score_loss'],
                    'train/width_loss': reduced_batch_logs['width_loss'],
                    'train/depth_loss': reduced_batch_logs['depth_loss'],
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/graspness_mae': reduced_batch_logs['graspness_mae'],
                    'train/score_mae': reduced_batch_logs['score_mae'],
                    'train/approach_angle_error': reduced_batch_logs['approach_angle_error'],
                    'train/gripper_angle_error': reduced_batch_logs['gripper_angle_error'],
                    'train/width_mae': reduced_batch_logs['width_mae'],
                    'train/depth_mae': reduced_batch_logs['depth_mae'],
                    'global_step': global_step
                }
                if 'orientation_loss' in reduced_batch_logs:
                    log_metrics['train/orientation_loss'] = reduced_batch_logs['orientation_loss']
                else:
                    log_metrics['train/approach_loss'] = reduced_batch_logs['approach_loss']
                    log_metrics['train/gripper_loss'] = reduced_batch_logs['gripper_loss']
                for key in losses:
                    if key.startswith('ranking_'):
                        log_metrics[f'train/{key}'] = reduced_batch_logs[key]
                for key in peft_log_stats:
                    log_metrics[f'train/{key}'] = reduced_batch_logs[key]
                logger.log(log_metrics, step=global_step)

            num_batches_processed += 1

            if batch_idx % config.training.print_freq == 0 and is_main_process():
                # Calculate time statistics for current progress
                current_time = time.time()
                elapsed_time = current_time - epoch_start_time
                batches_done = batch_idx + 1
                batches_total = len(train_loader)

                if batches_done >= 2:
                    time_per_batch = elapsed_time / batches_done
                    batches_remaining = batches_total - batches_done
                    eta_epoch_seconds = time_per_batch * batches_remaining
                    eta_epoch_mins = eta_epoch_seconds / 60

                    # Estimate total remaining time
                    if len(epoch_times) > 0:
                        avg_epoch_time = sum(epoch_times) / len(epoch_times)
                        epochs_remaining = config.training.num_epochs - epoch
                        total_eta_seconds = eta_epoch_seconds + (avg_epoch_time * (epochs_remaining - 1))
                        total_eta_hours = total_eta_seconds / 3600
                        total_eta_mins = (total_eta_seconds % 3600) / 60
                        time_info = f"ETA: {int(eta_epoch_mins)}m (epoch) | {int(total_eta_hours)}h {int(total_eta_mins)}m (total)"
                    else:
                        time_info = f"ETA: {int(eta_epoch_mins)}m (epoch)"
                else:
                    time_info = "ETA: warming up..."

                # Build ranking loss string if present
                ranking_str = ""
                for key, value in reduced_batch_logs.items():
                    if key.startswith('ranking_'):
                        target_name = key.replace('ranking_', '')
                        ranking_str += f"Rank_{target_name}: {value:.4f} | "

                if 'orientation_loss' in reduced_batch_logs:
                    orientation_str = (
                        f"JointOrientation: {reduced_batch_logs['orientation_loss']:.4f} | "
                    )
                else:
                    orientation_str = (
                        f"Approach: {reduced_batch_logs['approach_loss']:.4f} | "
                        f"Gripper: {reduced_batch_logs['gripper_loss']:.4f} | "
                    )

                print(f"  [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {reduced_batch_logs['loss']:.4f} | "
                      f"Segm: {reduced_batch_logs['segm_loss']:.4f} | "
                      f"Graspness: {reduced_batch_logs['graspness_loss']:.4f} | "
                      f"Score: {reduced_batch_logs['score_loss']:.4f} | "
                      f"{orientation_str}"
                      f"Width: {reduced_batch_logs['width_loss']:.4f} | "
                      f"Depth: {reduced_batch_logs['depth_loss']:.4f} | "
                      f"{ranking_str}"
                      f"{time_info}"
                )

            global_step += 1

            # Evaluation
            if global_step % config.training.evaluation_freq == 0:
                maybe_barrier()
                if is_main_process():
                    print("\n" + "="*80)
                    print(f"Running Evaluation at Step {global_step}...")
                    print("="*80)
                    val_loss, val_metrics, _ = evaluate(
                        base_model,
                        val_loader,
                        config,
                        epoch,
                        device,
                        log_dir=log_dir,
                        global_step=global_step,
                        eval_graspnet=validation_cfg.get('eval_graspnet', True),
                        collision_thresh=validation_cfg.get('collision_thresh', 0.0),
                        save_visualizations=validation_cfg.get('save_visualizations', True),
                        ann_ids_path=validation_cfg.get('ann_ids_path', None),
                    )

                    # Save best model based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                        current_avg_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else 0
                        checkpoint_data = {
                            'epoch': epoch + 1,
                            'global_step': global_step,
                            'model_state_dict': base_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss,
                            'val_loss': val_loss,
                            'train_loss': current_avg_loss,
                            'config': OmegaConf.to_container(config),
                            'rng_state': get_rng_state(),
                            'dataloader_generator_state': train_generator.get_state(),
                        }
                        if scheduler is not None:
                            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                        if scaler is not None:
                            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                        torch.save(checkpoint_data, best_model_path)
                        print(f"  ✓ New best model saved! Val Loss: {val_loss:.4f}")

                    # Return to training mode
                    model.train()
                    if config.model.backbone.freeze:
                        base_model.backbone.eval()
                
                    # Clear CUDA cache after evaluation to prevent fragmentation
                    torch.cuda.empty_cache()
                maybe_barrier()

            # Visualization
            if global_step % config.training.visualize_freq == 0 and is_main_process():
                with torch.no_grad():
                    vis_sample = {}
                    for key in data:
                        if isinstance(data[key], torch.Tensor):
                            vis_sample[key] = data[key][0]
                        else:
                            vis_sample[key] = data[key]
                    for key in output:
                        if isinstance(output[key], torch.Tensor):
                            vis_sample[f'pred_{key}'] = output[key][0]
                    viz_dir = os.path.join(log_dir, 'visualizations', 'training')
                    visualize_and_save_opencv(vis_sample, save_dir=viz_dir, prefix=f'step_{global_step}_')
                    del vis_sample

            # Explicitly free per-batch tensors to prevent memory accumulation
            del output, losses, metrics, valid_mask
            # Clear data tensors
            for key in list(data.keys()):
                if isinstance(data[key], torch.Tensor):
                    del data[key]
            del data


        # Epoch summary
        reduced_loss_sums = reduce_scalar_dict(
            {
                'total': epoch_loss,
                **epoch_losses,
                'num_batches': num_batches_processed,
            },
            device,
            average=False
        )
        reduced_metric_sums = reduce_scalar_dict(epoch_metrics, device, average=False)

        num_batches = max(reduced_loss_sums.pop('num_batches'), 1.0)
        avg_loss = reduced_loss_sums.pop('total') / num_batches

        # Calculate time statistics
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        # Estimate remaining time
        epochs_remaining = config.training.num_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * epochs_remaining
        eta_hours = eta_seconds / 3600
        eta_mins = (eta_seconds % 3600) / 60

        # Total elapsed time
        total_elapsed = time.time() - training_start_time
        elapsed_hours = total_elapsed / 3600
        elapsed_mins = (total_elapsed % 3600) / 60

        if is_main_process():
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1} Summary:")
            print(f"{'='*80}")
            print(f"  Time:")
            print(f"    Epoch Time: {epoch_time/60:.1f}m ({epoch_time:.1f}s)")
            print(f"    Avg Epoch Time: {avg_epoch_time/60:.1f}m")
            print(f"    Elapsed: {int(elapsed_hours)}h {int(elapsed_mins)}m")
            print(f"    ETA: {int(eta_hours)}h {int(eta_mins)}m ({epochs_remaining} epochs remaining)")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Losses:")
            print(f"    Segmentation: {reduced_loss_sums['segm']/num_batches:.4f}")
            print(f"    Graspness: {reduced_loss_sums['graspness']/num_batches:.4f}")
            print(f"    Score: {reduced_loss_sums['score']/num_batches:.4f}")
            if reduced_loss_sums['orientation'] != 0:
                print(f"    Joint Orientation: {reduced_loss_sums['orientation']/num_batches:.4f}")
            else:
                print(f"    Approach: {reduced_loss_sums['approach']/num_batches:.4f}")
                print(f"    Gripper: {reduced_loss_sums['gripper']/num_batches:.4f}")
            print(f"    Width: {reduced_loss_sums['width']/num_batches:.4f}")
            print(f"    Depth: {reduced_loss_sums['depth']/num_batches:.4f}")
            for key in reduced_loss_sums:
                if key.startswith('ranking_'):
                    target_name = key.replace('ranking_', '')
                    print(f"    Ranking ({target_name}): {reduced_loss_sums[key]/num_batches:.4f}")
            print(f"  Metrics:")
            print(f"    Graspness MAE: {reduced_metric_sums['graspness_mae']/num_batches:.4f}")
            print(f"    Score MAE: {reduced_metric_sums['score_mae']/num_batches:.4f}")
            print(f"    Approach Angle Error: {reduced_metric_sums['approach_angle_error']/num_batches:.2f}°")
            print(f"    Gripper Angle Error: {reduced_metric_sums['gripper_angle_error']/num_batches:.2f}°")
            print(f"    Width MAE: {reduced_metric_sums['width_mae']/num_batches:.4f}m")
            print(f"    Depth MAE: {reduced_metric_sums['depth_mae']/num_batches:.4f}m")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*80}")

        # Log epoch averages
        if is_main_process():
            epoch_wandb_log = {
                'epoch/avg_loss': avg_loss,
                'epoch/lr': optimizer.param_groups[0]['lr'],
                'epoch/graspness_mae': reduced_metric_sums['graspness_mae']/num_batches,
                'epoch/score_mae': reduced_metric_sums['score_mae']/num_batches,
                'epoch/approach_angle_error': reduced_metric_sums['approach_angle_error']/num_batches,
                'epoch/gripper_angle_error': reduced_metric_sums['gripper_angle_error']/num_batches,
                'epoch/width_mae': reduced_metric_sums['width_mae']/num_batches,
                'epoch/depth_mae': reduced_metric_sums['depth_mae']/num_batches,
                'time/epoch_time_minutes': epoch_time / 60,
                'time/avg_epoch_time_minutes': avg_epoch_time / 60,
                'time/elapsed_hours': elapsed_hours,
                'time/eta_hours': eta_hours,
                'epoch': epoch
            }
            if reduced_loss_sums['orientation'] != 0:
                epoch_wandb_log['epoch/orientation_loss'] = (
                    reduced_loss_sums['orientation']/num_batches
                )
            else:
                epoch_wandb_log['epoch/approach_loss'] = (
                    reduced_loss_sums['approach']/num_batches
                )
                epoch_wandb_log['epoch/gripper_loss'] = (
                    reduced_loss_sums['gripper']/num_batches
                )
            for key in reduced_loss_sums:
                if key.startswith('ranking_'):
                    epoch_wandb_log[f'epoch/{key}'] = reduced_loss_sums[key]/num_batches
            logger.log(epoch_wandb_log, step=global_step)


        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint every N epochs or at the last epoch
        if is_main_process() and ((epoch + 1) % config.training.get('checkpoint_freq', 5) == 0 or (epoch + 1) == config.training.num_epochs):
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            checkpoint_data = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_loss,
                'config': OmegaConf.to_container(config),
                'rng_state': get_rng_state(),
                'dataloader_generator_state': train_generator.get_state(),
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  Checkpoint saved to: {checkpoint_path}")

        maybe_barrier()

        # Kick off next epoch's DataLoader prefetch while checkpoint/logging is done.
        # Workers stay alive (persistent_workers) and start filling the prefetch buffer
        # so the first batch of the next epoch doesn't stall.
        next_epoch = epoch + 1
        if next_epoch < config.training.num_epochs:
            if train_sampler is not None:
                train_sampler.set_epoch(next_epoch)
            train_iter = iter(train_loader)

    # Save final model
    if is_main_process():
        final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save(base_model.state_dict(), final_model_path)
        print(f"\nTraining completed! Final model saved to: {final_model_path}")

    maybe_barrier()

    final_test_cfg = config.training.get('final_test', {})
    if is_main_process() and final_test_cfg.get('enabled', True):
        final_test_split = final_test_cfg.get('split', 'test')
        final_test_batch_size = final_test_cfg.get('batch_size', None) or config.training.batch_size
        final_test_num_workers = final_test_cfg.get('num_workers', None)
        if final_test_num_workers is None:
            final_test_num_workers = config.training.num_workers
        final_test_eval_graspnet = final_test_cfg.get('eval_graspnet', True)
        final_test_collision_thresh = final_test_cfg.get('collision_thresh', 0.0)
        final_test_save_visualizations = final_test_cfg.get('save_visualizations', False)
        final_test_ann_ids_path = final_test_cfg.get('ann_ids_path', None)
        inf_cfg = config.get('inference', {})
        final_test_use_cd = bool(inf_cfg.get('use_collision_detection', final_test_collision_thresh > 0))
        pred_subdir_name = 'grasp_predictions_cd' if final_test_use_cd else 'grasp_predictions'

        print(f"\n{'='*80}")
        print(f"Running Final Test Evaluation on split '{final_test_split}'...")
        print(f"{'='*80}")

        final_test_scene_ann_ids = load_scene_ann_ids(final_test_split, final_test_ann_ids_path)
        final_test_dataset = GraspNet1BDataset(
            root=config.dataset.root,
            camera=config.dataset.camera,
            split=final_test_split,
            image_size=tuple(config.dataset.image_size),
            backbone_name=config.model.backbone.name,
            augmentation=None,
            input_type=config.dataset.get('input_type', config.model.input_type),
            depth_corruption=None,
            scene_ann_ids=final_test_scene_ann_ids,
        )

        final_test_loader = DataLoader(
            final_test_dataset,
            batch_size=final_test_batch_size,
            shuffle=False,
            **build_dataloader_kwargs(config, num_workers=final_test_num_workers)
        )

        final_test_loss, _, final_graspnet_metrics = evaluate(
            base_model,
            final_test_loader,
            config,
            config.training.num_epochs,
            device,
            log_dir=log_dir,
            global_step=global_step,
            eval_graspnet=final_test_eval_graspnet,
            collision_thresh=final_test_collision_thresh,
            stage_name=f"Final Test ({final_test_split})",
            log_prefix='test',
            visualization_subdir=os.path.join('final_test', final_test_split),
            prediction_subdir=os.path.join('final_test', final_test_split, pred_subdir_name),
            graspnet_log_prefix='test/graspnet',
            save_visualizations=final_test_save_visualizations,
            use_amp=use_amp,
            ann_ids_path=final_test_ann_ids_path,
        )

        results = {
            'config': config_name,
            'split': final_test_split,
            'epoch': config.training.num_epochs,
            'avg_loss': final_test_loss,
            'graspnet_metrics': {k: round(float(v), 2) for k, v in final_graspnet_metrics.items()},
        }
        cd_tag = '_cd' if final_test_use_cd else ''
        results_path = os.path.join(log_dir, f'final_test_{final_test_split}{cd_tag}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Final test results saved to: {results_path}")

    gc6d_test_cfg = final_test_cfg.get('graspclutter6d', {})
    if is_main_process() and gc6d_test_cfg.get('enabled', False):
        gc6d_batch_size = gc6d_test_cfg.get('batch_size', None) or config.training.batch_size
        gc6d_num_workers = gc6d_test_cfg.get('num_workers', None)
        if gc6d_num_workers is None:
            gc6d_num_workers = config.training.num_workers
        _inf_cfg = config.get('inference', {})
        gc6d_use_cd = bool(_inf_cfg.get('use_collision_detection', False))

        gc6d_metrics = _run_gc6d_final_test(
            base_model=base_model,
            config=config,
            log_dir=log_dir,
            device=device,
            global_step=global_step,
            batch_size=gc6d_batch_size,
            num_workers=gc6d_num_workers,
            use_amp=use_amp,
        )

        gc6d_results = {
            'config': config_name,
            'split': 'test',
            'epoch': config.training.num_epochs,
            'gc6d_metrics': {k: round(float(v), 2) for k, v in gc6d_metrics.items()},
        }
        cd_tag = '_cd' if gc6d_use_cd else ''
        gc6d_results_path = os.path.join(log_dir, f'final_test_gc6d{cd_tag}_results.json')
        with open(gc6d_results_path, 'w') as f:
            json.dump(gc6d_results, f, indent=2)
        print(f"  GraspClutter6D final test results saved to: {gc6d_results_path}")

    # Finish experiment run
    if is_main_process():
        logger.finish()

    maybe_barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base.yaml')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the latest logs/<config>/checkpoints/checkpoint_epoch_*.pth',
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config_with_base(config_path)
    config.dataset.root = config.datasets[config.dataset.name].root
    if int(os.environ.get('RANK', '0')) == 0:
        print(OmegaConf.to_yaml(config))

    # Extract config name (without .yaml extension)
    config_name = config_name_from_path(config_path)

    try:
        main(config, config_name, resume=args.resume)
    finally:
        cleanup_distributed()
