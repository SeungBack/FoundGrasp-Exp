import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from model.found_grasp import FoundGrasp
from utils.config import load_config_with_base, resolve_config_path


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def depth_to_xyz(depth, intrinsics):
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32, copy=False)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def estimate_intrinsics(height, width, fov_deg=60.0):
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class GraspDetector(ABC):
    @abstractmethod
    def detect(
        self,
        color,
        depth,
        intrinsics,
        top_k=30,
        nms=True,
        use_collision_detection=True,
        collision_voxel_size=0.01,
    ):
        """Return a graspnetAPI.grasp.GraspGroup in the camera frame."""
        raise NotImplementedError


class FoundGraspDetector(GraspDetector):
    def __init__(self, config, checkpoint, device="cuda"):
        self.device = torch.device(device)
        self.config_path = resolve_config_path(config)
        self.config = load_config_with_base(self.config_path)
        self.model = FoundGrasp(self.config.model)

        ckpt = load_checkpoint(checkpoint, self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(3, 1, 1)

        self.last_color = None
        self.last_depth = None
        self.last_xyz = None
        self.last_vis_sample = None
        self.last_predictions = None

    def detect(
        self,
        color,
        depth,
        intrinsics,
        top_k=50,
        nms=True,
        use_collision_detection=True,
        collision_voxel_size=0.01,
    ):
        data, xyz_np = self._prepare_input(color, depth, intrinsics)

        with torch.no_grad():
            predictions = self.model(data)
        predictions = {
            k: v.float() if isinstance(v, torch.Tensor) else v
            for k, v in predictions.items()
        }

        output = {k: v[0] for k, v in predictions.items()}
        inf_cfg = self.config.get("inference", {})
        max_grasps = int(inf_cfg.get("max_grasps", max(int(top_k or 0), 2048)))

        gg = self.model.to_grasp_group(
            predictions=output,
            xyz=data["xyz"][0],
            graspness_threshold=inf_cfg.get("graspness_threshold", 0.01),
            objectness_threshold=inf_cfg.get("objectness_threshold", 0.5),
            graspness_exp=inf_cfg.get("graspness_exp", 0.1),
            grasp_score_exp=inf_cfg.get("grasp_score_exp", 2.0),
            total_score_threshold=inf_cfg.get("total_score_threshold", None),
            approach_kappa_risk_weight=inf_cfg.get("approach_kappa_risk_weight", 0.5),
            gripper_kappa_risk_weight=inf_cfg.get("gripper_kappa_risk_weight", 1.0),
            width_multiplier=inf_cfg.get("width_multiplier", 1.2),
            z_max=inf_cfg.get("z_max", 1.5),
            empty_grasp_min_points=inf_cfg.get("empty_grasp_min_points", 0),
            segm_erosion_k=inf_cfg.get("segm_erosion_k", 0),
            max_grasps=max_grasps,
            voxel_size=inf_cfg.get("voxel_size", 0.004),
        )

        if len(gg) > 0:
            gg = gg.sort_by_score()
            if use_collision_detection:
                gg = self._filter_collision_free(gg, data["xyz"][0], collision_voxel_size)
            # to_grasp_group already performs voxel candidate reduction.
            # This final GraspGroup NMS is for execution-time grasp diversity.
            if nms and len(gg) > 0:
                gg = gg.nms(translation_thresh=0.05, rotation_thresh=30.0 / 180.0 * np.pi)
            if top_k is not None and top_k > 0 and len(gg) > top_k:
                gg = gg[:top_k]

        if len(gg) == 0:
            print("Warning: No grasps detected after inference filtering.")

        self.last_xyz = xyz_np
        self.last_predictions = predictions
        self.last_vis_sample = self._make_vis_sample(data, predictions)
        return gg

    def _filter_collision_free(self, gg, xyz, voxel_size):
        from utils.collision_detector import ModelFreeCollisionDetectorGPU

        if len(gg) == 0:
            return gg

        points = _flatten_valid_xyz_points(xyz)
        mfcdetector = ModelFreeCollisionDetectorGPU(points, voxel_size=voxel_size, device=self.device)
        gg_array = torch.from_numpy(gg.grasp_group_array).to(self.device)
        no_collision_mask = mfcdetector.detect(gg_array, min_inner_points=5)
        return gg[no_collision_mask]

    def _prepare_input(self, color, depth, intrinsics):
        color = np.asarray(color)
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError(f"color must have shape (H, W, 3), got {color.shape}")
        if color.dtype != np.uint8:
            color = np.clip(color, 0, 255).astype(np.uint8)

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth must have shape (H, W), got {depth.shape}")
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")

        target_h, target_w = tuple(self.config.dataset.image_size)
        color_resized = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        xyz = depth_to_xyz(depth, intrinsics)
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        xyz_resized = cv2.resize(xyz, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        color_t = torch.from_numpy(np.ascontiguousarray(color_resized)).to(self.device)
        color_t = color_t.permute(2, 0, 1).float() / 255.0
        pixel_values = ((color_t - self.mean) / self.std).unsqueeze(0)
        og_color = color_t.unsqueeze(0)

        depth_t = torch.from_numpy(np.ascontiguousarray(depth_resized)).to(self.device)
        depth_t = depth_t.unsqueeze(0).unsqueeze(0).float()
        xyz_t = torch.from_numpy(np.ascontiguousarray(xyz_resized)).to(self.device)
        xyz_t = xyz_t.permute(2, 0, 1).float().unsqueeze(0)
        depth_mask = (depth_t > 0.1).float()

        self.last_color = color_resized
        self.last_depth = depth_resized

        return {
            "pixel_values": pixel_values,
            "og_color": og_color,
            "xyz": xyz_t,
            "depth": depth_t,
            "depth_mask": depth_mask,
        }, xyz_resized

    def _make_vis_sample(self, data, predictions):
        sample = {
            "og_color": data["og_color"][0].detach().cpu(),
            "xyz": data["xyz"][0].detach().cpu(),
            "depth": data["depth"][0].detach().cpu(),
        }
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                sample[f"pred_{key}"] = value[0].detach().cpu()
        return sample


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
