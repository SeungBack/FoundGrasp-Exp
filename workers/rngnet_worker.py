"""RNGNet worker using RegionNormalizedGrasp test decoding and native collision.

The model keeps its 1280x720 camera grid and 640x360 anchor input. The supplied
intrinsics follow that resize; output poses and learned widths stay in camera
metres. Invalid/ROI-masked depth is never replaced by another object's mean
depth. Final execution opening margins belong to the shared host pipeline.
Anchor depth is centred using valid pixels only, so ROI zeros do not shift the
model's depth baseline. Metric depth, local patches and collision input stay masked.
The optional collision stage in test_graspnet.py is enabled, matching the
FoundGrasp simulation adapter; demo-only score filtering/final NMS are omitted.
"""
import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from efg.worker_protocol import serve

CAMERA_SIZE = (1280, 720)
INPUT_SIZE = (640, 360)
EMPTY = np.empty((0, 17), dtype=np.float64)


def normalize_depth(depth_m):
    """Centre one resized depth tensor, or return None if it has no valid pixels.

    The upstream demo subtracts the full-image mean. After the shared ROI mask,
    counting zero placeholders in that mean would change every observed depth
    and make the anchor heatmap respond to the artificial ROI boundary. Ignore
    these placeholders when calculating the mean; keep their existing negative
    centred sentinel in the model input. The original metric tensor is untouched.
    """
    valid = depth_m > 0
    if not valid.any():
        return None
    return (depth_m - depth_m[valid].mean()).clamp(-1, 1)


def prepare_frame(rgb, depth_m, K, *, depth_max_m=1.5):
    """Validate RGB-D and resize to the released model's native camera grid."""
    import cv2

    rgb, depth = np.asarray(rgb), np.asarray(depth_m)
    K = np.asarray(K, dtype=np.float64)
    if (depth.ndim != 2 or not depth.size or rgb.shape != (*depth.shape, 3)
            or rgb.dtype != np.uint8):
        raise ValueError('RNGNet requires aligned uint8 RGB and HxW depth in metres')
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0., 0., 1.])
            or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8):
        raise ValueError('RNGNet requires finite pinhole intrinsics with positive focal lengths')
    if not np.isfinite(depth_max_m) or depth_max_m <= 0:
        raise ValueError('depth_max_m must be finite and positive')
    valid = np.isfinite(depth) & (depth > 0) & (depth <= depth_max_m)
    depth_mm = np.where(valid, depth, 0).astype(np.float32) * 1000.
    h, w = depth.shape
    camera_w, camera_h = CAMERA_SIZE
    scaled_K = np.diag([camera_w / w, camera_h / h, 1.]) @ K
    if (w, h) != CAMERA_SIZE:
        rgb = cv2.resize(rgb, CAMERA_SIZE, interpolation=cv2.INTER_LINEAR)
        depth_mm = cv2.resize(depth_mm, CAMERA_SIZE, interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth_mm), scaled_K


def xyz_from_depth(depth_mm, K):
    """H,W,3 optical-frame coordinates, with exact zeros at invalid depth."""
    z = np.asarray(depth_mm, dtype=np.float32) / 1000.
    h, w = z.shape
    x = z * ((np.arange(w, dtype=np.float32) - K[0, 2]) / K[0, 0])[None, :]
    y = z * ((np.arange(h, dtype=np.float32) - K[1, 2]) / K[1, 1])[:, None]
    return np.ascontiguousarray(np.stack([x, y, z], axis=-1), dtype=np.float32)


def project_centers(centers, depth_mm, K):
    """Lift anchor-grid (u,v) centres only where a real depth sample exists.

    On valid pixels this is the upstream center2dtopc inference equation. Its
    fallback to neighbouring/global mean depths is deliberately excluded: zero
    depth also encodes the shared experiment ROI and must remain invalid.
    """
    centers = np.asarray(centers)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('RNGNet centres must have shape (N,2)')
    h, w = depth_mm.shape
    scaled = centers * np.array([w / INPUT_SIZE[0], h / INPUT_SIZE[1]])
    finite = np.isfinite(scaled).all(axis=1)
    inside = (finite & (scaled[:, 0] >= 0) & (scaled[:, 0] < w)
              & (scaled[:, 1] >= 0) & (scaled[:, 1] < h))
    pixels = np.zeros((len(centers), 2), dtype=np.int64)
    pixels[inside] = scaled[inside].astype(np.int64)
    depth = np.zeros(len(centers), dtype=np.float32)
    depth[inside] = depth_mm[pixels[inside, 1], pixels[inside, 0]]
    valid = inside & np.isfinite(depth) & (depth > 0)
    uv, z = pixels[valid], depth[valid] / 1000.
    xyz = np.column_stack(((uv[:, 0] - K[0, 2]) / K[0, 0] * z,
                           (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z))
    return valid, np.ascontiguousarray(xyz, dtype=np.float32)


def grasp_array(group, *, alpha=.02):
    """Official test.py export; retain learned widths without hidden margins."""
    n = len(group)
    rows = np.zeros((n, 17), dtype=np.float64)
    if n:
        rows[:, 0], rows[:, 1] = group.scores, group.widths
        rows[:, 2] = np.asarray(group.heights) / alpha * .02
        rows[:, 3] = np.asarray(group.depths) / alpha * .02
        rows[:, 4:13] = np.asarray(group.rotations).reshape(n, 9)
        rows[:, 13:16] = group.translations
        rows[:, 16] = -1
        if not np.isfinite(rows).all():
            raise RuntimeError('RNGNet returned non-finite grasp data')
    return rows


class RNGNetRuntime:
    def __init__(self, repo_dir, checkpoint, *, seed=0, depth_max_m=1.5):
        if not np.isfinite(depth_max_m) or depth_max_m <= 0:
            raise ValueError('depth_max_m must be finite and positive')
        repo = Path(repo_dir).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not (repo / 'models/localgraspnet.py').is_file():
            raise FileNotFoundError(f'RegionNormalizedGrasp repository missing: {repo}')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'RNGNet checkpoint missing: {checkpoint}')
        sys.path.insert(0, str(repo))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('RNGNet requires a CUDA device')
        from models.anchornet import AnchorGraspNet
        from models.localgraspnet import PatchMultiGraspNet
        from dataset import evaluation

        self.torch, self.evaluation = torch, evaluation
        self.depth_max_m = depth_max_m
        self.current_K = np.eye(3)
        # Upstream helpers import the function by name; bind only their own
        # dataset modules, never unrelated modules in the host process.
        for name in ('config', 'evaluation', 'pc_dataset_tools', 'utils', 'grasp', 'graspnet_utils'):
            module = importlib.import_module('dataset.' + name)
            module.get_camera_intrinsic = lambda camera=None: self.current_K
        print('RNGNet intrinsics: supplied frame K replaces the upstream camera preset', flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        self.anchornet = AnchorGraspNet(in_dim=4, ratio=8, anchor_k=6, feature_dim=128).cuda().eval()
        self.localnet = PatchMultiGraspNet(49, theta_k_cls=6, feat_dim=256, anchor_w=60.).cuda().eval()
        weights = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        strip = lambda state: {key: value for key, value in state.items()
                               if not key.endswith(('total_ops', 'total_params'))}
        self.anchornet.load_state_dict(strip(weights['anchor']))
        self.localnet.load_state_dict(strip(weights['local']))
        self.anchors = {name: weights[name].cpu() for name in ('gamma', 'beta')}
        import cupoch
        self.metadata = {
            'python': sys.version.split()[0], 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'seed': seed, 'repo_dir': str(repo), 'checkpoint': str(checkpoint),
            'inference_profile': 'test_graspnet_decoding_with_native_collision',
            'width_semantics': 'learned_width_m', 'internal_width_multiplier': 1.,
            'camera_size': list(CAMERA_SIZE), 'network_size': list(INPUT_SIZE),
            'network_resize_interpolation': 'nearest',
            'depth_normalization': 'positive_depth_mean',
            'depth_normalization_clip_m': [-1., 1.],
            'depth_normalization_reference': 'ROI-zero-aware adaptation of upstream global mean',
            'intrinsics_source': 'request_K_scaled_to_native_grid',
            'depth_max_m': depth_max_m, 'invalid_depth_centers': 'reject_without_inpainting',
            'center_count': 48, 'center_nms_px': 12, 'patch_size': 64,
            'local_k': 10, 'anchor_width_mm': 60., 'final_grasp_nms': False,
            'scene_points_max': 25600, 'gamma_limit_deg': 60.,
            'collision_detection': True, 'collision_voxel_size_m': .01,
            'collision_approach_dist_m': .05, 'collision_mode': 'graspnet',
            'collision_dtype': 'float16',
            'voxel_backend': getattr(cupoch, '__version__', 'cupoch'),
        }

    def predict(self, rgb, depth_m, K):
        rgb, depth_mm, K = prepare_frame(rgb, depth_m, K, depth_max_m=self.depth_max_m)
        if not np.any(depth_mm > 0):
            return EMPTY.copy()
        self.current_K = K
        torch, ev = self.torch, self.evaluation
        import torch.nn.functional as F

        # Upstream tensors intentionally use W,H rather than conventional H,W.
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.).permute(2, 1, 0).cuda()[None]
        depth_t = torch.from_numpy(depth_mm.T.copy()).cuda()[None]
        xyz = torch.from_numpy(xyz_from_depth(depth_mm, K)).permute(2, 1, 0).cuda()
        rgbd = torch.cat((rgb_t[0], xyz), dim=0)
        points = xyz.permute(1, 2, 0)[depth_t[0] > 0]
        if len(points) > 25600:
            points = points[random.sample(range(len(points)), 25600)]
        rgb_small = F.interpolate(rgb_t, INPUT_SIZE)
        depth_small = F.interpolate(depth_t[None], INPUT_SIZE)[0] / 1000.
        centered = normalize_depth(depth_small)
        if centered is None:
            return EMPTY.copy()
        model_input = torch.cat((centered[None], rgb_small), dim=1)

        with torch.inference_mode():
            pred_2d, _ = self.anchornet(model_input)
            loc_map = ev.anchor_output_process(*pred_2d, sigma=10)[0]
            loc_map[depth_small[0].cpu().numpy() <= 0] = 0
            rect = ev.sample_2d_centers(loc_map, 12, 48, 12)
            keep, center_xyz = project_centers(rect.centers, depth_mm, K)
            rect = rect[keep]
            if not len(center_xyz):
                return EMPTY.copy()
            centers = torch.from_numpy(center_xyz).cuda()
            xs = torch.linspace(0, 1, 64, device='cuda', dtype=torch.float32)
            grid = torch.stack(torch.meshgrid(xs, xs, indexing='ij'), dim=-1) - .5
            centers_px = 2 * torch.from_numpy(rect.centers).cuda()
            radius = .10 * 2 * K[0, 0] / centers[:, 2]
            grid = grid[None] * radius[:, None, None, None]
            grid = grid + centers_px[:, None, None].flip(-1)
            grid = grid / torch.tensor([719., 1279.], device='cuda') * 2 - 1
            patches = F.grid_sample(rgbd[None].expand(len(centers), -1, -1, -1),
                                    grid, mode='nearest', align_corners=False)
            patches = patches.permute(0, 3, 2, 1).contiguous()
            valid = patches[..., -1:] > 0
            patches[..., 3:] -= valid * centers[:, None, None]
            patches[..., 3:] /= .060
            _, pred, offset, theta_cls, theta_offset, width_reg = self.localnet(patches)
            # Preserve N=1: upstream squeeze() otherwise loses the centre axis.
            theta_cls = theta_cls.sigmoid().clamp(1e-6, 1 - 1e-6).cpu().numpy().reshape(len(centers), -1)
            theta_offset = theta_offset.clamp(-.5, .5).cpu().numpy().reshape(len(centers), -1)
            width_reg = width_reg.cpu().numpy().reshape(len(centers), -1)
            thetas, widths = ev.get_thetas_widths(theta_cls, theta_offset, width_reg,
                                                 anchor_w=60., rotation_num=1)
            _, grasps = ev.detect_6d_grasp_multi(thetas, widths, pred, offset, [centers],
                                                self.anchors, alpha=.02, k=10)
            if not len(grasps):
                return EMPTY.copy()
            grasps, _ = ev.collision_detect(points, grasps, mode='graspnet')
            return grasp_array(grasps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--repo_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--depth_max_m', type=float, default=1.5)
    args = parser.parse_args()
    runtime = RNGNetRuntime(args.repo_dir, args.checkpoint,
                            seed=args.seed, depth_max_m=args.depth_max_m)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'runtime.json').write_text(json.dumps(runtime.metadata, indent=2))
    print(f'-> RNGNet loaded {args.checkpoint}; width=learned_width_m; native collision=True', flush=True)
    serve(str(work_dir), runtime.predict, poll_interval=.005,
          synchronize=runtime.torch.cuda.synchronize)


if __name__ == '__main__':
    main()
