import os
os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'stdlib')
import warnings

import numpy as np
import scipy.io as scio
import json
import torch
import collections.abc as container_abcs

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import time 

from transformers import AutoImageProcessor
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import cv2

import sys
# BASE_DIR/ROOT_DIR setup... (환경에 맞게 유지)
import albumentations as A
import numpy as np
import scipy.ndimage as ndimage
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.visualization import visualize_and_save_opencv


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


def _resolve_image_stats(backbone_name):
    lower_name = str(backbone_name).lower()

    # Depth Anything encoders follow DINOv2/ImageNet normalization,
    # but HF image processors are not always published for these repos.
    if "depth-anything" in lower_name or "da3" in lower_name:
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    try:
        processor = AutoImageProcessor.from_pretrained(backbone_name, use_fast=True)
        image_mean = getattr(processor, "image_mean", None) or IMAGENET_DEFAULT_MEAN
        image_std = getattr(processor, "image_std", None) or IMAGENET_DEFAULT_STD
        return image_mean, image_std
    except OSError as exc:
        warnings.warn(
            f"Falling back to ImageNet normalization for backbone '{backbone_name}' "
            f"because no HF image processor was found: {exc}"
        )
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_int_range(value, default):
    if value is None:
        return default
    if isinstance(value, container_abcs.Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        lo, hi = int(value[0]), int(value[1])
        return (min(lo, hi), max(lo, hi))
    v = int(value)
    return (v, v)


def _normalize_float_range(value, default):
    if value is None:
        return default
    if isinstance(value, container_abcs.Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        return (min(lo, hi), max(lo, hi))
    v = float(value)
    return (v, v)


class GraspNet1BDataset(Dataset):
    def __init__(self, root, camera='realsense', split='debug',
                 image_size=(720, 1280), # (H, W)
                 backbone_name='facebook/dinov2-base',
                 augmentation='None',
                 input_type='dxyzc',
                 depth_corruption=None,
                 color_aug_strength=1.0,
                 scene_ann_ids=None,     # dict[int scene_id -> list[int ann_id]] for test_mini_mixed
                 pixelwise_gt_dir='pixelwise_grasp_gt_top300',
                 depth_root=None,        # mirror root holding replacement depth (e.g. learned completion); None = sensor depth
                 xyz_from_depth=False,   # unproject the loaded depth image instead of reading the stored sensor cloud
                 ):
        self.root = root
        self.xyz_from_depth = xyz_from_depth
        self._camK_cache = {}
        self.split = split
        camera = {'realsense-d435': 'realsense', 'azure-kinect': 'kinect'}.get(camera, camera)
        self.camera = camera
        self.image_size = image_size  # (H, W)
        self.input_type = input_type

        # Scene split definition
        scene_ranges = {
            'train': range(100),
            'test': range(100, 190),
            'test_seen': range(100, 130),
            'test_similar': range(130, 160),
            'test_novel': range(160, 190),
            'test_mini_seen': range(100, 130),
            'test_mini_similar': range(130, 160),
            'test_mini_novel': range(160, 190),
            'test_mini': range(100, 190),
            'test_mini_mixed': range(100, 190),
            'debug': [0]
        }

        if split not in scene_ranges:
            raise ValueError(f'Unknown split: {split}')

        self.sceneIds = [f'scene_{str(x).zfill(4)}' for x in scene_ranges[split]]
        default_ann_range = range(5) if (split.startswith('test_mini') and split != 'test_mini_mixed') else range(256)

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.pixelgtpath = []
        self.scenename = []
        self.frameid = []

        scene_root = os.path.join(root, 'scenes')
        pixelgt_root = os.path.join(root, pixelwise_gt_dir, camera)

        for scene_id in tqdm(self.sceneIds, desc='Loading scene data...'):
            scene_int = int(scene_id.split('_')[1])
            if scene_ann_ids is not None and scene_int in scene_ann_ids:
                ann_range = scene_ann_ids[scene_int]
            else:
                ann_range = default_ann_range
            scene_path = os.path.join(scene_root, scene_id, camera)
            for img_num in ann_range:
                img_str = str(img_num).zfill(4)
                self.colorpath.append(os.path.join(scene_path, 'rgb', f'{img_str}.png'))
                self.depthpath.append(os.path.join(depth_root or root, 'scenes', scene_id, camera, 'depth', f'{img_str}.png'))
                self.labelpath.append(os.path.join(scene_path, 'label', f'{img_str}.png'))
                self.pixelgtpath.append(os.path.join(pixelgt_root, f'{scene_id}_ann_{img_str}.npz'))
                self.scenename.append(scene_id)
                self.frameid.append(img_num)


        # Backbone-specific input normalization.
        self.image_mean, self.image_std = _resolve_image_stats(backbone_name)

        self.depth_corruption_type = str(_cfg_get(depth_corruption, 'type', 'none')).lower()
        self.depth_corruption_prob = float(_cfg_get(depth_corruption, 'prob', 0.0))
        self.depth_corruption_patches = _normalize_int_range(
            _cfg_get(depth_corruption, 'num_patches', _cfg_get(depth_corruption, 'num_blobs', (2, 5))),
            (2, 5),
        )
        self.depth_corruption_patch_size = _normalize_int_range(
            _cfg_get(depth_corruption, 'patch_size_range', _cfg_get(depth_corruption, 'radius_range', (12, 40))),
            (12, 40),
        )
        self.depth_corruption_edge_dropout = float(_cfg_get(depth_corruption, 'edge_dropout', 0.0))
        self.depth_corruption_edge_kernel = max(1, int(_cfg_get(depth_corruption, 'edge_kernel', 5)))
        self.depth_corruption_edge_grad_thresh = float(_cfg_get(depth_corruption, 'edge_grad_thresh', 0.02))
        self.depth_corruption_far_bias = float(_cfg_get(depth_corruption, 'far_bias', 0.0))
        self.depth_corruption_sigma_range = _normalize_float_range(
            _cfg_get(depth_corruption, 'sigma_range', (0.001, 0.002)),
            (0.001, 0.002),
        )
        self.depth_corruption_sigma_depth_scale = float(_cfg_get(depth_corruption, 'sigma_depth_scale', 0.5))
        # lowres_factor > 1: 노이즈를 저해상도에서 생성 후 bilinear 업샘플 → 공간상관 노이즈.
        # (실측 센서 오차는 물체/영역 단위로 coherent — iid보다 실제 분포에 가깝고,
        #  crop-resize 시 sparse GT 좌표의 1px 반올림에도 GT 동시 변환이 강건해짐)
        self.depth_corruption_lowres = max(1, int(_cfg_get(depth_corruption, 'lowres_factor', 1)))
        # object_coherent: one random axial offset per segmented object (plus the low-res field), matching the
        # measured sensor-vs-annotation depth error, which is coherent per object rather than per pixel
        self.depth_corruption_object_coherent = bool(_cfg_get(depth_corruption, 'object_coherent', False))
        self._depth_corruption_enabled = (
            self.depth_corruption_type in {'sensor_style', 'axial_noise'}
            and self.depth_corruption_prob > 0.0
        )
        
        # Augmentation Setup
        color_aug_list = []
        geometric_aug_list = []

        if augmentation and 'color' in augmentation:
            s = color_aug_strength
            color_aug_list = [
                A.ColorJitter(brightness=0.4*s, contrast=0.4*s, saturation=0.4*s, hue=min(0.1*s, 0.5), p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(p=0.5)
            ]

        if augmentation and 'crop' in augmentation:
            # RandomSizedCrop: Crop followed by Resize to size arg
            # 여기서는 size=self.image_size이므로, Crop 후 다시 1280x720으로 복구됨
            geometric_aug_list.append(
                A.RandomSizedCrop(
                    min_max_height=(int(self.image_size[0]*0.6), self.image_size[0]),
                    w2h_ratio=self.image_size[1]/self.image_size[0],
                    size=self.image_size, p=1.0
                )
            )

        # Flip augmentations
        self.hflip = 'hflip' in augmentation if augmentation else False
        self.vflip = 'vflip' in augmentation if augmentation else False

        self.color_tfs = A.Compose(color_aug_list) if color_aug_list else None

        # Geometric transform (Crop Only in Dataset)
        self.geometric_tfs = A.ReplayCompose(geometric_aug_list, additional_targets={
            'depth': 'mask',
            'segm': 'mask',
        }) if geometric_aug_list else None

    def _apply_depth_corruption(self, depth_img, xyz_dense, segm_img):
        if not self._depth_corruption_enabled or np.random.rand() >= self.depth_corruption_prob:
            return depth_img, xyz_dense

        if self.depth_corruption_type == 'axial_noise':
            return self._apply_axial_depth_noise(depth_img, xyz_dense, segm_img)

        valid_mask = depth_img > 0.1
        if not np.any(valid_mask):
            return depth_img, xyz_dense

        # Lightweight sensor-style corruption:
        # object-aware random patch masking + depth-edge shadow dropouts.
        invalid_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
        obj_valid_mask = (segm_img > 0) & valid_mask
        valid_coords = np.column_stack(np.nonzero(obj_valid_mask))

        num_patches_min, num_patches_max = self.depth_corruption_patches
        if num_patches_max > 0 and len(valid_coords) > 0:
            num_patches = np.random.randint(num_patches_min, num_patches_max + 1)
            patch_min, patch_max = self.depth_corruption_patch_size
            max_depth = max(float(depth_img.max()), 1e-6)
            for _ in range(num_patches):
                cy, cx = valid_coords[np.random.randint(len(valid_coords))]
                obj_id = segm_img[cy, cx]
                if obj_id <= 0:
                    continue

                patch_h = np.random.randint(patch_min, patch_max + 1)
                patch_w = np.random.randint(patch_min, patch_max + 1)
                if self.depth_corruption_far_bias > 0.0:
                    depth_scale = np.clip(depth_img[cy, cx] / max_depth, 0.0, 1.0)
                    size_scale = 1.0 + self.depth_corruption_far_bias * depth_scale
                    patch_h = max(1, int(round(patch_h * size_scale)))
                    patch_w = max(1, int(round(patch_w * size_scale)))

                y0 = max(0, int(cy - patch_h // 2))
                y1 = min(depth_img.shape[0], int(cy + (patch_h + 1) // 2))
                x0 = max(0, int(cx - patch_w // 2))
                x1 = min(depth_img.shape[1], int(cx + (patch_w + 1) // 2))
                local_mask = (segm_img[y0:y1, x0:x1] == obj_id) & valid_mask[y0:y1, x0:x1]
                invalid_mask[y0:y1, x0:x1] |= local_mask.astype(np.uint8)

        if self.depth_corruption_edge_dropout > 0.0:
            depth_valid = depth_img * valid_mask.astype(np.float32)
            grad_x = cv2.Sobel(depth_valid, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_valid, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.abs(grad_x) + np.abs(grad_y)
            edge_mask = (grad_mag > self.depth_corruption_edge_grad_thresh) & valid_mask

            if self.depth_corruption_edge_kernel > 1:
                kernel = np.ones(
                    (self.depth_corruption_edge_kernel, self.depth_corruption_edge_kernel),
                    dtype=np.uint8,
                )
                edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

            if np.any(edge_mask):
                edge_prob = np.full(depth_img.shape, self.depth_corruption_edge_dropout, dtype=np.float32)
                if self.depth_corruption_far_bias > 0.0:
                    far_weight = np.clip(depth_img / max(depth_img.max(), 1e-6), 0.0, 1.0)
                    edge_prob = np.clip(
                        edge_prob * (1.0 + self.depth_corruption_far_bias * far_weight),
                        0.0,
                        1.0,
                    )
                invalid_mask |= (edge_mask & (np.random.rand(*depth_img.shape) < edge_prob)).astype(np.uint8)

        invalid_mask = invalid_mask.astype(bool) & valid_mask
        if not np.any(invalid_mask):
            return depth_img, xyz_dense

        depth_img = depth_img.copy()
        xyz_dense = xyz_dense.copy()
        depth_img[invalid_mask] = 0.0
        xyz_dense[invalid_mask] = 0.0
        return depth_img, xyz_dense

    def _apply_axial_depth_noise(self, depth_img, xyz_dense, segm_img=None):
        valid_mask = (depth_img > 0.1) & (xyz_dense[..., 2] > 1e-6)
        if not np.any(valid_mask):
            return depth_img, xyz_dense

        depth_img = depth_img.copy()
        xyz_dense = xyz_dense.copy()

        sigma_min, sigma_max = self.depth_corruption_sigma_range
        sigma_base = np.random.uniform(sigma_min, sigma_max)
        sigma_map = sigma_base * (1.0 + self.depth_corruption_sigma_depth_scale * np.square(depth_img))

        if self.depth_corruption_lowres > 1:
            lh = max(1, depth_img.shape[0] // self.depth_corruption_lowres)
            lw = max(1, depth_img.shape[1] // self.depth_corruption_lowres)
            base = np.random.randn(lh, lw).astype(np.float32)
            base = cv2.resize(base, (depth_img.shape[1], depth_img.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        else:
            base = np.random.randn(*depth_img.shape).astype(np.float32)
        if self.depth_corruption_object_coherent and segm_img is not None:
            ids = np.unique(segm_img[segm_img > 0])
            if len(ids):
                offs = np.random.randn(int(ids.max()) + 1).astype(np.float32)   # one standard-normal draw per object
                obj = segm_img > 0
                base = base.copy()
                base[obj] = offs[segm_img[obj].astype(np.int64)] + 0.25 * base[obj]   # object offset + weak local field
        noise = base * sigma_map.astype(np.float32)
        depth_noisy = depth_img.astype(np.float32, copy=True)
        depth_noisy[valid_mask] = np.clip(depth_noisy[valid_mask] + noise[valid_mask], 1e-4, None)

        z_clean = xyz_dense[..., 2].astype(np.float32, copy=False)
        scale = np.ones_like(depth_noisy, dtype=np.float32)
        scale[valid_mask] = depth_noisy[valid_mask] / np.maximum(z_clean[valid_mask], 1e-6)

        xyz_dense[..., 0][valid_mask] = xyz_dense[..., 0][valid_mask] * scale[valid_mask]
        xyz_dense[..., 1][valid_mask] = xyz_dense[..., 1][valid_mask] * scale[valid_mask]
        xyz_dense[..., 2][valid_mask] = depth_noisy[valid_mask]

        return depth_noisy.astype(depth_img.dtype, copy=False), xyz_dense

    def _unproject(self, depth_img, scene_name):
        K = self._camK_cache.get(scene_name)
        if K is None:
            K = np.load(os.path.join(self.root, 'scenes', scene_name, self.camera, 'camK.npy')).astype(np.float32)
            self._camK_cache[scene_name] = K
        h, w = depth_img.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return np.stack([(u - K[0, 2]) * depth_img / K[0, 0], (v - K[1, 2]) * depth_img / K[1, 1], depth_img], -1)

    def __len__(self):
        return len(self.depthpath)

    def __getitem__(self, index):
        sample = {}

        # ------------------------------------------------------------------
        # 1. Load Data
        # ------------------------------------------------------------------
        color_img = cv2.cvtColor(cv2.imread(self.colorpath[index]), cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(self.depthpath[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        segm_img = cv2.imread(self.labelpath[index], cv2.IMREAD_UNCHANGED)

        with np.load(self.pixelgtpath[index], allow_pickle=False) as data:
            gt_sparse = {key: data[key] for key in data.files}

        xyz_dense = gt_sparse['xyz'] # (H, W, 3) Numpy
        if self.xyz_from_depth:   # the stored cloud is the sensor depth unprojected once; redo it so a replacement depth takes effect
            xyz_dense = self._unproject(depth_img, self.scenename[index])

        # ------------------------------------------------------------------
        # 2. Base Resize (to self.image_size if needed)
        # ------------------------------------------------------------------
        original_size = depth_img.shape[:2]
        need_resize = (original_size[0] != self.image_size[0]) or (original_size[1] != self.image_size[1])

        if need_resize:
            target_size_cv = (self.image_size[1], self.image_size[0]) # (W, H)
            
            color_img = cv2.resize(color_img, target_size_cv, interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.resize(depth_img, target_size_cv, interpolation=cv2.INTER_NEAREST)
            segm_img = cv2.resize(segm_img, target_size_cv, interpolation=cv2.INTER_NEAREST)
            xyz_dense = cv2.resize(xyz_dense, target_size_cv, interpolation=cv2.INTER_NEAREST)

            scale_h = self.image_size[0] / original_size[0]
            scale_w = self.image_size[1] / original_size[1]
            
            # Sparse 좌표 스케일링
            gt_sparse['valid_pixel_indices'] = (
                gt_sparse['valid_pixel_indices'].astype(np.float32) * np.array([scale_h, scale_w])
            ).astype(np.int16)
            gt_sparse['image_shape'] = self.image_size

        # ------------------------------------------------------------------
        # 3. Augmentation Phase 1: Color & Crop (Albumentations)
        # ------------------------------------------------------------------
        if self.color_tfs is not None:
            color_img = self.color_tfs(image=color_img)['image']

        crop_params = None
        
        if self.geometric_tfs is not None:
            # RGB, Depth, Segm 적용
            geo_aug_input = {
                'image': color_img,
                'depth': depth_img,
                'segm': segm_img,
            }
            geo_augmented = self.geometric_tfs(**geo_aug_input)

            color_img = geo_augmented['image']
            depth_img = geo_augmented['depth']
            segm_img = geo_augmented['segm']

            # XYZ와 Sparse GT를 위한 Crop 파라미터 추출
            replay = geo_augmented.get('replay', {})
            if 'transforms' in replay:
                for t in replay['transforms']:
                    class_name = t.get('__class_fullname__', '')
                    if 'Crop' in class_name and t.get('applied', False):
                        params = t.get('params', {})
                        crop_coords = params.get('crop_coords') # (x_min, y_min, x_max, y_max)
                        
                        if crop_coords is not None:
                            x_min, y_min, x_max, y_max = crop_coords
                            
                            # [Correction] XYZ Manual Slicing
                            xyz_dense = xyz_dense[y_min:y_max, x_min:x_max, :]

                            crop_params = {
                                'y_min': y_min, 'y_max': y_max,
                                'x_min': x_min, 'x_max': x_max,
                                'target_size': (color_img.shape[0], color_img.shape[1]) 
                            }
                            
                            # [Correction] XYZ Dense Map Resize (Resize동작 대응)
                            if (xyz_dense.shape[0] != color_img.shape[0]) or (xyz_dense.shape[1] != color_img.shape[1]):
                                xyz_dense = cv2.resize(xyz_dense, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        break

        # ------------------------------------------------------------------
        # 4. Flip Augmentation
        # ------------------------------------------------------------------
        flip_params = {'hflip': False, 'vflip': False}

        if self.hflip and np.random.rand() < 0.5:
            flip_params['hflip'] = True
            # Flip images horizontally
            color_img = np.flip(color_img, axis=1).copy()
            depth_img = np.flip(depth_img, axis=1).copy()
            segm_img = np.flip(segm_img, axis=1).copy()
            # xyz: (H, W, 3) - flip spatially and negate x component
            xyz_dense = np.flip(xyz_dense, axis=1).copy()
            xyz_dense[:, :, 0] = -xyz_dense[:, :, 0]  # x = -x

        if self.vflip and np.random.rand() < 0.5:
            flip_params['vflip'] = True
            # Flip images vertically
            color_img = np.flip(color_img, axis=0).copy()
            depth_img = np.flip(depth_img, axis=0).copy()
            segm_img = np.flip(segm_img, axis=0).copy()
            # xyz: (H, W, 3) - flip spatially and negate y component
            xyz_dense = np.flip(xyz_dense, axis=0).copy()
            xyz_dense[:, :, 1] = -xyz_dense[:, :, 1]  # y = -y

        # ------------------------------------------------------------------
        # 5. Sensor-style Depth Corruption
        # ------------------------------------------------------------------
        xyz_pre_corruption = xyz_dense
        depth_img, xyz_dense = self._apply_depth_corruption(depth_img, xyz_dense, segm_img)

        # ------------------------------------------------------------------
        # 6. To Tensor (No Padding here!)
        # ------------------------------------------------------------------
        color_tensor = torch.from_numpy(color_img).permute(2, 0, 1).float() / 255.0
        sample['pixel_values'] = TF.normalize(color_tensor, mean=self.image_mean, std=self.image_std)
        sample['og_color'] = color_tensor

        sample['depth'] = torch.from_numpy(depth_img).unsqueeze(0)
        sample['segm'] = torch.from_numpy(segm_img).unsqueeze(0).long()
        sample['xyz'] = torch.from_numpy(xyz_dense).permute(2, 0, 1).float()

        # Depth Mask 생성 (make_depth_values 대체 X, 단순히 마스크만 생성)
        depth_mask = sample['depth'] > 0.1
        sample['depth_mask'] = depth_mask.float()
        sample['depth'] = sample['depth'].masked_fill(~depth_mask, 0)
        
        # [Deleted] _make_depth_values call removed.
        # [Deleted] padding logic removed.
        
        # ------------------------------------------------------------------
        # 7. Apply Augmentation to Sparse GT (Crop + Flip)
        # ------------------------------------------------------------------
        aug_params = {}
        if crop_params is not None:
            aug_params['crop'] = crop_params
        aug_params['flip'] = flip_params
        aug_params['image_size'] = (color_img.shape[0], color_img.shape[1])  # (H, W)

        # Padding 정보는 없으므로 'pad_hw'는 None 또는 (0,0) 취급됨
        sample['aug_params'] = aug_params 

        # Sparse GT 텐서 변환
        for k, v in gt_sparse.items():
            if isinstance(v, np.ndarray):
                if k == 'valid_pixel_indices':
                    gt_sparse[k] = torch.from_numpy(v).long()
                elif k not in ['image_shape', 'top_k', 'top_p']:
                    gt_sparse[k] = torch.from_numpy(v).float()
                    # float16 양자화 오차 제거: score는 0.1 단위 이산값
                    if k == 'grasp_score':
                        gt_sparse[k] = (gt_sparse[k] * 10).round() / 10

        # [Final] Sparse GT에 Crop 적용 (Resize는 crop_params target_size로 처리됨)
        sample['gt_sparse'] = apply_augmentation_to_sparse(gt_sparse, aug_params)

        # Depth corruption 적용 시 grasp_depth GT 동시 변환 (realized-consistent 유지):
        # decode가 픽셀의 센서 3D점을 center로 복사하므로, 섭동 Δp = xyz_후 - xyz_전 의
        # approach-축 성분을 depth 타겟에서 빼야 실현 gripper가 라벨 pose에 남는다.
        # (gen_pixelwise_gt_rc.py와 동일 수식의 on-the-fly 버전; 맵/인덱스/방향 모두 최종 프레임)
        if xyz_dense is not xyz_pre_corruption:
            gt = sample['gt_sparse']
            idx = gt['valid_pixel_indices']
            if len(idx) > 0:
                dp = torch.from_numpy(
                    (xyz_dense - xyz_pre_corruption)[idx[:, 0].numpy(), idx[:, 1].numpy()]
                ).float()                                                    # (N, 3)
                corr = torch.einsum('nkc,nc->nk', gt['approach_dir'], dp)    # (N, K)
                gt['grasp_depth'] = (gt['grasp_depth'] - corr).clamp_(0.0, 0.04)
        sample['scene_id'] = self.scenename[index]
        sample['im_id'] = self.frameid[index]

        return sample

    # [Deleted] _make_depth_values method removed from Dataset


def sparse_to_dense(sparse_data, valid_pixel_indices, image_shape):
    """
    Sparse 데이터를 Dense 맵으로 변환합니다.
    """
    H, W = image_shape
    extra_dims = sparse_data.shape[1:] if sparse_data.ndim > 1 else ()

    dense_map = torch.zeros((H, W) + extra_dims, dtype=sparse_data.dtype, device=sparse_data.device)
    v_coords, u_coords = valid_pixel_indices[:, 0], valid_pixel_indices[:, 1]
    dense_map[v_coords, u_coords] = sparse_data

    return dense_map


def apply_augmentation_to_sparse(gt_sparse, aug_params):
    """
    Sparse GT 데이터에 crop, flip augmentation을 적용합니다.
    approach_dir, gripper_dir는 3D 방향 벡터이므로 flip 시 해당 축 성분을 반전해야 합니다.
    """
    if aug_params is None or not aug_params:
        return gt_sparse

    indices = gt_sparse['valid_pixel_indices'].clone()  # (N_valid, 2) [v, u]
    v_coords, u_coords = indices[:, 0], indices[:, 1]
    H_out, W_out = gt_sparse['image_shape']
    mask = torch.ones(len(indices), dtype=torch.bool)

    # Clone direction tensors for modification
    approach_dir = gt_sparse['approach_dir'].clone()  # (N_valid, K, 3)
    gripper_dir = gt_sparse['gripper_dir'].clone()    # (N_valid, K, 3)

    # 1. Crop 적용
    crop = aug_params.get('crop')
    if crop is not None:
        y_min, y_max = crop['y_min'], crop['y_max']
        x_min, x_max = crop['x_min'], crop['x_max']
        target_h, target_w = crop['target_size']

        # crop 영역 내의 픽셀만 선택
        mask = (v_coords >= y_min) & (v_coords < y_max) & (u_coords >= x_min) & (u_coords < x_max)

        # 새로운 좌표 계산 (crop 후 resize)
        crop_h, crop_w = y_max - y_min, x_max - x_min
        scale_h, scale_w = target_h / crop_h, target_w / crop_w

        v_coords = ((v_coords - y_min) * scale_h).long()
        u_coords = ((u_coords - x_min) * scale_w).long()

        H_out, W_out = target_h, target_w

    # 2. Flip 적용
    flip = aug_params.get('flip', {})
    image_size = aug_params.get('image_size', (H_out, W_out))
    H_img, W_img = image_size

    if flip.get('hflip', False):
        # Horizontal flip: u = W - 1 - u, x 방향 반전
        u_coords = W_img - 1 - u_coords
        # approach_dir, gripper_dir의 x 성분(index 0) 반전
        approach_dir[:, :, 0] = -approach_dir[:, :, 0]
        gripper_dir[:, :, 0] = -gripper_dir[:, :, 0]

    if flip.get('vflip', False):
        # Vertical flip: v = H - 1 - v, y 방향 반전
        v_coords = H_img - 1 - v_coords
        # approach_dir, gripper_dir의 y 성분(index 1) 반전
        approach_dir[:, :, 1] = -approach_dir[:, :, 1]
        gripper_dir[:, :, 1] = -gripper_dir[:, :, 1]

    # 3. Padding 적용 (Dataset에서는 안쓰지만 로직은 유지)
    pad_hw = aug_params.get('pad_hw')
    if pad_hw is not None:
        pad_h, pad_w = pad_hw
        H_out += pad_h
        W_out += pad_w

    # 최종 좌표 생성 (mask 적용)
    new_v = v_coords[mask].clamp(0, H_out - 1)
    new_u = u_coords[mask].clamp(0, W_out - 1)
    new_indices = torch.stack([new_v, new_u], dim=1)

    augmented_gt_sparse = {
        'valid_pixel_indices': new_indices,
        'image_shape': (H_out, W_out),
        'top_k': gt_sparse['top_k'],
        'graspness': gt_sparse['graspness'][mask],
        'grasp_score': gt_sparse['grasp_score'][mask],
        'grasp_width': gt_sparse['grasp_width'][mask],
        'grasp_depth': gt_sparse['grasp_depth'][mask],
        'approach_dir': approach_dir[mask],
        'gripper_dir': gripper_dir[mask],
    }

    return augmented_gt_sparse


def apply_augmentation_to_tensor(tensor, aug_params):
    """
    Dense 텐서에 crop (및 padding) augmentation을 적용합니다.
    """
    if aug_params is None or not aug_params:
        return tensor

    result = tensor

    # 1. Crop 적용
    crop = aug_params.get('crop')
    if crop is not None:
        y_min, y_max = crop['y_min'], crop['y_max']
        x_min, x_max = crop['x_min'], crop['x_max']
        target_h, target_w = crop['target_size']

        result = result[:, y_min:y_max, x_min:x_max]
        result = F.interpolate(
            result.unsqueeze(0), size=(target_h, target_w), mode='nearest'
        ).squeeze(0)

    # 2. Padding 적용 (Dataset에서는 안쓰지만 로직 유지)
    pad_hw = aug_params.get('pad_hw')
    if pad_hw is not None:
        pad_h, pad_w = pad_hw
        if pad_h > 0 or pad_w > 0:
            result = F.pad(result, (0, pad_w, 0, pad_h))

    return result


def graspnet_collate_fn(batch):
    """
    Collated batch, but without Padding! 
    Padding should be handled inside Model or by a wrapping BatchProcessor if needed.
    """
    collated = {}
    # pixel_values, depth, etc. are stacked here.
    # Assumes all images in a batch have the same size (after RandomSizedCrop resize).
    stackable_keys = ['pixel_values', 'og_color', 'depth', 'segm', 'depth_mask', 'xyz']
    # removed 'depth_values' as it is created in model now

    list_keys = ['gt_sparse', 'aug_params', 'crop_params', 'scene_id', 'im_id'] 
    # removed 'pad_hw'

    for key in batch[0].keys():
        values = [sample[key] for sample in batch]

        if key in stackable_keys:
            if values[0] is not None:
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = None
        elif key in list_keys:
            collated[key] = values
        else:
            try:
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    collated[key] = values
            except (RuntimeError, TypeError):
                collated[key] = values
    return collated


def visualize_sample(sample, gt_sparse, save_dir, prefix=""):
    """Helper function to visualize a sample with dense maps."""
    H, W = gt_sparse['image_shape']
    K = gt_sparse['top_k'].item() if isinstance(gt_sparse['top_k'], torch.Tensor) else gt_sparse['top_k']

    view_idx = 0  # Use best view for visualization

    # Convert to dense (for visualization)
    graspness_dense = sparse_to_dense(
        gt_sparse['graspness'],
        gt_sparse['valid_pixel_indices'],
        (H, W)
    )
    grasp_score_dense = sparse_to_dense(
        gt_sparse['grasp_score'][:, view_idx],
        gt_sparse['valid_pixel_indices'],
        (H, W)
    )
    grasp_width_dense = sparse_to_dense(
        gt_sparse['grasp_width'][:, view_idx],
        gt_sparse['valid_pixel_indices'],
        (H, W)
    )
    grasp_depth_dense = sparse_to_dense(
        gt_sparse['grasp_depth'][:, view_idx],
        gt_sparse['valid_pixel_indices'],
        (H, W)
    )
    approach_dir_dense = sparse_to_dense(
        gt_sparse['approach_dir'][:, view_idx, :],
        gt_sparse['valid_pixel_indices'],
        (H, W)
    )
    gripper_dir_dense = sparse_to_dense(
        gt_sparse['gripper_dir'][:, view_idx, :],
        gt_sparse['valid_pixel_indices'],
        (H, W)
    )

    vis_sample = {
        'og_color': sample['og_color'],
        'depth': sample['depth'],
        'xyz': sample['xyz'],
        'graspness': graspness_dense.unsqueeze(0),
        'grasp_score': grasp_score_dense.unsqueeze(0),
        'grasp_width': grasp_width_dense.unsqueeze(0),
        'grasp_depth': grasp_depth_dense.unsqueeze(0),
        'approach_dir': approach_dir_dense.permute(2, 0, 1),
        'gripper_dir': gripper_dir_dense.permute(2, 0, 1),
        'scene_id': sample.get('scene_id', 'unknown'),
        'im_id': sample.get('im_id', 0),
    }

    vis_results = visualize_and_save_opencv(
        vis_sample,
        save_dir=save_dir,
        prefix=prefix
    )
    return vis_results


if __name__ == '__main__':
    import os
    os.makedirs('./figures', exist_ok=True)

    # ================================================================
    # 1. Original (No Augmentation) 시각화
    # ================================================================
    print("=" * 60)
    print("1. Loading dataset WITHOUT flip augmentation...")
    print("=" * 60)

    dataset_no_flip = GraspNet1BDataset(
        root='/data/Grasp/GraspNet-1Billion',
        camera='realsense',
        split='debug',
        augmentation='None'  # No augmentation
    )

    sample_original = dataset_no_flip[0]

    print("\n=== Original Sample (No Augmentation) ===")
    print(f"Aug params: {sample_original['aug_params']}")

    # Visualize original
    visualize_sample(
        sample_original,
        sample_original['gt_sparse'],
        save_dir='./figures',
        prefix='original_'
    )
    print("Saved original visualization to ./figures/original_*")

    # ================================================================
    # 2. HFlip Augmentation 시각화
    # ================================================================
    print("\n" + "=" * 60)
    print("2. Loading dataset WITH hflip augmentation...")
    print("=" * 60)

    # Force hflip by setting random seed
    np.random.seed(42)

    dataset_hflip = GraspNet1BDataset(
        root='/data/Grasp/GraspNet-1Billion',
        camera='realsense',
        split='debug',
        augmentation='hflip'
    )

    # Get sample with hflip applied (may need multiple tries)
    for _ in range(10):
        np.random.seed(0)  # Force flip to happen
        sample_hflip = dataset_hflip[0]
        if sample_hflip['aug_params']['flip']['hflip']:
            break

    print(f"\n=== HFlip Sample ===")
    print(f"Aug params: {sample_hflip['aug_params']}")

    visualize_sample(
        sample_hflip,
        sample_hflip['gt_sparse'],
        save_dir='./figures',
        prefix='hflip_'
    )
    print("Saved hflip visualization to ./figures/hflip_*")

    # ================================================================
    # 3. VFlip Augmentation 시각화
    # ================================================================
    print("\n" + "=" * 60)
    print("3. Loading dataset WITH vflip augmentation...")
    print("=" * 60)

    dataset_vflip = GraspNet1BDataset(
        root='/data/Grasp/GraspNet-1Billion',
        camera='realsense',
        split='debug',
        augmentation='vflip'
    )

    for _ in range(10):
        np.random.seed(0)  # Force flip to happen
        sample_vflip = dataset_vflip[0]
        if sample_vflip['aug_params']['flip']['vflip']:
            break

    print(f"\n=== VFlip Sample ===")
    print(f"Aug params: {sample_vflip['aug_params']}")

    visualize_sample(
        sample_vflip,
        sample_vflip['gt_sparse'],
        save_dir='./figures',
        prefix='vflip_'
    )
    print("Saved vflip visualization to ./figures/vflip_*")

    # ================================================================
    # 4. HFlip + VFlip Augmentation 시각화
    # ================================================================
    print("\n" + "=" * 60)
    print("4. Loading dataset WITH hflip+vflip augmentation...")
    print("=" * 60)

    dataset_hvflip = GraspNet1BDataset(
        root='/data/Grasp/GraspNet-1Billion',
        camera='realsense',
        split='debug',
        augmentation='hflip,vflip'
    )

    for _ in range(10):
        np.random.seed(0)  # Force flip to happen
        sample_hvflip = dataset_hvflip[0]
        if sample_hvflip['aug_params']['flip']['hflip'] and sample_hvflip['aug_params']['flip']['vflip']:
            break

    print(f"\n=== HFlip + VFlip Sample ===")
    print(f"Aug params: {sample_hvflip['aug_params']}")

    visualize_sample(
        sample_hvflip,
        sample_hvflip['gt_sparse'],
        save_dir='./figures',
        prefix='hvflip_'
    )
    print("Saved hvflip visualization to ./figures/hvflip_*")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Visualization Summary")
    print("=" * 60)
    print("Saved visualizations:")
    print("  - ./figures/original_*  : No augmentation")
    print("  - ./figures/hflip_*     : Horizontal flip")
    print("  - ./figures/vflip_*     : Vertical flip")
    print("  - ./figures/hvflip_*    : Both flips")
    print("\nCheck approach_dir and gripper_dir images to verify:")
    print("  - HFlip: x component (R channel) should be inverted")
    print("  - VFlip: y component (G channel) should be inverted")
    print("  - xyz: spatial positions and corresponding axis values should be mirrored")
