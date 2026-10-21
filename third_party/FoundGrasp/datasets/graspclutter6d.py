import os
import json
import warnings

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm

from datasets.graspnet1b import _resolve_image_stats


CAMERA_OFFSET = {
    'realsense-d415': 1,
    'realsense-d435': 2,
    'azure-kinect': 3,
    'zivid': 4,
}

DEPTH_SCALE = {
    'realsense-d415': 1000.0,
    'realsense-d435': 1000.0,
    'azure-kinect': 10000.0,
    'zivid': 10000.0,
}

CAMERA_ALIASES = {
    'realsense': 'realsense-d435',
    'kinect': 'azure-kinect',
}


class GraspClutter6DDataset(Dataset):
    """Inference-only dataset for GraspClutter6D.

    Returns the same dict format as GraspNet1BDataset
    (pixel_values, depth, xyz, depth_mask, scene_id, im_id)
    so it can be dropped into the existing test.py inference loop.

    No gt_sparse is produced — use GraspClutter6DEval.eval_all for AP metrics.
    """

    def __init__(self, root, camera='realsense-d435', split='test',
                 image_size=(360, 720), backbone_name='facebook/dinov2-base',
                 workspace_crop=True, workspace_pad=0.1, depth_root=None):
        camera = CAMERA_ALIASES.get(camera, camera)
        assert camera in CAMERA_OFFSET, f'Unknown camera: {camera}'
        assert split in ('train', 'test', 'all'), 'split must be train/test/all'

        self.root = root
        self.camera = camera
        self.image_size = image_size  # (H, W)
        self.depth_scale = DEPTH_SCALE[camera]
        self.cam_offset = CAMERA_OFFSET[camera]
        self.workspace_crop = workspace_crop
        self.workspace_pad = workspace_pad  # fraction of min(H, W)

        # Load scene IDs from split files
        if split == 'all':
            scene_ids = []
            for fname in ('grasp_train_scene_ids.json', 'grasp_test_scene_ids.json'):
                with open(os.path.join(root, 'split_info', fname)) as f:
                    scene_ids += [int(x) for x in json.load(f)]
        else:
            fname = f'grasp_{split}_scene_ids.json'
            with open(os.path.join(root, 'split_info', fname)) as f:
                scene_ids = [int(x) for x in json.load(f)]

        self.rgb_paths = []
        self.depth_paths = []
        self.segm_paths = []
        self.scene_ids = []   # int
        self.img_ids = []     # int  (actual file number, used for dump path)
        self._cam_K = {}      # (scene_id_int, img_id_int) → (3, 3) ndarray

        scenes_dir = os.path.join(root, 'scenes')
        for scene_id in tqdm(scene_ids, desc='Loading GC6D data paths...'):
            cam_json_path = os.path.join(scenes_dir, f'{scene_id:06d}', 'scene_camera.json')
            with open(cam_json_path) as f:
                cam_data = json.load(f)
            for ann_id in range(13):
                img_id = ann_id * 4 + self.cam_offset
                self.rgb_paths.append(
                    os.path.join(scenes_dir, f'{scene_id:06d}', 'rgb', f'{img_id:06d}.png'))
                self.depth_paths.append(
                    os.path.join(depth_root or root, 'scenes', f'{scene_id:06d}', 'depth', f'{img_id:06d}.png'))
                self.segm_paths.append(
                    os.path.join(scenes_dir, f'{scene_id:06d}', 'label', f'{img_id:06d}.png'))
                self.scene_ids.append(scene_id)
                self.img_ids.append(img_id)
                self._cam_K[(scene_id, img_id)] = np.array(
                    cam_data[str(img_id)]['cam_K'], dtype=np.float64).reshape(3, 3)

        self.image_mean, self.image_std = _resolve_image_stats(backbone_name)

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        scene_id_int = self.scene_ids[index]
        img_id = self.img_ids[index]

        # Load images
        color_img = cv2.cvtColor(
            cv2.imread(self.rgb_paths[index]), cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(
            self.depth_paths[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / self.depth_scale

        orig_H, orig_W = depth_img.shape[:2]
        K = self._cam_K[(scene_id_int, img_id)].copy()

        if self.workspace_crop:
            # Segmentation mask-based workspace bbox (seg > 0 = foreground objects)
            segm_img = cv2.imread(self.segm_paths[index], cv2.IMREAD_UNCHANGED)[:, :, 0]
            valid = segm_img > 0
            rows = np.any(valid, axis=1)
            cols = np.any(valid, axis=0)
            if rows.any() and cols.any():
                y0 = int(np.argmax(rows))
                y1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
                x0 = int(np.argmax(cols))
                x1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
                pad = int(min(orig_H, orig_W) * self.workspace_pad)
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(orig_W - 1, x1 + pad)
                y1 = min(orig_H - 1, y1 + pad)
                color_img = color_img[y0:y1+1, x0:x1+1]
                depth_img = depth_img[y0:y1+1, x0:x1+1]
                K[0, 2] -= x0  # cx
                K[1, 2] -= y0  # cy
                crop_H, crop_W = depth_img.shape[:2]
            else:
                crop_H, crop_W = orig_H, orig_W
        else:
            crop_H, crop_W = orig_H, orig_W

        # Resize to target image_size and scale intrinsics accordingly
        tgt_H, tgt_W = self.image_size
        if (crop_H, crop_W) != (tgt_H, tgt_W):
            color_img = cv2.resize(color_img, (tgt_W, tgt_H), interpolation=cv2.INTER_LINEAR)
            depth_img = cv2.resize(depth_img, (tgt_W, tgt_H), interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite('/home/seung/test.png', color_img)
        K[0] *= tgt_W / crop_W  # fx, cx
        K[1] *= tgt_H / crop_H  # fy, cy

        # Back-project depth → XYZ
        H, W = depth_img.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        z = depth_img
        x = (us - cx) / fx * z
        y = (vs - cy) / fy * z
        xyz = np.stack([x, y, z], axis=-1)  # (H, W, 3)

        # To tensors
        color_tensor = torch.from_numpy(color_img).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0)
        xyz_tensor = torch.from_numpy(xyz).permute(2, 0, 1).float()

        depth_mask = (depth_tensor > 0.1).float()
        depth_tensor = depth_tensor.masked_fill(depth_mask == 0, 0.0)

        return {
            'pixel_values': TF.normalize(color_tensor, mean=self.image_mean, std=self.image_std),
            'og_color': color_tensor,
            'depth': depth_tensor,
            'xyz': xyz_tensor,
            'depth_mask': depth_mask,
            'scene_id': f'{scene_id_int:06d}',
            'im_id': img_id,
        }
