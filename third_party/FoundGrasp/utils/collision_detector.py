import os
import sys
import numpy as np
import open3d as o3d
import torch




class ModelFreeCollisionDetectorGPU():

    def __init__(self, scene_points, voxel_size=0.005, device='cuda'):
        """ Collision detection in scenes without object labels using GPU acceleration.
            scene_points: [torch.Tensor, (N,3), torch.float32]
                the scene points to detect
            voxel_size: [float]
                used for downsample
            device: [str]
        """
        self.voxel_size = voxel_size
        self.device = torch.device(device) # Use torch.device object

        assert isinstance(scene_points, torch.Tensor), "scene_points must be a torch.Tensor"
        
        self.scene_points = self.voxel_down_sample(scene_points, voxel_size)
        self.num_points = self.scene_points.shape[0]

    def voxel_down_sample(self, points, voxel_size):
        """ Voxel downsample a point cloud using PyTorch.

            Input:
                points: [torch.Tensor, (N,3), torch.float32]
                    the input point cloud
                voxel_size: [float]
                    the voxel size for downsampling

            Output:
                downsampled_points: [torch.Tensor, (M,3), torch.float32]
                    the downsampled point cloud
        """
        if points.shape[0] == 0:
            return points
        voxel_coords = torch.floor(points / voxel_size).long()
        unique_voxel_coords, inverse_indices = torch.unique(
        voxel_coords, dim=0, return_inverse=True
        )
        num_unique_voxels = unique_voxel_coords.shape[0]
        summed_points = torch.zeros(num_unique_voxels, 3, device=self.device)
        summed_points.scatter_add_(0, inverse_indices.unsqueeze(-1).expand(-1, 3), points)
        counts = torch.bincount(inverse_indices)
        downsampled_points = summed_points / (counts.unsqueeze(-1) + 1e-6)
        return downsampled_points
        
    def detect_soft(self, gg_array, approach_dist=0.03, collision_thresh=0.05,
                    finger_width=0.01, finger_length=0.06,
                    return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """GPU collision detection — exact same logic as ModelFreeCollisionDetector (CPU)."""
        approach_dist = max(approach_dist, finger_width)

        T = gg_array[:, 13:16]                        # (N, 3)
        R = gg_array[:, 4:13].reshape(-1, 3, 3)       # (N, 3, 3)
        heights = gg_array[:, 2].unsqueeze(1)          # (N, 1)
        depths  = gg_array[:, 3].unsqueeze(1)          # (N, 1)
        widths  = gg_array[:, 1].unsqueeze(1)          # (N, 1)

        if T.shape[0] == 0:
            ret_value = [np.array([], dtype=bool)]
            if return_empty_grasp:
                ret_value.append(np.array([], dtype=bool))
            if return_ious:
                ret_value.append([np.array([], dtype=np.float32)] * 5)
            return ret_value if len(ret_value) > 1 else ret_value[0]

        targets = self.scene_points.unsqueeze(0) - T.unsqueeze(1)   # (N, P, 3)
        targets = torch.einsum('bpi,bij->bpj', targets, R)           # (N, P, 3)

        # mirrors CPU mask1–mask8 exactly
        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        mask2 = (targets[:, :, 0] > depths - finger_length) & (targets[:, :, 0] < depths)
        mask3 = (targets[:, :, 1] > -(widths / 2 + finger_width))
        mask4 = (targets[:, :, 1] < -widths / 2)
        mask5 = (targets[:, :, 1] < (widths / 2 + finger_width))
        mask6 = (targets[:, :, 1] > widths / 2)
        mask7 = (targets[:, :, 0] <= depths - finger_length) & \
                (targets[:, :, 0] >  depths - finger_length - finger_width)
        mask8 = (targets[:, :, 0] <= depths - finger_length - finger_width) & \
                (targets[:, :, 0] >  depths - finger_length - finger_width - approach_dist)

        left_mask     = mask1 & mask2 & mask3 & mask4
        right_mask    = mask1 & mask2 & mask5 & mask6
        bottom_mask   = mask1 & mask3 & mask5 & mask7
        shifting_mask = mask1 & mask3 & mask5 & mask8
        global_mask   = left_mask | right_mask | bottom_mask | shifting_mask

        left_right_volume = (heights * finger_length * finger_width / self.voxel_size ** 3).squeeze(-1)
        bottom_volume     = (heights * (widths + 2 * finger_width) * finger_width / self.voxel_size ** 3).squeeze(-1)
        shifting_volume   = (heights * (widths + 2 * finger_width) * approach_dist / self.voxel_size ** 3).squeeze(-1)
        volume = left_right_volume * 2 + bottom_volume + shifting_volume

        global_iou = global_mask.float().sum(dim=1) / (volume + 1e-6)
        collision_mask = (global_iou > collision_thresh).cpu().numpy()

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask]
        if return_empty_grasp:
            inner_mask   = mask1 & mask2 & (~mask4) & (~mask6)
            inner_volume = (heights * finger_length * widths / self.voxel_size ** 3).squeeze(-1)
            empty_mask   = (inner_mask.float().sum(dim=1) / (inner_volume + 1e-6) < empty_thresh).cpu().numpy()
            ret_value.append(empty_mask)
        if return_ious:
            left_iou     = left_mask.float().sum(dim=1)     / (left_right_volume + 1e-6)
            right_iou    = right_mask.float().sum(dim=1)    / (left_right_volume + 1e-6)
            bottom_iou   = bottom_mask.float().sum(dim=1)   / (bottom_volume     + 1e-6)
            shifting_iou = shifting_mask.float().sum(dim=1) / (shifting_volume   + 1e-6)
            ret_value.append([
                global_iou.cpu().numpy(),
                left_iou.cpu().numpy(),
                right_iou.cpu().numpy(),
                bottom_iou.cpu().numpy(),
                shifting_iou.cpu().numpy(),
            ])
        return ret_value

    def detect(self, gg_array, approach_dist=0.05,
               min_inner_points=10, finger_length=0.04, finger_width=0.01,
               height=0.02):
        """HGGD-style hard binary collision detection.

        A grasp is kept only when ALL three hold:
          1. inner region (between fingers) has > min_inner_points
          2. finger regions have 0 points
          3. combined bottom+approach corridor has 0 points

        Returns:
            no_collision_mask: np.ndarray (N,) bool, True = keep
        """
        T = gg_array[:, 13:16]                       # (N, 3)
        R = gg_array[:, 4:13].reshape(-1, 3, 3)      # (N, 3, 3)
        widths = gg_array[:, 1].unsqueeze(1)          # (N, 1) per-grasp width
        depths = gg_array[:, 3].unsqueeze(1)          # (N, 1) per-grasp depth

        if T.shape[0] == 0:
            return np.array([], dtype=bool)

        targets = self.scene_points.unsqueeze(0) - T.unsqueeze(1)
        targets = torch.einsum('bpi,bij->bpj', targets, R)

        tx, ty, tz = targets[:, :, 0], targets[:, :, 1], targets[:, :, 2]

        half_w = widths / 2
        mask1 = (tz > -height / 2) & (tz < height / 2)
        mask2 = (tx > depths - finger_length) & (tx < depths)
        mask3 = (ty > -(half_w + finger_width))
        mask4 = (ty < -half_w)
        mask5 = (ty < (half_w + finger_width))
        mask6 = (ty > half_w)
        # combined bottom+approach: x in (depth - fl - fw - approach_dist, depth - fl]
        mask7 = (tx <= depths - finger_length) & (tx > depths - finger_length - finger_width - approach_dist)

        depth_mask   = mask1 & mask2
        left_mask    = depth_mask & mask3 & mask4
        right_mask   = depth_mask & mask5 & mask6
        shifting_mask = mask1 & mask3 & mask5 & mask7

        has_object        = (depth_mask & (~mask4) & (~mask6)).sum(1) > min_inner_points
        no_finger_collision = (left_mask | right_mask).sum(1) == 0
        approach_clear    = shifting_mask.sum(1) == 0

        no_collision = has_object & no_finger_collision & approach_clear
        return no_collision.cpu().numpy()


class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """

    def __init__(self, scene_points, voxel_size=0.005):
        self.finger_width = 0.01
        self.finger_length = 0.06
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False,
               empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious
                    
            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:, np.newaxis]
        depths = grasp_group.depths[:, np.newaxis]
        widths = grasp_group.widths[:, np.newaxis]
        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2))
        # left finger mask
        mask2 = ((targets[:, :, 0] > depths - self.finger_length) & (targets[:, :, 0] < depths))
        mask3 = (targets[:, :, 1] > -(widths / 2 + self.finger_width))
        mask4 = (targets[:, :, 1] < -widths / 2)
        # right finger mask
        mask5 = (targets[:, :, 1] < (widths / 2 + self.finger_width))
        mask6 = (targets[:, :, 1] > widths / 2)
        # bottom mask
        mask7 = ((targets[:, :, 0] <= depths - self.finger_length) \
                 & (targets[:, :, 0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:, :, 0] <= depths - self.finger_length - self.finger_width) \
                 & (targets[:, :, 0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size ** 3)).reshape(-1)
        bottom_volume = (
                    heights * (widths + 2 * self.finger_width) * self.finger_width / (self.voxel_size ** 3)).reshape(-1)
        shifting_volume = (heights * (widths + 2 * self.finger_width) * approach_dist / (self.voxel_size ** 3)).reshape(
            -1)
        volume = left_right_volume * 2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume + 1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask, ]
        if return_empty_grasp:
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size ** 3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1) / inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume + 1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume + 1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume + 1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume + 1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        return ret_value
