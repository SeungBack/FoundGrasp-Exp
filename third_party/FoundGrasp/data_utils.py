import numpy as np
import torch



def generate_grasp_views(N=300, phi=(np.sqrt(5)-1)/2, center=np.zeros(3), r=1):
    """ View sampling on a unit sphere using Fibonacci lattices.
        Ref: https://arxiv.org/abs/0912.4540

        Input:
            N: [int]
                number of sampled views
            phi: [float]
                constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
            center: [np.ndarray, (3,), np.float32]
                sphere center
            r: [float]
                sphere radius

        Output:
            views: [torch.FloatTensor, (N,3)]
                sampled view coordinates
    """
    views = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    return torch.from_numpy(views.astype(np.float32))

def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """ Transform approach vectors and in-plane rotation angles to rotation matrices.

        Input:
            batch_towards: [torch.FloatTensor, (N,3)]
                approach vectors in batch
            batch_angle: [torch.floatTensor, (N,)]
                in-plane rotation angles in batch
                
        Output:
            batch_matrix: [torch.floatTensor, (N,3,3)]
                rotation matrices in batch
    """
    if isinstance(batch_towards, torch.Tensor):
        axis_x = batch_towards
        ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
        zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
        axis_y = torch.stack([-axis_x[:,1], axis_x[:,0], zeros], dim=-1)
        mask_y = (torch.norm(axis_y, dim=-1) == 0)
        axis_y[mask_y,1] = 1
        axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
        axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
        axis_z = torch.cross(axis_x, axis_y)
        sin = torch.sin(batch_angle)
        cos = torch.cos(batch_angle)
        R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
        R1 = R1.reshape([-1,3,3])
        R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
        return torch.matmul(R2, R1)
    else:
        axis_x = batch_towards
        ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
        zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
        axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
        mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)
        axis_y[mask_y] = np.array([0, 1, 0])
        axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
        axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
        axis_z = np.cross(axis_x, axis_y)
        sin = np.sin(batch_angle)
        cos = np.cos(batch_angle)
        R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
        R1 = R1.reshape([-1,3,3])
        R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
        matrix = np.matmul(R2, R1)
        return matrix.astype(np.float32)


class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale




def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale  #
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if isinstance(cloud, np.ndarray):
        if format == '3x3':
            cloud_transformed = np.dot(transform, cloud.T).T
        elif format == '4x4' or format == '3x4':
            ones = np.ones(cloud.shape[0])[:, np.newaxis]
            cloud_ = np.concatenate([cloud, ones], axis=1)
            cloud_transformed = np.dot(transform, cloud_.T).T
            cloud_transformed = cloud_transformed[:, :3]
    elif isinstance(cloud, torch.Tensor):
        if format == '3x3':
            cloud_transformed = torch.matmul(transform, cloud.T).T
        elif format == '4x4' or format == '3x4':
            ones = cloud.new_ones(cloud.size(0), device=cloud.device).unsqueeze(-1)
            cloud_ = torch.cat([cloud, ones], dim=1)
            cloud_transformed = torch.matmul(transform, cloud_.T).T
            cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    if isinstance(A, np.ndarray):
        A = A[:, np.newaxis, :]
        B = B[np.newaxis, :, :]
        dists = np.linalg.norm(A - B, axis=-1)
    elif isinstance(A, torch.Tensor):
        A = A.unsqueeze(1)
        B = B.unsqueeze(0)
        dists = torch.norm(A - B, dim=-1)
    return dists


def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """
    Compute mutual visibility between grasp points and scene cloud.

    Input:
        cloud: (N, 3) np.ndarray or torch.Tensor
            Scene point cloud (camera coords)
        grasp_points: (M, 3)
            Object grasp points (object coords)
        pose: (4,4) or (3,4)
            Object → camera transform
        th: float
            Threshold in meters for mutual visibility

    Output:
        visible_mask_grasp: (M,)
            True if grasp point matches visible scene within th
        visible_mask_cloud: (N,)
            True if scene point lies close to transformed object within th
    """
    # change cloud, grap_points, pose to torch.Tensor if needed
    if isinstance(cloud, np.ndarray):
        cloud = torch.from_numpy(cloud).float().cuda()
    if isinstance(grasp_points, np.ndarray):
        grasp_points = torch.from_numpy(grasp_points).float().cuda()
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose).float().cuda()

    grasp_points_cam = transform_point_cloud(grasp_points, pose)  # (M,3)
    dists = compute_point_dists(grasp_points_cam, cloud)          # (M,N)
    min_dists_grasp = dists.min(dim=1)[0]  # for each grasp point
    min_dists_cloud = dists.min(dim=0)[0]  # for each scene point
    visible_mask_grasp = (min_dists_grasp < th)
    visible_mask_cloud = (min_dists_cloud < th)
    visible_mask_grasp = visible_mask_grasp.cpu().numpy()
    visible_mask_cloud = visible_mask_cloud.cpu().numpy()

    return visible_mask_grasp, visible_mask_cloud


def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.
        Workspace here refers to the space with objects.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg > 0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
    mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
    mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask