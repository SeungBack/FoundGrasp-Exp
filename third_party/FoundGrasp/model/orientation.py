import math

import torch
import torch.nn.functional as F


def generate_grasp_views(num_views=300, device=None, dtype=torch.float32):
    """Fibonacci-sampled GraspNet view directions."""
    idx = torch.arange(num_views, device=device, dtype=dtype)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    z = (2.0 * idx + 1.0) / float(num_views) - 1.0
    r = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    x = r * torch.cos(2.0 * idx * math.pi * phi)
    y = r * torch.sin(2.0 * idx * math.pi * phi)
    return torch.stack([x, y, z], dim=-1)


def batch_viewpoint_params_to_matrix(towards, angles):
    """Torch version of GraspNet's viewpoint+angle to rotation conversion."""
    axis_x = F.normalize(towards, dim=-1, eps=1e-6)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    singular = axis_y.norm(dim=-1) <= 1e-6
    if singular.any():
        axis_y = axis_y.clone()
        axis_y[singular] = axis_y.new_tensor([0.0, 1.0, 0.0])
    axis_y = F.normalize(axis_y, dim=-1, eps=1e-6)
    axis_z = torch.linalg.cross(axis_x, axis_y, dim=-1)

    ones = torch.ones_like(angles)
    zeros = torch.zeros_like(angles)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    r1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
    r1 = r1.reshape(-1, 3, 3)
    r2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    return torch.matmul(r2, r1)


def get_view_angle_tables(num_views=300, num_angles=12, device=None, dtype=torch.float32):
    """Return approach templates, gripper templates, and rotations for discrete bins."""
    views = generate_grasp_views(num_views, device=device, dtype=dtype)
    approach_templates = -views
    angles = torch.arange(num_angles, device=device, dtype=dtype) * (math.pi / float(num_angles))

    towards = approach_templates[:, None, :].expand(num_views, num_angles, 3).reshape(-1, 3)
    angles_flat = angles[None, :].expand(num_views, num_angles).reshape(-1)
    rotations = batch_viewpoint_params_to_matrix(towards, angles_flat)
    rotations = rotations.reshape(num_views, num_angles, 3, 3)
    gripper_templates = rotations[:, :, :, 1]
    return approach_templates, gripper_templates, rotations


def quantize_joint_orientation_vectors(
    approach_dir,
    gripper_dir,
    num_views=300,
    num_angles=12,
    symmetric_gripper=True,
):
    """Quantize direction vectors to joint view-angle class ids."""
    device, dtype = approach_dir.device, approach_dir.dtype
    approach_templates, gripper_templates, _ = get_view_angle_tables(
        num_views, num_angles, device=device, dtype=dtype
    )

    app = F.normalize(approach_dir.reshape(-1, 3), dim=-1, eps=1e-6)
    grip = F.normalize(gripper_dir.reshape(-1, 3), dim=-1, eps=1e-6)

    view_scores = app @ approach_templates.t()
    view_idx = torch.argmax(view_scores, dim=-1)

    selected_grippers = gripper_templates[view_idx]  # (BHW, A, 3)
    angle_scores = torch.sum(selected_grippers * grip[:, None, :], dim=-1)
    if symmetric_gripper:
        angle_scores = angle_scores.abs()
    angle_idx = torch.argmax(angle_scores, dim=-1)
    return view_idx * int(num_angles) + angle_idx


def decode_joint_orientation_classes(class_idx, num_views=300, num_angles=12):
    """Decode joint view-angle class ids to approach and gripper directions."""
    device = class_idx.device
    approach_templates, gripper_templates, _ = get_view_angle_tables(
        num_views, num_angles, device=device, dtype=torch.float32
    )
    flat_idx = class_idx.reshape(-1).long()
    view_idx = torch.div(flat_idx, int(num_angles), rounding_mode='floor')
    angle_idx = flat_idx % int(num_angles)
    approach = approach_templates[view_idx].to(dtype=torch.float32)
    gripper = gripper_templates[view_idx, angle_idx].to(dtype=torch.float32)
    return approach.reshape(*class_idx.shape, 3), gripper.reshape(*class_idx.shape, 3)


def decode_joint_orientation_logits(logits, num_views=300, num_angles=12):
    """Decode dense joint orientation logits to direction maps."""
    class_idx = torch.argmax(logits, dim=1)
    approach, gripper = decode_joint_orientation_classes(class_idx, num_views, num_angles)
    return approach.permute(0, 3, 1, 2), gripper.permute(0, 3, 1, 2)
