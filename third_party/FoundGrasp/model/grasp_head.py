import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AddFusion, ConcatFusion


def _get_num_groups(channels, max_groups=8):
    for groups in [max_groups, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


def build_branch_module(in_channels, hidden_channels, out_channels,
                        activation='none', output_layer=False):
    layers = []
    prev_channels = in_channels

    for hidden_ch in hidden_channels:
        layers.extend([
            nn.Conv2d(prev_channels, hidden_ch, kernel_size=3, padding=1),
            nn.GroupNorm(_get_num_groups(hidden_ch), hidden_ch),
            nn.GELU(),
        ])
        prev_channels = hidden_ch

    if output_layer:
        layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=1))
        if activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'gelu':
            layers.append(nn.GELU())

    return nn.Sequential(*layers)


def _decode_grasp_outputs(graspness_raw, grasp_score_raw, approach_raw, gripper_raw,
                          width_raw, depth_raw, use_kappa):
    graspness = torch.sigmoid(graspness_raw)
    grasp_score = torch.sigmoid(grasp_score_raw)

    if use_kappa:
        approach_out, approach_kappa_raw = approach_raw[:, :3], approach_raw[:, 3:]
        gripper_out, gripper_kappa_raw = gripper_raw[:, :3], gripper_raw[:, 3:]
    else:
        approach_out, gripper_out = approach_raw, gripper_raw
        approach_kappa_raw = None
        gripper_kappa_raw = None

    pred_appr = F.normalize(approach_out, dim=1)
    dot = torch.sum(gripper_out * pred_appr, dim=1, keepdim=True)
    pred_grip = F.normalize(gripper_out - dot * pred_appr, p=2, dim=1)

    outputs = {
        'graspness': graspness,
        'grasp_score': grasp_score,
        'approach_dir': pred_appr,
        'gripper_dir': pred_grip,
        'grasp_width': torch.sigmoid(width_raw),
        'grasp_depth': torch.sigmoid(depth_raw),
    }
    if use_kappa:
        outputs['approach_kappa'] = F.elu(approach_kappa_raw) + 1.0
        outputs['gripper_kappa'] = F.elu(gripper_kappa_raw) + 1.0
    return outputs



class ParallelGraspHead(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 head_dim=None,
                 num_trunk_layers=1,
                 num_hidden_layers=2,
                 num_head_layers=1,
                 rgbd_fusion_type='concat',
                 output_size=None,
                 use_kappa=False,
                 print_param_counts=True,
                 **kwargs,
                 ):
        super().__init__()

        if num_trunk_layers < 1:
            raise ValueError(f"num_trunk_layers must be >= 1, got {num_trunk_layers}")

        self.feature_dim = feature_dim
        self.head_dim = head_dim or feature_dim
        self.output_size = output_size
        self.use_kappa = use_kappa

        self.rgbd_fusion = self._get_fusion_module(rgbd_fusion_type, 2 * feature_dim, feature_dim)
        self.trunk = build_branch_module(
            in_channels=feature_dim,
            hidden_channels=[self.head_dim] * num_trunk_layers,
            out_channels=self.head_dim,
            output_layer=False,
        )
        self.grasp_branch = build_branch_module(
            in_channels=self.head_dim,
            hidden_channels=[self.head_dim] * num_hidden_layers,
            out_channels=self.head_dim,
            output_layer=False,
        )

        head_hidden = [self.head_dim] * num_head_layers
        self.objectness_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)
        self.graspness_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)
        self.grasp_score_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)

        dir_out = 4 if use_kappa else 3
        self.approach_head = build_branch_module(self.head_dim, head_hidden, dir_out, output_layer=True)
        self.gripper_head = build_branch_module(self.head_dim, head_hidden, dir_out, output_layer=True)
        self.width_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)
        self.depth_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)

        if print_param_counts:
            self._print_param_counts()

    def _print_param_counts(self):
        print("\n=== Grasp Head Parameter Counts ===")
        for name, module in self.named_children():
            print(f"  [{name}]: {sum(p.numel() for p in module.parameters()) / 1e6:.3f}M params")
        total = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  [TOTAL]: {total:.3f}M params")
        print("=" * 40 + "\n")

    def _get_fusion_module(self, fusion_type, in_channels, out_channels=None):
        out_channels = out_channels or self.feature_dim
        if fusion_type == 'none':
            return None
        if fusion_type == 'concat':
            return ConcatFusion(in_channels, out_channels)
        if fusion_type == 'add':
            return AddFusion(in_channels, out_channels)
        raise ValueError(
            f"Unknown rgbd_fusion_type: {fusion_type}. Supported types: 'none', 'concat', 'add'"
        )

    def forward(self, shared_feat, depth_feat, H=None, W=None):
        if self.rgbd_fusion is None:
            base_feat = shared_feat
        else:
            depth_feat = F.interpolate(depth_feat, size=shared_feat.shape[-2:], mode='bilinear', align_corners=False)
            base_feat = self.rgbd_fusion(torch.cat([shared_feat, depth_feat], dim=1))
        base_feat = self.trunk(base_feat)

        feat = self.grasp_branch(base_feat)

        objectness = self.objectness_head(feat)
        graspness_raw = self.graspness_head(feat)
        grasp_score_raw = self.grasp_score_head(feat)
        approach_raw = self.approach_head(feat)
        gripper_raw = self.gripper_head(feat)
        width_raw = self.width_head(feat)
        depth_raw = self.depth_head(feat)

        outputs = {'objectness': objectness}
        outputs.update(
            _decode_grasp_outputs(
                graspness_raw,
                grasp_score_raw,
                approach_raw,
                gripper_raw,
                width_raw,
                depth_raw,
                use_kappa=self.use_kappa,
            )
        )

        if (H, W) != shared_feat.shape[-2:]:
            for key, value in outputs.items():
                outputs[key] = F.interpolate(
                    value,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                )

        return outputs



class DiscreteJointOrientationGraspHead(nn.Module):
    """Joint view-angle orientation baseline with the same scalar regressors.

    The head intentionally does not materialize dense 300x12 logits. It exposes
    low-resolution orientation features plus classifier weights so the loss can
    apply sparse CE only at supervised pixels and inference can classify only
    candidate grasps.
    """

    def __init__(self,
                 feature_dim=128,
                 head_dim=None,
                 num_trunk_layers=1,
                 num_hidden_layers=2,
                 num_head_layers=1,
                 rgbd_fusion_type='concat',
                 output_size=None,
                 num_views=300,
                 num_angles=12,
                 orientation_dim=None,
                 print_param_counts=True,
                 **kwargs,
                 ):
        super().__init__()

        if num_trunk_layers < 1:
            raise ValueError(f"num_trunk_layers must be >= 1, got {num_trunk_layers}")

        self.feature_dim = feature_dim
        self.head_dim = head_dim or feature_dim
        self.output_size = output_size
        self.num_views = int(num_views)
        self.num_angles = int(num_angles)
        self.num_orientation_classes = self.num_views * self.num_angles
        self.orientation_dim = orientation_dim or self.head_dim

        self.rgbd_fusion = self._get_fusion_module(rgbd_fusion_type, 2 * feature_dim, feature_dim)
        self.trunk = build_branch_module(
            in_channels=feature_dim,
            hidden_channels=[self.head_dim] * num_trunk_layers,
            out_channels=self.head_dim,
            output_layer=False,
        )
        self.grasp_branch = build_branch_module(
            in_channels=self.head_dim,
            hidden_channels=[self.head_dim] * num_hidden_layers,
            out_channels=self.head_dim,
            output_layer=False,
        )

        head_hidden = [self.head_dim] * num_head_layers
        self.objectness_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)
        self.graspness_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)
        self.grasp_score_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)
        self.orientation_head = build_branch_module(
            self.head_dim,
            [self.orientation_dim] * num_head_layers,
            self.orientation_dim,
            output_layer=False,
        )
        self.orientation_classifier = nn.Linear(self.orientation_dim, self.num_orientation_classes)
        self.width_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)
        self.depth_head = build_branch_module(self.head_dim, head_hidden, 1, output_layer=True)

        if print_param_counts:
            self._print_param_counts()

    def _print_param_counts(self):
        print("\n=== Discrete Joint Orientation Head Parameter Counts ===")
        for name, module in self.named_children():
            print(f"  [{name}]: {sum(p.numel() for p in module.parameters()) / 1e6:.3f}M params")
        total = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  [TOTAL]: {total:.3f}M params")
        print("=" * 40 + "\n")

    def _get_fusion_module(self, fusion_type, in_channels, out_channels=None):
        out_channels = out_channels or self.feature_dim
        if fusion_type == 'none':
            return None
        if fusion_type == 'concat':
            return ConcatFusion(in_channels, out_channels)
        if fusion_type == 'add':
            return AddFusion(in_channels, out_channels)
        raise ValueError(
            f"Unknown rgbd_fusion_type: {fusion_type}. Supported types: 'none', 'concat', 'add'"
        )

    def forward(self, shared_feat, depth_feat, H=None, W=None):
        if self.rgbd_fusion is None:
            base_feat = shared_feat
        else:
            depth_feat = F.interpolate(depth_feat, size=shared_feat.shape[-2:], mode='bilinear', align_corners=False)
            base_feat = self.rgbd_fusion(torch.cat([shared_feat, depth_feat], dim=1))
        base_feat = self.trunk(base_feat)
        feat = self.grasp_branch(base_feat)

        scalar_outputs = {
            'objectness': self.objectness_head(feat),
            'graspness': torch.sigmoid(self.graspness_head(feat)),
            'grasp_score': torch.sigmoid(self.grasp_score_head(feat)),
            'grasp_width': torch.sigmoid(self.width_head(feat)),
            'grasp_depth': torch.sigmoid(self.depth_head(feat)),
        }

        if (H, W) != shared_feat.shape[-2:]:
            for key, value in scalar_outputs.items():
                scalar_outputs[key] = F.interpolate(
                    value,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                )

        orientation_feat = self.orientation_head(feat)
        B = orientation_feat.shape[0]
        scalar_outputs['_orientation_feat'] = orientation_feat
        scalar_outputs['_orientation_weight'] = (
            self.orientation_classifier.weight.unsqueeze(0).expand(B, -1, -1)
        )
        scalar_outputs['_orientation_bias'] = (
            self.orientation_classifier.bias.unsqueeze(0).expand(B, -1)
        )
        scalar_outputs['_orientation_meta'] = orientation_feat.new_tensor(
            [self.num_views, self.num_angles], dtype=torch.float32
        ).view(1, 2).expand(B, -1)
        scalar_outputs['_orientation_full_hw'] = orientation_feat.new_tensor(
            [H or shared_feat.shape[-2], W or shared_feat.shape[-1]], dtype=torch.float32
        ).view(1, 2).expand(B, -1)
        return scalar_outputs


class DiscreteJointAllAttributesGraspHead(DiscreteJointOrientationGraspHead):
    """Joint 300x12 orientation plus discrete score, width, and depth."""

    def __init__(
        self,
        score_bins=11,
        width_bins=21,
        max_width=0.10,
        depth_values=(0.01, 0.02, 0.03, 0.04),
        attribute_decode='argmax',
        print_param_counts=True,
        **kwargs,
    ):
        super().__init__(print_param_counts=False, **kwargs)
        self.attribute_decode = str(attribute_decode).lower()
        if self.attribute_decode not in {'argmax', 'expectation'}:
            raise ValueError(
                f"attribute_decode must be 'argmax' or 'expectation', got {attribute_decode!r}"
            )

        score_bins = int(score_bins)
        width_bins = int(width_bins)
        if score_bins < 2 or width_bins < 2:
            raise ValueError("score_bins and width_bins must both be >= 2")
        if float(max_width) <= 0:
            raise ValueError(f"max_width must be positive, got {max_width}")
        depth_values = list(depth_values)
        if len(depth_values) < 2:
            raise ValueError("depth_values must contain at least two insertion depths")

        head_hidden = [self.head_dim] * int(kwargs.get('num_head_layers', 1))
        self.grasp_score_head = build_branch_module(
            self.head_dim, head_hidden, score_bins, output_layer=True
        )
        self.width_head = build_branch_module(
            self.head_dim, head_hidden, width_bins, output_layer=True
        )
        self.depth_head = build_branch_module(
            self.head_dim, head_hidden, len(depth_values), output_layer=True
        )

        self.register_buffer(
            'score_centers',
            torch.linspace(0.0, 1.0, score_bins, dtype=torch.float32),
        )
        self.register_buffer(
            'width_centers',
            torch.linspace(0.0, float(max_width), width_bins, dtype=torch.float32),
        )
        self.register_buffer(
            'depth_centers',
            torch.tensor(depth_values, dtype=torch.float32),
        )

        if print_param_counts:
            self._print_param_counts()

    def _decode_attribute(self, logits, centers):
        centers = centers.to(device=logits.device, dtype=logits.dtype)
        if self.attribute_decode == 'expectation':
            value = torch.sum(torch.softmax(logits, dim=1) * centers.view(1, -1, 1, 1), dim=1)
        else:
            class_idx = torch.argmax(logits, dim=1)
            value = centers[class_idx]
        return value.unsqueeze(1)

    @torch.no_grad()
    def _decode_attribute_map(self, logits, centers, H, W, row_chunk=32):
        if (H, W) == logits.shape[-2:]:
            return self._decode_attribute(logits.float(), centers.float())

        B = logits.shape[0]
        logits = logits.float()
        centers = centers.to(device=logits.device, dtype=logits.dtype)
        decoded = logits.new_empty((B, 1, H, W))
        x = (2.0 * (torch.arange(W, device=logits.device, dtype=logits.dtype) + 0.5) / W) - 1.0

        for y_start in range(0, H, row_chunk):
            y_end = min(y_start + row_chunk, H)
            y = (
                2.0
                * (torch.arange(y_start, y_end, device=logits.device, dtype=logits.dtype) + 0.5)
                / H
            ) - 1.0
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            sampled_logits = F.grid_sample(
                logits,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=False,
            )
            decoded[:, :, y_start:y_end] = self._decode_attribute(
                sampled_logits, centers
            )
        return decoded

    def forward(self, shared_feat, depth_feat, H=None, W=None):
        if self.rgbd_fusion is None:
            base_feat = shared_feat
        else:
            depth_feat = F.interpolate(
                depth_feat,
                size=shared_feat.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            base_feat = self.rgbd_fusion(torch.cat([shared_feat, depth_feat], dim=1))
        base_feat = self.trunk(base_feat)
        feat = self.grasp_branch(base_feat)

        score_logits = self.grasp_score_head(feat)
        width_logits = self.width_head(feat)
        depth_logits = self.depth_head(feat)
        target_h = H or shared_feat.shape[-2]
        target_w = W or shared_feat.shape[-1]
        outputs = {
            'objectness': self.objectness_head(feat),
            'graspness': torch.sigmoid(self.graspness_head(feat)),
            'grasp_score': self._decode_attribute_map(
                score_logits, self.score_centers, target_h, target_w
            ),
            'grasp_width': self._decode_attribute_map(
                width_logits, self.width_centers, target_h, target_w
            )
            / self.width_centers[-1].clamp_min(1e-6),
            'grasp_depth': self._decode_attribute_map(
                depth_logits, self.depth_centers, target_h, target_w
            )
            / self.depth_centers[-1].clamp_min(1e-6),
        }
        if (H, W) != shared_feat.shape[-2:]:
            for key in ('objectness', 'graspness'):
                value = outputs[key]
                outputs[key] = F.interpolate(
                    value,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                )

        outputs['_attribute_score_logits'] = score_logits
        outputs['_attribute_width_logits'] = width_logits
        outputs['_attribute_depth_logits'] = depth_logits

        orientation_feat = self.orientation_head(feat)
        B = orientation_feat.shape[0]
        outputs['_orientation_feat'] = orientation_feat
        outputs['_orientation_weight'] = (
            self.orientation_classifier.weight.unsqueeze(0).expand(B, -1, -1)
        )
        outputs['_orientation_bias'] = (
            self.orientation_classifier.bias.unsqueeze(0).expand(B, -1)
        )
        outputs['_orientation_meta'] = orientation_feat.new_tensor(
            [self.num_views, self.num_angles], dtype=torch.float32
        ).view(1, 2).expand(B, -1)
        outputs['_orientation_full_hw'] = orientation_feat.new_tensor(
            [H or shared_feat.shape[-2], W or shared_feat.shape[-1]], dtype=torch.float32
        ).view(1, 2).expand(B, -1)
        outputs['_grasp_score_centers'] = self.score_centers.view(1, -1).expand(B, -1)
        outputs['_grasp_width_centers'] = self.width_centers.view(1, -1).expand(B, -1)
        outputs['_grasp_depth_centers'] = self.depth_centers.view(1, -1).expand(B, -1)
        return outputs


GRASP_HEAD_REGISTRY = {
    'ParallelGraspHead': ParallelGraspHead,
    'DiscreteJointOrientationGraspHead': DiscreteJointOrientationGraspHead,
    'DiscreteJointAllAttributesGraspHead': DiscreteJointAllAttributesGraspHead,
}
