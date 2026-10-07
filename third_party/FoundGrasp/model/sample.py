import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. Helper Modules (Economic Grasp Interaction Blocks)
# ==============================================================================

class MultiHeadAttn(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = int(dim // nhead)
        assert self.nhead * self.head_dim == self.dim
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def attention(queries, keys, values, mask=None, dropout=None):
        head_dim = queries.size(-1)
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ values

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # Input: (B, L, D) -> Linear -> Split Heads -> (B, H, L, D_head)
        queries, keys, values = [
            layer(x).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears[:3], (query, key, value))
        ]
        result = self.attention(queries, keys, values, mask, self.dropout)
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.linears[-1](result)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head, msa_dropout):
        super().__init__()
        self.dim = dim
        self.msa = MultiHeadAttn(dim, n_head, dropout=msa_dropout)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, q, k, v, mask=None):
        msa = self.msa(q, k, v, mask)
        x = self.norm1(v + msa)
        return x


class EconomicInteractionHead(nn.Module):
    """
    Economic Grasp Style Interaction Head adapted for Dense Feature Map (2D).
    Replacing simple Conv heads in Stage 2.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Feature Projectors (Shared Feature -> Specific Features)
        # Note: Reduced channels (256 -> 64) for efficiency as in paper
        self.conv_approach_feat = nn.Conv2d(input_dim, hidden_dim, 1)
        self.conv_gripper_feat = nn.Conv2d(input_dim, hidden_dim, 1)
        self.conv_depth_feat = nn.Conv2d(input_dim, hidden_dim, 1)
        self.conv_width_feat = nn.Conv2d(input_dim, hidden_dim, 1)
        self.conv_quality_feat = nn.Conv2d(input_dim, hidden_dim, 1) # Score -> Quality

        # 2. Global Interaction Module
        # Concatenated feature dim = hidden_dim * 5 (App + Grp + Dep + Wid + Qual)
        total_dim = hidden_dim * 5
        self.global_interaction_module = AttentionModule(dim=total_dim, n_head=4, msa_dropout=0.05)

        # 3. Final Predictors (From Refined Features)
        self.pred_approach = nn.Conv2d(total_dim, 3, 1) # Approach Vector (3)
        self.pred_gripper = nn.Conv2d(total_dim, 3, 1)  # Gripper Vector (3)
        self.pred_depth = nn.Conv2d(total_dim, 1, 1)    # Depth (0~1)
        self.pred_width = nn.Conv2d(total_dim, 1, 1)    # Width (0~1)
        self.pred_quality = nn.Conv2d(total_dim, 1, 1)  # Quality (0~1)

        # Activations
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - Base Feature
        Returns:
            Dict of predictions
        """
        B, C, H, W = x.shape

        # 1. Project to specific sub-spaces
        f_app = self.conv_approach_feat(x)
        f_grp = self.conv_gripper_feat(x)
        f_dep = self.conv_depth_feat(x)
        f_wid = self.conv_width_feat(x)
        f_qual = self.conv_quality_feat(x)

        # 2. Interaction (Concat -> Flatten -> Attention -> Reshape)
        # (B, 5*hidden, H, W)
        interaction_feat = torch.cat([f_app, f_grp, f_dep, f_wid, f_qual], dim=1)
        
        # Flatten for Attention: (B, H*W, D)
        feat_flat = interaction_feat.flatten(2).transpose(1, 2)
        
        # Apply Self-Attention (Interaction between params per pixel & spatial context)
        # [Warning] If H*W is very large (e.g. > 10000), this will be slow/OOM.
        # Assuming reasonable feature map size or downsampling.
        feat_refined = self.global_interaction_module(feat_flat, feat_flat, feat_flat)
        
        # Reshape back: (B, D, H, W)
        feat_refined = feat_refined.transpose(1, 2).view(B, -1, H, W)

        # 3. Split & Predict (Note: Economic Grasp splits feature back, 
        # but simpler implementation is to predict from the shared refined feature)
        # Or strictly following logic: use the refined chunks. 
        # Here we use the FULL refined feature for all predictors to maximize context.
        
        out_approach = self.pred_approach(feat_refined) # Raw vector
        out_gripper = self.pred_gripper(feat_refined)   # Raw vector
        out_width = self.sigmoid(self.pred_width(feat_refined))
        out_depth = self.sigmoid(self.pred_depth(feat_refined))
        out_quality = self.sigmoid(self.pred_quality(feat_refined))

        return out_approach, out_gripper, out_width, out_depth, out_quality


# ==============================================================================
# 2. Main Hierarchical Head (Integrating Stages)
# ==============================================================================

def build_head_module(in_dim, hidden_dims, out_dim, norm_type, act_type):
    """ Helper to build simple conv heads """
    layers = []
    prev_dim = in_dim
    for h_dim in hidden_dims:
        layers.append(nn.Conv2d(prev_dim, h_dim, 3, padding=1))
        if norm_type == 'group':
            layers.append(nn.GroupNorm(8, h_dim))
        layers.append(nn.ReLU(inplace=True))
        prev_dim = h_dim
    layers.append(nn.Conv2d(prev_dim, out_dim, 1))
    
    if act_type == 'sigmoid':
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class HierarchicalGraspHeadv1(nn.Module):
    def __init__(self, 
                 input_type='xyzc',
                 input_dim=128, 
                 hidden_dim=128):
        super().__init__()
        
        # --- Depth Encoder & Fusion (Same as before) ---
        if 'xyz' not in input_type and 'd' not in input_type:
            raise ValueError("Ours mode requires depth-related input_type")
        in_channels = 0
        if 'xyz' in input_type: in_channels += 3
        if 'd' in input_type: in_channels += 1
        if 'c' in input_type: in_channels += 1

        self.depth_encoder = GeometricDepthStem(in_channels, 32, 1)
        self.fusion = ResidualFusion(input_dim + 32, hidden_dim)

        # ======================================================================
        # Stage 1: Graspness (Region Proposal)
        # ======================================================================
        # "여기가 잡을만한 구역인가?" (Target: Average Score)
        self.graspness_head = build_head_module(hidden_dim, [hidden_dim], 1, 'group', 'sigmoid')
        self.objectness_head = build_head_module(hidden_dim, [hidden_dim], 1, 'group', 'sigmoid')

        # ======================================================================
        # Stage 2: Interaction Head (Pose & Quality)
        # ======================================================================
        # "어떻게 잡을 것인가?" + "이 땅의 잠재력(Quality)은?"
        # Economic Grasp의 Interaction 로직 적용
        self.interaction_head = EconomicInteractionHead(input_dim=hidden_dim, hidden_dim=64)

        # ======================================================================
        # Stage 3: Confidence Head (Dynamic Critic)
        # ======================================================================
        # "내 자세가 얼마나 정확한가?" (Target: Alignment Score)
        # Input: Base Feature + All Predicted Params
        critic_input_dim = hidden_dim + 3 + 3 + 1 + 1 # Feat + App + Grp + Wid + Dep
        self.confidence_head = build_head_module(critic_input_dim, [hidden_dim], 1, 'group', 'sigmoid')

    def forward(self, shared_feat, depth):
        # 1. Depth Encoding & Fusion
        _, _, H, W = depth.shape
        d_feat = self.depth_encoder(depth)
        shared_feat = F.interpolate(shared_feat, size=d_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        base = torch.cat([shared_feat, d_feat], dim=1)
        base = self.fusion(base) 

        # ----------------------------------------------------------------------
        # Stage 1: Static Graspness
        # ----------------------------------------------------------------------
        objectness = self.objectness_head(base)
        graspness = self.graspness_head(base)

        # ----------------------------------------------------------------------
        # Stage 2: Interaction Head (Pose & Quality)
        # ----------------------------------------------------------------------
        # Interaction 모듈이 5가지 값을 동시에 예측
        # Quality(Potential)가 여기서 예측됨 (Economic Grasp의 Score 역할)
        pred_app_raw, pred_grp_raw, width, depth, grasp_score = self.interaction_head(base)

        # Vector Normalization & Orthogonalization
        approach_normed = F.normalize(pred_app_raw, p=2, dim=1)
        
        # Gram-Schmidt for Gripper
        dot = torch.sum(approach_normed * pred_grp_raw, dim=1, keepdim=True)
        gripper_orthog = F.normalize(pred_grp_raw - dot * approach_normed, p=2, dim=1)

        # ----------------------------------------------------------------------
        # Stage 3: Confidence Head (Dynamic Alignment)
        # ----------------------------------------------------------------------
        # 내 행동(Pose)을 평가 (Critic)
        critic_input = torch.cat([
            base,
            approach_normed.detach(), # 평가 대상이므로 Gradient 차단 권장 (선택)
            gripper_orthog.detach(),
            width.detach(),
            depth.detach()
        ], dim=1)
        
        confidence_score = self.confidence_head(critic_input)

        # ----------------------------------------------------------------------
        # Output Collection
        # ----------------------------------------------------------------------
        outs = {
            'objectness': objectness,
            'graspness': graspness,        # Target: GT Average Score
            'grasp_score': grasp_score,# Target: GT Best Score (from Stage 2)
            'confidence': confidence_score,# Target: Alignment Score (from Stage 3)
            
            # [최종 추론 Score] Potential * Precision
            'final_score': grasp_score * confidence_score, 
            
            'approach_dir': approach_normed,
            'gripper_dir': gripper_orthog,
            'grasp_width': width, 
            'grasp_depth': depth, 
            'approach_dir_raw': pred_app_raw,
            'gripper_dir_raw': pred_grp_raw
        }

        # Resize to Original Input Resolution
        for k, v in outs.items():
            outs[k] = F.interpolate(v, size=(H, W), mode='bilinear', align_corners=False)
            
        return outs

# (GeometricDepthStem, ResidualFusion 클래스는 기존과 동일하다고 가정)