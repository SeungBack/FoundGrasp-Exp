"""Structural compatibility check — instantiate a single backbone block with
random weights (no HF download), monkey-patch it via PeftManager.install(),
and run a forward pass to verify the patched forward matches expected shapes.

Used for backbones whose weights are not in local cache (ViT, SigLIP).
"""
import argparse
import torch
import torch.nn as nn

from model.peft import PeftManager


def make_dinov2_layer(hidden=768):
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer, Dinov2Config
    cfg = Dinov2Config(hidden_size=hidden, num_attention_heads=12)
    return Dinov2Layer(cfg)


def make_dinov3_layer(hidden=768):
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer, DINOv3ViTConfig
    cfg = DINOv3ViTConfig(hidden_size=hidden, num_attention_heads=12)
    return DINOv3ViTLayer(cfg)


def make_vit_layer(hidden=768):
    from transformers.models.vit.modeling_vit import ViTLayer, ViTConfig
    cfg = ViTConfig(hidden_size=hidden, num_attention_heads=12)
    cfg._attn_implementation = "eager"
    return ViTLayer(cfg)


def make_siglip_layer(hidden=768):
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionConfig
    cfg = SiglipVisionConfig(hidden_size=hidden, num_attention_heads=12)
    cfg._attn_implementation = "eager"
    return SiglipEncoderLayer(cfg)


BACKBONES = {
    "dinov2": make_dinov2_layer,
    "dinov3": make_dinov3_layer,
    "vit":    make_vit_layer,
    "siglip": make_siglip_layer,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=list(BACKBONES))
    parser.add_argument("--peft", default="geo_adapter")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=768)
    args = parser.parse_args()

    print(f"=== synthetic {args.backbone} | peft={args.peft} ===")

    # Build N layers of the requested backbone
    layers = [BACKBONES[args.backbone](hidden=args.hidden) for _ in range(args.num_layers)]
    print(f"layer class: {type(layers[0]).__name__}")

    # Build PeftManager
    peft = PeftManager(
        peft_type=args.peft,
        num_layers=args.num_layers,
        embed_dim=args.hidden,
        depth_dim=64,
        layers_to_adapt="all",
        adapter_bottleneck=64,
        adapter_dropout=0.0,
        adapter_scale=1.0,
        adapter_drop_path=0.0,
    )

    # Install
    peft.install(layers)
    print(f"_installed: {peft._installed}")

    # Set context for forward
    B, HW, C = 2, 196, args.hidden
    num_prefix = 1
    peft._ctx_feat_h = 14
    peft._ctx_feat_w = 14
    peft._ctx_num_prefix = num_prefix
    peft._ctx_depth_feat = torch.randn(B, 64, 14, 14)

    # Dummy input: (B, prefix+HW, C)
    x = torch.randn(B, num_prefix + HW, C)
    peft.eval()
    for layer in layers:
        layer.eval()

    # Forward — kwargs depend on backbone
    try:
        with torch.no_grad():
            out = x
            for layer in layers:
                if args.backbone == "dinov3":
                    out = layer(out, position_embeddings=None)
                elif args.backbone == "siglip":
                    out = layer(out, attention_mask=None)
                else:
                    out = layer(out)
        print(f"forward OK. output shape: {tuple(out.shape)}")
        assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
        print("shape preserved.")
    except Exception as e:
        import traceback
        print(f"forward FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
