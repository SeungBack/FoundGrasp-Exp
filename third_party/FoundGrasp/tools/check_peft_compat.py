"""Compatibility check for PEFT install across backbones.

Usage:
    python tools/check_peft_compat.py --config configs/paper/dinov2.yaml [--peft geo_adapter]
"""
import argparse
import io
import sys
import torch

from utils.config import load_config_with_base, resolve_config_path
from model.found_grasp import FoundGrasp


def capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.stdout = saved
    return result, buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--peft", default=None, help="Override config.model.peft.type")
    parser.add_argument("--no-forward", action="store_true")
    args = parser.parse_args()

    config = load_config_with_base(resolve_config_path(args.config))
    if args.peft is not None:
        config.model.peft.type = args.peft

    print(f"=== {args.config} | backbone={config.model.backbone.name} | peft={config.model.peft.type} ===")

    model, log = capture_stdout(FoundGrasp, config=config.model)

    install_lines = [
        l for l in log.splitlines()
        if "[PEFT" in l or "[AdaptFormer" in l or "fall" in l.lower() or "Installed" in l
    ]
    print("--- install() output ---")
    for l in install_lines:
        print(l)
    if not install_lines:
        print("(no install log)")

    print(f"--- state ---")
    if model.peft is None:
        print("peft = None (no PEFT applied)")
    else:
        print(f"peft._installed = {model.peft._installed}")
        print(f"peft.peft_type  = {model.peft.peft_type}")

    if args.no_forward:
        return

    # Dummy forward
    print(f"--- dummy forward ---")
    img_h, img_w = config.dataset.image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    B = 1
    data = {
        "pixel_values": torch.randn(B, 3, img_h, img_w, device=device),
        "depth": torch.rand(B, 1, img_h, img_w, device=device) * 0.5 + 0.5,
        "xyz": torch.randn(B, 3, img_h, img_w, device=device),
        "depth_mask": torch.ones(B, 1, img_h, img_w, device=device),
    }
    try:
        with torch.no_grad():
            out = model(data)
        if isinstance(out, dict):
            shapes = {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for k, v in out.items()}
            print(f"forward OK. output: {shapes}")
        else:
            print(f"forward OK. output type: {type(out)}")
    except Exception as e:
        import traceback
        print(f"forward FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
