"""
Optimizer utilities including layer-wise learning rate decay.
"""
import torch


def get_layer_id_for_vit(name, num_layers):
    """
    Get layer ID for Vision Transformer (ViT) style models like DINOv2.

    Args:
        name: Parameter name
        num_layers: Total number of transformer layers

    Returns:
        Layer ID (0 for embeddings, 1 to num_layers for transformer layers, num_layers+1 for head)
    """
    if name.startswith('backbone'):
        # Remove 'backbone.' prefix to parse the actual layer structure
        name = name[len('backbone.'):]

        # Embedding layers (patch_embed, cls_token, pos_embed, etc.)
        if name.startswith('embeddings') or 'cls_token' in name or 'pos_embed' in name or 'patch_embed' in name:
            return 0
        # Transformer encoder layers
        elif 'encoder.layer' in name or 'layers' in name:
            # Extract layer number from encoder.layer.X or layers.X
            if 'encoder.layer.' in name:
                layer_id = int(name.split('encoder.layer.')[1].split('.')[0]) + 1
            elif 'layers.' in name:
                layer_id = int(name.split('layers.')[1].split('.')[0]) + 1
            else:
                layer_id = 1
            return layer_id
        # Normalization layer after encoder
        elif 'layernorm' in name.lower() or 'norm' in name:
            return num_layers
        else:
            return num_layers
    else:
        # Non-backbone parameters (decoder, head, etc.) get highest layer ID
        return num_layers + 1


def get_parameter_groups_with_layer_decay(
    model,
    learning_rate,
    weight_decay,
    layer_decay_rate=0.65,
    skip_list=('bias', 'LayerNorm.weight', 'LayerNorm.bias', 'norm.weight', 'norm.bias'),
    no_decay_prefix=None
):
    """
    Create parameter groups with layer-wise learning rate decay.

    This is commonly used for fine-tuning Vision Transformers where lower layers
    (closer to input) get smaller learning rates than higher layers (closer to output).

    Args:
        model: The model to create parameter groups for
        learning_rate: Base learning rate for the highest layer
        weight_decay: Weight decay coefficient
        layer_decay_rate: Decay rate for learning rate (e.g., 0.65 means each lower layer
                         gets 0.65x the learning rate of the layer above it)
        skip_list: Parameter names to skip weight decay (typically bias and normalization)

    Returns:
        List of parameter groups for optimizer

    Example:
        If layer_decay_rate=0.65, base_lr=1e-4, num_layers=12:
        - Layer 13 (head): lr = 1e-4
        - Layer 12 (top encoder): lr = 6.5e-5
        - Layer 11: lr = 4.225e-5
        - ...
        - Layer 1 (bottom encoder): lr = very small
        - Layer 0 (embeddings): lr = very very small
    """
    parameter_group_names = {}
    parameter_group_vars = {}

    # Get number of layers in backbone
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'layer'):
            num_layers = len(model.backbone.encoder.layer)
        elif hasattr(model.backbone, 'layers'):
            num_layers = len(model.backbone.layers)
        elif hasattr(model.backbone, 'config') and hasattr(model.backbone.config, 'num_hidden_layers'):
            num_layers = model.backbone.config.num_hidden_layers
        else:
            print("Warning: Could not determine number of layers, using default 12")
            num_layers = 12
    else:
        print("Warning: Model has no backbone attribute, using default 12 layers")
        num_layers = 12

    print(f"\nSetting up layer-wise learning rate decay with {num_layers} backbone layers")
    print(f"  Base LR: {learning_rate:.2e}")
    print(f"  Decay rate: {layer_decay_rate}")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Get layer id and corresponding learning rate
        layer_id = get_layer_id_for_vit(name, num_layers)
        group_name = f"layer_{layer_id}"

        # Calculate layer-wise learning rate
        # Higher layer_id = closer to output = higher learning rate
        if layer_id == 0:
            # Embeddings: apply full decay
            layer_lr = learning_rate * (layer_decay_rate ** (num_layers + 1))
        elif layer_id == num_layers + 1:
            # Head/decoder: use base learning rate
            layer_lr = learning_rate
        else:
            # Transformer layers: decay based on distance from top
            # layer_id=1 (bottom) gets more decay, layer_id=num_layers (top) gets less
            layer_lr = learning_rate * (layer_decay_rate ** (num_layers + 1 - layer_id))

        # Determine if weight decay should be applied
        # Skip weight decay for bias, normalization, and prompt parameters
        if no_decay_prefix and name.startswith(no_decay_prefix):
            group_name_wd = f"{group_name}_no_decay"
            this_weight_decay = 0.0
        elif any(skip_name in name for skip_name in skip_list):
            group_name_wd = f"{group_name}_no_decay"
            this_weight_decay = 0.0
        else:
            group_name_wd = f"{group_name}_decay"
            this_weight_decay = weight_decay

        # Create parameter group if it doesn't exist
        if group_name_wd not in parameter_group_names:
            parameter_group_names[group_name_wd] = {
                "params": [],
                "lr": layer_lr,
                "weight_decay": this_weight_decay,
                "layer_id": layer_id,
            }
            parameter_group_vars[group_name_wd] = {
                "params": [],
                "lr": layer_lr,
                "weight_decay": this_weight_decay,
                "layer_id": layer_id,
            }

        # Add parameter to group
        parameter_group_names[group_name_wd]["params"].append(name)
        parameter_group_vars[group_name_wd]["params"].append(param)

    # Print summary of learning rates per layer
    print(f"\n  Layer-wise Learning Rates:")
    layer_lrs = {}
    for group_name in sorted(parameter_group_names.keys()):
        layer_id = parameter_group_names[group_name]["layer_id"]
        lr = parameter_group_names[group_name]["lr"]
        wd = parameter_group_names[group_name]["weight_decay"]
        num_params = len(parameter_group_names[group_name]["params"])

        if layer_id not in layer_lrs:
            layer_lrs[layer_id] = lr
            layer_type = "Embeddings" if layer_id == 0 else f"Encoder-{layer_id}" if layer_id <= num_layers else "Head/Decoder"
            print(f"    Layer {layer_id:2d} ({layer_type:15s}): lr={lr:.2e}, wd={wd:.1e}, params={num_params:4d}")

    return list(parameter_group_vars.values())


def build_optimizer(model, config):
    """
    Build optimizer with optional layer-wise learning rate decay.

    Args:
        model: The model to optimize
        config: Training configuration containing:
            - learning_rate: Base learning rate
            - weight_decay: Weight decay coefficient
            - layer_decay_rate: (Optional) Layer-wise decay rate. If provided and backbone is not frozen,
                               will use layer-wise LR decay. Set to None or omit to disable.
            - optimizer: (Optional) Optimizer type, default is 'adamw'

    Returns:
        Configured optimizer
    """
    learning_rate = config.training.learning_rate
    weight_decay = config.training.weight_decay
    layer_decay_rate = config.training.get('layer_decay_rate', None)
    optimizer_type = config.training.get('optimizer', 'adamw').lower()

    # Check if backbone is frozen
    backbone_frozen = config.model.backbone.get('freeze', False)

    # Check if using VPT prompt tuning
    prompt_mode = config.model.prompt.get('mode', 'none').lower()
    use_vpt = prompt_mode in ('vpt', 'vipt', 'ours')

    # Use layer-wise LR decay only if:
    # 1. layer_decay_rate is provided and > 0
    # 2. backbone is NOT frozen
    use_layer_decay = (layer_decay_rate is not None and
                      layer_decay_rate > 0 and
                      layer_decay_rate < 1.0 and
                      not backbone_frozen)

    if use_layer_decay:
        print(f"\nUsing layer-wise learning rate decay (rate={layer_decay_rate})")
        parameter_groups = get_parameter_groups_with_layer_decay(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            layer_decay_rate=layer_decay_rate,
            no_decay_prefix='prompt_generator' if use_vpt else None
        )
    else:
        if backbone_frozen:
            print(f"\nBackbone is frozen, using standard optimizer without layer-wise decay")
        else:
            print(f"\nLayer-wise decay disabled, using standard optimizer")

        if use_vpt:
            # Separate prompt parameters (wd=0) from the rest
            prompt_params = []
            other_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith('prompt_generator'):
                    prompt_params.append(param)
                else:
                    other_params.append(param)
            parameter_groups = [
                {'params': other_params, 'weight_decay': weight_decay},
                {'params': prompt_params, 'weight_decay': 0.0},
            ]
            print(f"  VPT prompt parameters ({len(prompt_params)}) set to weight_decay=0")
        else:
            parameter_groups = [
                {'params': [p for n, p in model.named_parameters() if p.requires_grad], 'weight_decay': weight_decay},
            ]

    # Create optimizer
    if optimizer_type == 'adamw':
        if use_layer_decay:
            optimizer = torch.optim.AdamW(parameter_groups)
        else:
            optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=learning_rate,
                weight_decay=weight_decay
            )
    elif optimizer_type == 'adam':
        if use_layer_decay:
            optimizer = torch.optim.Adam(parameter_groups)
        else:
            optimizer = torch.optim.Adam(
                parameter_groups,
                lr=learning_rate,
                weight_decay=weight_decay
            )
    elif optimizer_type == 'sgd':
        if use_layer_decay:
            optimizer = torch.optim.SGD(parameter_groups, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(
                parameter_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    print(f"Optimizer: {optimizer_type.upper()}")
    print(f"Base learning rate: {learning_rate:.2e}")
    print(f"Weight decay: {weight_decay:.2e}")

    return optimizer
