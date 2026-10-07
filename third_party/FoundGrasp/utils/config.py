"""Configuration utilities for loading and merging OmegaConf configs."""

import os
from omegaconf import OmegaConf


def resolve_config_path(config_arg, config_root='configs'):
    """Resolve a config argument to an existing YAML path.

    Supported inputs:
      - `exp8/foo.yaml`
      - `configs/exp8/foo.yaml`
      - `exp8/foo`
      - absolute paths
    """
    config_arg = os.path.expanduser(config_arg)
    candidates = []

    def _add(path):
        if path and path not in candidates:
            candidates.append(path)

    # Use the path as-is first, then try adding the yaml suffix.
    _add(config_arg)
    root, ext = os.path.splitext(config_arg)
    if ext == '':
        _add(config_arg + '.yaml')

    # If the user did not include the config root, try under `configs/`.
    norm_arg = os.path.normpath(config_arg)
    norm_root = os.path.normpath(config_root)
    starts_with_root = (
        norm_arg == norm_root or
        norm_arg.startswith(norm_root + os.sep)
    )
    if not os.path.isabs(config_arg) and not starts_with_root:
        _add(os.path.join(config_root, config_arg))
        if ext == '':
            _add(os.path.join(config_root, config_arg + '.yaml'))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve config '{config_arg}'. Tried: {candidates}"
    )


def config_name_from_path(config_path, config_root='configs'):
    """Create a stable config name for logs/checkpoints from a resolved path."""
    norm_path = os.path.normpath(config_path)
    norm_root = os.path.normpath(config_root)
    rel_path = norm_path

    if not os.path.isabs(norm_path):
        if norm_path == norm_root or norm_path.startswith(norm_root + os.sep):
            rel_path = os.path.relpath(norm_path, norm_root)
    else:
        abs_root = os.path.abspath(config_root)
        abs_path = os.path.abspath(config_path)
        if abs_path == abs_root or abs_path.startswith(abs_root + os.sep):
            rel_path = os.path.relpath(abs_path, abs_root)

    return os.path.splitext(rel_path)[0]


def load_config_with_base(config_path):
    """Load config and merge with base config if __base__ is specified.

    Args:
        config_path: Path to the config file (YAML format)

    Returns:
        OmegaConf DictConfig with base configs merged

    Example:
        >>> config = load_config_with_base('configs/dinov3.yaml')
        >>> print(config.dataset.name)
        GraspNet1B
    """
    config = OmegaConf.load(config_path)

    # Check if this config has a base config
    if '__base__' in config:
        base_name = config.pop('__base__')
        base_path = os.path.join(os.path.dirname(config_path), base_name)

        # Recursively load base config (in case it also has a base)
        base_config = load_config_with_base(base_path)

        # Merge: base config first, then override with current config
        config = OmegaConf.merge(base_config, config)

    return config
