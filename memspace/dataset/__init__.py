"""Dataset loading utilities for RGB-D data"""

from omegaconf import DictConfig
from typing import Any


def get_dataset(cfg: DictConfig, device: str = 'cuda', **override_kwargs) -> Any:
    """
    Factory function to create dataset instances from configuration.

    Follows ConceptGraphs' pattern: config-driven dataset instantiation.

    Args:
        cfg: OmegaConf dataset config (from cfg.dataset)
        device: Device to load data on ('cuda' or 'cpu')
        **override_kwargs: Additional keyword arguments to override config values

    Returns:
        Dataset instance

    Example:
        >>> from omegaconf import OmegaConf
        >>> from memspace.dataset import get_dataset
        >>>
        >>> cfg = OmegaConf.load('memspace/configs/demo_1_1.yaml')
        >>> dataset = get_dataset(cfg.dataset, device='cuda')
    """
    # Get dataset type from config (default to 'replica')
    dataset_type = cfg.get('dataset_type', 'replica').lower()

    # Get common parameters
    dataset_path = override_kwargs.pop('dataset_path', cfg.dataset_path)
    stride = override_kwargs.pop('stride', cfg.stride)
    height = override_kwargs.pop('height', cfg.get('height', 480))
    width = override_kwargs.pop('width', cfg.get('width', 640))

    # Handle max_frames
    max_frames = override_kwargs.pop('max_frames', cfg.get('max_frames', None))
    start_frame = override_kwargs.pop('start_frame', cfg.get('start_frame', 0))

    # Calculate end frame
    if max_frames is not None and max_frames > 0:
        end_frame = start_frame + (max_frames * stride)
    else:
        end_frame = override_kwargs.pop('end_frame', cfg.get('end_frame', -1))

    # Import and instantiate appropriate dataset class
    if dataset_type == 'replica':
        from memspace.dataset.replica_dataset import ReplicaDataset
        return ReplicaDataset(
            dataset_path=dataset_path,
            stride=stride,
            start=start_frame,
            end=end_frame,
            height=height,
            width=width,
            device=device,
            **override_kwargs
        )
    else:
        raise ValueError(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Supported types: 'replica'. "
            f"Check dataset_type field in your config YAML."
        )
