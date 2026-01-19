"""
Base dataset class for RGB-D data loading
Adapted from ConceptGraphs: https://github.com/concept-graphs/concept-graphs
"""

import abc
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


class BaseRGBDDataset:
    """Abstract base class for RGB-D datasets"""

    def __init__(
        self,
        dataset_path: str,
        stride: int = 1,
        start: int = 0,
        end: int = -1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Args:
            dataset_path: Path to dataset root directory
            stride: Skip every N frames
            start: Starting frame index
            end: Ending frame index (-1 for all frames)
            height: Desired image height (None = use original)
            width: Desired image width (None = use original)
            device: Device to load tensors to
        """
        self.dataset_path = Path(dataset_path)
        self.stride = stride
        self.start = start
        self.end = end
        self.device = device

        # Will be set by subclass
        self.orig_height = None
        self.orig_width = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.depth_scale = None  # Scale factor to convert depth to meters

        # Desired dimensions (None = use original)
        self.height = height
        self.width = width

        # File paths
        self.color_paths = []
        self.depth_paths = []
        self.poses = []

    @abc.abstractmethod
    def _load_filepaths(self):
        """Load color, depth, and pose file paths. Implement in subclass."""
        raise NotImplementedError

    @abc.abstractmethod
    def _load_poses(self):
        """Load camera poses. Implement in subclass."""
        raise NotImplementedError

    def _get_intrinsics_matrix(self) -> np.ndarray:
        """
        Get camera intrinsics matrix K

        Returns:
            K: 3x3 intrinsics matrix
        """
        K = np.eye(3)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    def _preprocess_color(self, color: np.ndarray) -> np.ndarray:
        """
        Preprocess color image (resize if needed)

        Args:
            color: RGB image (H, W, 3)

        Returns:
            Preprocessed RGB image
        """
        if self.height is not None and self.width is not None:
            color = cv2.resize(
                color,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
        return color

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Preprocess depth image (resize and convert to meters)

        Args:
            depth: Depth image (H, W)

        Returns:
            Preprocessed depth in meters (H, W)
        """
        if self.height is not None and self.width is not None:
            depth = cv2.resize(
                depth.astype(float),
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )
        # Convert to meters
        return depth / self.depth_scale

    def __len__(self) -> int:
        """Return number of frames in dataset"""
        return len(self.color_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single frame from the dataset

        Args:
            index: Frame index

        Returns:
            color: RGB image tensor (H, W, 3), float32, range [0, 255]
            depth: Depth image tensor (H, W), float32, in meters
            intrinsics: Camera intrinsics matrix (3, 3)
            pose: Camera pose matrix (4, 4), world-to-camera transform
        """
        raise NotImplementedError

    def get_camera_params(self) -> dict:
        """
        Get camera parameters

        Returns:
            Dictionary with camera parameters
        """
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width if self.width is not None else self.orig_width,
            'height': self.height if self.height is not None else self.orig_height,
        }
