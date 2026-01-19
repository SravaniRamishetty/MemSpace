"""
Replica dataset loader
Adapted from ConceptGraphs ReplicaDataset
"""

import glob
from pathlib import Path
from typing import Optional, Tuple

import imageio
import numpy as np
import torch
from natsort import natsorted

from memspace.dataset.base_dataset import BaseRGBDDataset


class ReplicaDataset(BaseRGBDDataset):
    """
    Dataset class for Replica dataset

    Expected directory structure:
        <dataset_path>/
            results/
                frame000000.jpg
                frame000001.jpg
                ...
                depth000000.png
                depth000001.png
                ...
            traj.txt  # Camera poses
    """

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
        super().__init__(dataset_path, stride, start, end, height, width, device)

        # Replica camera parameters (fixed for all scenes)
        self.orig_height = 680
        self.orig_width = 1200
        self.fx = 600.0
        self.fy = 600.0
        self.cx = 599.5
        self.cy = 339.5
        self.depth_scale = 6553.5  # Replica depth scale to meters

        # Load filepaths and poses
        self._load_filepaths()
        self._load_poses()

        # Apply start/end/stride
        total_frames = len(self.color_paths)
        if self.end == -1:
            self.end = total_frames

        self.color_paths = self.color_paths[self.start : self.end : self.stride]
        self.depth_paths = self.depth_paths[self.start : self.end : self.stride]
        self.poses = self.poses[self.start : self.end : self.stride]

        print(f"Loaded Replica dataset: {len(self)} frames")
        print(f"  Resolution: {self.get_camera_params()['width']}x{self.get_camera_params()['height']}")
        print(f"  Stride: {self.stride}, Start: {self.start}, End: {self.end}")

    def _load_filepaths(self):
        """Load color and depth image paths"""
        results_dir = self.dataset_path / "results"

        # Find color images (frame*.jpg)
        color_paths = natsorted(glob.glob(str(results_dir / "frame*.jpg")))
        if len(color_paths) == 0:
            raise ValueError(f"No color images found in {results_dir}")

        # Find depth images (depth*.png)
        depth_paths = natsorted(glob.glob(str(results_dir / "depth*.png")))
        if len(depth_paths) == 0:
            raise ValueError(f"No depth images found in {results_dir}")

        if len(color_paths) != len(depth_paths):
            raise ValueError(
                f"Mismatch: {len(color_paths)} color images but {len(depth_paths)} depth images"
            )

        self.color_paths = [Path(p) for p in color_paths]
        self.depth_paths = [Path(p) for p in depth_paths]

    def _load_poses(self):
        """
        Load camera poses from traj.txt

        Each line in traj.txt is a 4x4 transformation matrix (16 values space-separated)
        These are camera-to-world transforms
        """
        pose_file = self.dataset_path / "traj.txt"
        if not pose_file.exists():
            raise ValueError(f"Pose file not found: {pose_file}")

        with open(pose_file, "r") as f:
            lines = f.readlines()

        num_images = len(self.color_paths)
        if len(lines) < num_images:
            raise ValueError(
                f"Not enough poses: {len(lines)} lines but {num_images} images"
            )

        poses = []
        for i in range(num_images):
            # Parse 16 values into 4x4 matrix
            values = list(map(float, lines[i].split()))
            if len(values) != 16:
                raise ValueError(f"Invalid pose on line {i}: expected 16 values, got {len(values)}")

            c2w = np.array(values).reshape(4, 4)
            poses.append(c2w)

        self.poses = poses

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single frame

        Returns:
            color: (H, W, 3) RGB image, float32, range [0, 255]
            depth: (H, W) depth in meters, float32
            intrinsics: (3, 3) camera intrinsics matrix
            pose: (4, 4) camera-to-world pose matrix
        """
        # Load color image
        color = np.asarray(imageio.imread(self.color_paths[index]), dtype=np.float32)
        color = self._preprocess_color(color)

        # Load depth image
        depth = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.float32)
        depth = self._preprocess_depth(depth)

        # Get intrinsics
        K = self._get_intrinsics_matrix()

        # Adjust intrinsics if resizing
        if self.height is not None and self.width is not None:
            scale_y = self.height / self.orig_height
            scale_x = self.width / self.orig_width
            K[0, 0] *= scale_x  # fx
            K[1, 1] *= scale_y  # fy
            K[0, 2] *= scale_x  # cx
            K[1, 2] *= scale_y  # cy

        # Get pose (camera-to-world)
        pose = self.poses[index]

        # Convert to tensors
        color = torch.from_numpy(color).float().to(self.device)
        depth = torch.from_numpy(depth).float().to(self.device)
        intrinsics = torch.from_numpy(K).float().to(self.device)
        pose = torch.from_numpy(pose).float().to(self.device)

        return color, depth, intrinsics, pose
