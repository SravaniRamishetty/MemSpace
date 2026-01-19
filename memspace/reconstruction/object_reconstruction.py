#!/usr/bin/env python3
"""
Per-Object 3D Reconstruction

This module provides object-level 3D reconstruction by combining:
- Object tracking (Phase 2.3)
- Point cloud accumulation (Phase 3.2)
- Masked depth integration for each tracked object
"""

import numpy as np
import open3d as o3d
from typing import Dict, List, Optional
import torch

from memspace.reconstruction.pointcloud_fusion import PointCloudAccumulator


class ObjectReconstructor:
    """
    Manages 3D reconstruction for multiple tracked objects

    Creates and maintains a separate point cloud for each tracked object,
    integrating only the pixels that belong to that object (using masks).
    """

    def __init__(
        self,
        voxel_size: float = 0.02,
        max_depth: float = 10.0,
        min_points: int = 100,  # Minimum points to keep an object
        outlier_removal: bool = True,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
    ):
        """
        Initialize Object Reconstructor

        Args:
            voxel_size: Voxel size for downsampling each object's point cloud
            max_depth: Maximum depth to consider (meters)
            min_points: Minimum points for an object to be valid
            outlier_removal: Whether to remove statistical outliers
            outlier_nb_neighbors: Neighbors for outlier detection
            outlier_std_ratio: Std ratio for outlier detection
        """
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        self.min_points = min_points
        self.outlier_removal = outlier_removal
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio

        # Dictionary: object_id -> PointCloudAccumulator
        self.object_clouds: Dict[int, PointCloudAccumulator] = {}

        # Track how many frames each object has been integrated
        self.object_frame_counts: Dict[int, int] = {}

    def integrate_frame(
        self,
        object_ids: List[int],
        masks: np.ndarray,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        """
        Integrate a frame with tracked objects

        Args:
            object_ids: List of object IDs for each mask (length M)
            masks: Binary masks (M, H, W), bool or uint8
            color: RGB image (H, W, 3), uint8
            depth: Depth image (H, W), float32
            intrinsics: Camera intrinsics (3, 3) or (4, 4)
            extrinsics: Camera pose (4, 4) - camera-to-world transform
        """
        # Convert inputs to numpy if needed
        if isinstance(color, torch.Tensor):
            color = color.cpu().numpy()
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        # Ensure correct types
        color = color.astype(np.uint8)
        depth = depth.astype(np.float32)

        # Process each object
        for obj_id, mask in zip(object_ids, masks):
            # Create or get accumulator for this object
            if obj_id not in self.object_clouds:
                self.object_clouds[obj_id] = PointCloudAccumulator(
                    voxel_size=self.voxel_size,
                    max_depth=self.max_depth,
                    outlier_removal=self.outlier_removal,
                    outlier_nb_neighbors=self.outlier_nb_neighbors,
                    outlier_std_ratio=self.outlier_std_ratio,
                )
                self.object_frame_counts[obj_id] = 0

            # Create masked depth (zero out non-object pixels)
            masked_depth = depth.copy()
            masked_depth[~mask] = 0.0  # Zero depth for non-object pixels

            # Create masked color (optional: zero out or keep background)
            # We'll keep the color for better visualization even if masked
            # The depth mask will control which points are generated

            # Integrate this object's masked depth
            self.object_clouds[obj_id].integrate(
                color=color,
                depth=masked_depth,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )

            self.object_frame_counts[obj_id] += 1

    def get_object_cloud(self, object_id: int, clean: bool = True) -> Optional[o3d.geometry.PointCloud]:
        """
        Get point cloud for a specific object

        Args:
            object_id: Object ID
            clean: Whether to apply outlier removal

        Returns:
            Point cloud or None if object not found or too few points
        """
        if object_id not in self.object_clouds:
            return None

        pcd = self.object_clouds[object_id].get_point_cloud(clean=clean)

        # Filter out objects with too few points
        if len(pcd.points) < self.min_points:
            return None

        return pcd

    def get_all_objects(self) -> Dict[int, o3d.geometry.PointCloud]:
        """
        Get point clouds for all objects

        Returns:
            Dictionary mapping object_id to point cloud
        """
        result = {}
        for obj_id in self.object_clouds.keys():
            pcd = self.get_object_cloud(obj_id, clean=True)
            if pcd is not None:
                result[obj_id] = pcd
        return result

    def extract_object_mesh(
        self,
        object_id: int,
        method: str = 'poisson',
        poisson_depth: int = 8,
    ) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Extract mesh for a specific object

        Args:
            object_id: Object ID
            method: Mesh reconstruction method
            poisson_depth: Octree depth for Poisson

        Returns:
            Triangle mesh or None
        """
        if object_id not in self.object_clouds:
            return None

        return self.object_clouds[object_id].extract_mesh(
            method=method,
            poisson_depth=poisson_depth,
            clean=True,
        )

    def get_statistics(self) -> dict:
        """
        Get statistics about all reconstructed objects

        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_objects': len(self.object_clouds),
            'objects': {}
        }

        for obj_id, accumulator in self.object_clouds.items():
            pcd = self.get_object_cloud(obj_id, clean=False)

            obj_stats = {
                'num_frames': self.object_frame_counts[obj_id],
                'num_points': len(pcd.points) if pcd else 0,
            }

            if pcd and len(pcd.points) > 0:
                bbox = pcd.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                obj_stats['bounding_box_extent'] = extent.tolist()
                obj_stats['volume'] = extent[0] * extent[1] * extent[2]

            stats['objects'][obj_id] = obj_stats

        return stats

    def filter_objects_by_points(self, min_points: int) -> List[int]:
        """
        Get list of valid object IDs with enough points

        Args:
            min_points: Minimum number of points required

        Returns:
            List of valid object IDs
        """
        valid_objects = []
        for obj_id in self.object_clouds.keys():
            pcd = self.get_object_cloud(obj_id, clean=False)
            if pcd and len(pcd.points) >= min_points:
                valid_objects.append(obj_id)
        return valid_objects

    def get_object_color(self, object_id: int) -> np.ndarray:
        """
        Get a consistent color for visualizing an object

        Args:
            object_id: Object ID

        Returns:
            RGB color (0-1 range)
        """
        # Use object_id as seed for consistent colors
        np.random.seed(object_id)
        return np.random.rand(3)
