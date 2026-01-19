#!/usr/bin/env python3
"""
Point Cloud Accumulation for 3D Reconstruction

This module provides point cloud-based reconstruction similar to ConceptGraphs:
- Accumulates point clouds from multiple RGB-D frames
- Transforms points to world coordinates using camera poses
- Merges and downsamples for efficiency
- Faster and more memory-efficient than TSDF
- Suitable for online/real-time applications
"""

import numpy as np
import open3d as o3d
from typing import Optional, List
import torch


class PointCloudAccumulator:
    """
    Point Cloud Accumulator for multi-view 3D reconstruction

    This class provides a simpler, faster alternative to TSDF fusion:
    - Direct point cloud accumulation (no volumetric grid)
    - Transform points to world coordinates
    - Voxel downsampling to control density
    - Statistical outlier removal
    - Suitable for real-time applications
    """

    def __init__(
        self,
        voxel_size: float = 0.02,  # 2cm downsampling
        max_depth: float = 10.0,   # Ignore depth > 10m
        outlier_removal: bool = True,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
    ):
        """
        Initialize Point Cloud Accumulator

        Args:
            voxel_size: Voxel size for downsampling (larger = faster, less dense)
            max_depth: Maximum depth to consider (meters)
            outlier_removal: Whether to remove statistical outliers
            outlier_nb_neighbors: Number of neighbors for outlier detection
            outlier_std_ratio: Standard deviation ratio for outlier detection
        """
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        self.outlier_removal = outlier_removal
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio

        # Accumulated point cloud
        self.global_pcd = o3d.geometry.PointCloud()
        self.num_frames_integrated = 0

    def integrate(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        depth_scale: float = 1.0,
    ):
        """
        Integrate an RGB-D frame into the global point cloud

        Args:
            color: RGB image (H, W, 3), uint8, range [0, 255]
            depth: Depth image (H, W), float32, in meters
            intrinsics: Camera intrinsics (3, 3) or (4, 4)
            extrinsics: Camera pose (4, 4) - camera-to-world transform
            depth_scale: Scale factor for depth (default 1.0 for meters)
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

        # Ensure correct types
        color = color.astype(np.uint8)
        depth = depth.astype(np.float32)

        # Handle 2D depth (add channel dimension if needed)
        if depth.ndim == 2:
            depth = depth[..., None]  # (H, W) -> (H, W, 1)

        # Create Open3D images
        o3d_color = o3d.geometry.Image(color)
        o3d_depth = o3d.geometry.Image(depth)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=depth_scale,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False,
        )

        # Create camera intrinsics
        height, width = depth.shape[:2]

        # Handle 3x3 or 4x4 intrinsics
        if intrinsics.shape == (3, 3):
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        elif intrinsics.shape == (4, 4):
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        else:
            raise ValueError(f"Invalid intrinsics shape: {intrinsics.shape}")

        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d_intrinsics,
        )

        # Transform to world coordinates
        # extrinsics is camera-to-world transform
        pcd.transform(extrinsics)

        # Downsample current frame point cloud
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.voxel_size)

        # Merge with global point cloud
        self.global_pcd += pcd

        # Downsample global point cloud to keep size manageable
        if self.voxel_size > 0:
            self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)

        self.num_frames_integrated += 1

    def get_point_cloud(self, clean: bool = True) -> o3d.geometry.PointCloud:
        """
        Get the accumulated point cloud

        Args:
            clean: Whether to apply outlier removal

        Returns:
            Point cloud with points, colors, and optionally normals
        """
        pcd = self.global_pcd

        if clean and self.outlier_removal and len(pcd.points) > 0:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio,
            )

        # Compute normals if not present
        if len(pcd.points) > 0 and not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2, max_nn=30
                )
            )

        return pcd

    def extract_mesh(
        self,
        method: str = 'poisson',
        poisson_depth: int = 9,
        clean: bool = True,
    ) -> o3d.geometry.TriangleMesh:
        """
        Extract mesh from accumulated point cloud

        Args:
            method: Mesh reconstruction method ('poisson' or 'ball_pivoting')
            poisson_depth: Octree depth for Poisson reconstruction (higher = more detail)
            clean: Whether to clean point cloud before meshing

        Returns:
            Triangle mesh
        """
        pcd = self.get_point_cloud(clean=clean)

        if len(pcd.points) == 0:
            return o3d.geometry.TriangleMesh()

        if method == 'poisson':
            # Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth
            )

            # Remove low-density vertices (artifacts)
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

        elif method == 'ball_pivoting':
            # Ball pivoting algorithm
            radii = [self.voxel_size * 2, self.voxel_size * 4, self.voxel_size * 8]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError(f"Unknown meshing method: {method}")

        # Compute normals
        mesh.compute_vertex_normals()

        return mesh

    def get_statistics(self) -> dict:
        """
        Get statistics about the accumulated point cloud

        Returns:
            Dictionary with statistics
        """
        num_points = len(self.global_pcd.points)

        stats = {
            'num_frames_integrated': self.num_frames_integrated,
            'num_points': num_points,
            'voxel_size': self.voxel_size,
            'max_depth': self.max_depth,
        }

        if num_points > 0:
            bbox = self.global_pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            stats['bounding_box_extent'] = extent.tolist()
            stats['scene_volume'] = extent[0] * extent[1] * extent[2]

        return stats

    def reset(self):
        """Reset the accumulator (clear all data)"""
        self.global_pcd = o3d.geometry.PointCloud()
        self.num_frames_integrated = 0


def create_pointcloud_from_rgbd_frames(
    color_images: list,
    depth_images: list,
    intrinsics_list: list,
    poses: list,
    voxel_size: float = 0.02,
    max_depth: float = 10.0,
) -> o3d.geometry.PointCloud:
    """
    Convenience function to create point cloud from multiple RGB-D frames

    Args:
        color_images: List of RGB images
        depth_images: List of depth images
        intrinsics_list: List of camera intrinsics (can be same for all)
        poses: List of camera poses
        voxel_size: Downsampling voxel size
        max_depth: Maximum depth threshold

    Returns:
        Accumulated point cloud
    """
    accumulator = PointCloudAccumulator(
        voxel_size=voxel_size,
        max_depth=max_depth,
    )

    # Handle case where intrinsics is the same for all frames
    if not isinstance(intrinsics_list, list):
        intrinsics_list = [intrinsics_list] * len(color_images)

    for color, depth, intrinsics, pose in zip(
        color_images, depth_images, intrinsics_list, poses
    ):
        accumulator.integrate(color, depth, intrinsics, pose)

    return accumulator.get_point_cloud()
