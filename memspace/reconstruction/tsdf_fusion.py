#!/usr/bin/env python3
"""
TSDF (Truncated Signed Distance Function) Fusion for 3D Reconstruction

This module provides TSDF-based volumetric fusion for RGB-D data:
- Integrates depth frames from multiple viewpoints
- Builds a volumetric representation of the scene
- Extracts triangle meshes using marching cubes
- Supports both regular and scalable TSDF volumes
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple
import torch


class TSDFVolume:
    """
    TSDF Volume for volumetric fusion of RGB-D frames

    This class wraps Open3D's ScalableTSDFVolume to provide:
    - Multi-view depth integration
    - Color integration from RGB
    - Mesh extraction via marching cubes
    - Voxel-based scene representation
    """

    def __init__(
        self,
        voxel_size: float = 0.01,  # 1cm voxels
        sdf_trunc: float = 0.04,   # 4cm truncation distance
        color_type: o3d.pipelines.integration.TSDFVolumeColorType =
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    ):
        """
        Initialize TSDF Volume

        Args:
            voxel_size: Size of each voxel in meters (smaller = higher resolution)
            sdf_trunc: Truncation distance for SDF in meters
                      (should be ~2-4x voxel_size)
            color_type: Color integration type (RGB8 or Gray32)
        """
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.color_type = color_type

        # Create scalable TSDF volume (grows dynamically)
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=color_type,
        )

        self.num_frames_integrated = 0

    def integrate(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        """
        Integrate an RGB-D frame into the TSDF volume

        Args:
            color: RGB image (H, W, 3), uint8, range [0, 255]
            depth: Depth image (H, W), float32, in meters
            intrinsics: Camera intrinsics (3, 3) or (4, 4)
            extrinsics: Camera pose (4, 4) - world-to-camera transform
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
            depth_scale=1.0,  # Depth already in meters
            depth_trunc=10.0,  # Ignore depth > 10m
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

        # Integrate into volume
        # Note: Open3D expects camera-to-world transform (inverse of extrinsics)
        # If extrinsics is world-to-camera, invert it
        # Assuming extrinsics is camera-to-world (pose), use directly
        self.volume.integrate(
            rgbd,
            o3d_intrinsics,
            extrinsics,  # Camera-to-world transform
        )

        self.num_frames_integrated += 1

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """
        Extract triangle mesh from TSDF volume using marching cubes

        Returns:
            Triangle mesh with vertices, triangles, colors, and normals
        """
        mesh = self.volume.extract_triangle_mesh()

        # Compute vertex normals for better visualization
        mesh.compute_vertex_normals()

        return mesh

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Extract point cloud from TSDF volume

        Returns:
            Point cloud with points, colors, and normals
        """
        pcd = self.volume.extract_point_cloud()
        return pcd

    def get_statistics(self) -> dict:
        """
        Get statistics about the TSDF volume

        Returns:
            Dictionary with volume statistics
        """
        return {
            'num_frames_integrated': self.num_frames_integrated,
            'voxel_size': self.voxel_size,
            'sdf_trunc': self.sdf_trunc,
        }

    def reset(self):
        """Reset the TSDF volume (clear all data)"""
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=self.color_type,
        )
        self.num_frames_integrated = 0


def create_mesh_from_rgbd_frames(
    color_images: list,
    depth_images: list,
    intrinsics_list: list,
    poses: list,
    voxel_size: float = 0.01,
    sdf_trunc: float = 0.04,
) -> Tuple[o3d.geometry.TriangleMesh, TSDFVolume]:
    """
    Convenience function to create mesh from multiple RGB-D frames

    Args:
        color_images: List of RGB images
        depth_images: List of depth images
        intrinsics_list: List of camera intrinsics (can be same for all)
        poses: List of camera poses
        voxel_size: TSDF voxel size
        sdf_trunc: TSDF truncation distance

    Returns:
        Tuple of (mesh, tsdf_volume)
    """
    tsdf = TSDFVolume(voxel_size=voxel_size, sdf_trunc=sdf_trunc)

    # Handle case where intrinsics is the same for all frames
    if not isinstance(intrinsics_list, list):
        intrinsics_list = [intrinsics_list] * len(color_images)

    for color, depth, intrinsics, pose in zip(
        color_images, depth_images, intrinsics_list, poses
    ):
        tsdf.integrate(color, depth, intrinsics, pose)

    mesh = tsdf.extract_mesh()

    return mesh, tsdf
