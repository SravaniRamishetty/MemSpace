"""
Rerun visualization utilities
Adapted from ConceptGraphs optional_rerun_wrapper.py
"""

import numpy as np
import torch
import rerun as rr


def log_rgb_image(entity_path: str, image: torch.Tensor):
    """
    Log RGB image to Rerun

    Args:
        entity_path: Rerun entity path (e.g., "world/camera/rgb")
        image: RGB image tensor (H, W, 3), range [0, 255], float or uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Ensure uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    rr.log(entity_path, rr.Image(image))


def log_depth_image(entity_path: str, depth: torch.Tensor, meter: float = 1.0):
    """
    Log depth image to Rerun

    Args:
        entity_path: Rerun entity path (e.g., "world/camera/depth")
        depth: Depth image tensor (H, W), in meters
        meter: Meter scale factor (default 1.0)
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()

    # Ensure 2D
    if len(depth.shape) == 3:
        depth = depth.squeeze()

    assert len(depth.shape) == 2, f"Depth must be 2D, got shape {depth.shape}"

    rr.log(entity_path, rr.DepthImage(depth, meter=meter))


def log_camera_pose(
    entity_path: str,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
):
    """
    Log camera intrinsics and pose to Rerun

    Args:
        entity_path: Rerun entity path (e.g., "world/camera")
        pose: Camera-to-world pose matrix (4, 4)
        intrinsics: Camera intrinsics matrix (3, 3)
        width: Image width
        height: Image height
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()

    # Extract intrinsic parameters
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    # Log camera intrinsics
    rr.log(
        entity_path,
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
        ),
    )

    # Log camera pose (camera-to-world transform)
    # Rerun expects translation and rotation (quaternion)
    translation = pose[:3, 3].tolist()
    rotation_matrix = pose[:3, :3]

    # Convert rotation matrix to quaternion [x, y, z, w]
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    # Create rotation quaternion object
    rotation_quat = rr.Quaternion(xyzw=quaternion)

    rr.log(
        entity_path,
        rr.Transform3D(
            translation=translation, rotation=rotation_quat, from_parent=False
        ),
    )


def log_point_cloud(
    entity_path: str,
    points: torch.Tensor,
    colors: torch.Tensor = None,
    labels: list = None,
):
    """
    Log point cloud to Rerun

    Args:
        entity_path: Rerun entity path (e.g., "world/pointcloud")
        points: Point positions (N, 3)
        colors: Point colors (N, 3), range [0, 255] or [0, 1]
        labels: Point labels (list of N strings)
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()

        # Ensure colors are uint8 in range [0, 255]
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    rr.log(entity_path, rr.Points3D(points, colors=colors, labels=labels))


def log_camera_trajectory(
    entity_path: str,
    prev_pose: torch.Tensor,
    curr_pose: torch.Tensor,
    color: list = [255, 0, 0],
):
    """
    Log camera trajectory line between two poses

    Args:
        entity_path: Rerun entity path (e.g., "world/camera_trajectory/frame_001")
        prev_pose: Previous camera pose (4, 4)
        curr_pose: Current camera pose (4, 4)
        color: Line color [R, G, B]
    """
    if isinstance(prev_pose, torch.Tensor):
        prev_pose = prev_pose.cpu().numpy()
    if isinstance(curr_pose, torch.Tensor):
        curr_pose = curr_pose.cpu().numpy()

    prev_translation = prev_pose[:3, 3].tolist()
    curr_translation = curr_pose[:3, 3].tolist()

    rr.log(
        entity_path,
        rr.LineStrips3D(
            [np.vstack([prev_translation, curr_translation]).tolist()],
            colors=[color],
        ),
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> list:
    """
    Convert 3x3 rotation matrix to quaternion [x, y, z, w]

    Args:
        R: Rotation matrix (3, 3)

    Returns:
        Quaternion [x, y, z, w]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return [x, y, z, w]


def depth_to_point_cloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    color: torch.Tensor = None,
    max_depth: float = 10.0,
) -> tuple:
    """
    Convert depth image to point cloud

    Args:
        depth: Depth image (H, W) in meters
        intrinsics: Camera intrinsics matrix (3, 3)
        color: RGB image (H, W, 3), range [0, 255]
        max_depth: Maximum depth to include (meters)

    Returns:
        points: Point positions (N, 3)
        colors: Point colors (N, 3), None if color not provided
    """
    device = depth.device
    H, W = depth.shape

    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # Get camera intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Back-project to 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to (H, W, 3)
    points = torch.stack([x, y, z], dim=-1)

    # Filter by valid depth
    valid_mask = (z > 0) & (z < max_depth)

    # Reshape and filter
    points = points[valid_mask]  # (N, 3)

    if color is not None:
        colors = color[valid_mask]  # (N, 3)
        return points, colors
    else:
        return points, None
