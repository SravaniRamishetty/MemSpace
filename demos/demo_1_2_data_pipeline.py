#!/usr/bin/env python3
"""
Demo 1.2: RGB-D Data Pipeline

This demo demonstrates:
- Loading RGB-D data from Replica dataset
- Converting depth to 3D point cloud
- Visualizing RGB, depth, point cloud, and camera pose in Rerun
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import torch
import rerun as rr

from memspace.dataset.replica_dataset import ReplicaDataset
from memspace.utils import rerun_utils


@hydra.main(config_path="../memspace/configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 70)
    print("MemSpace Demo 1.2: RGB-D Data Pipeline")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        rr.init("memspace/demo_1_2", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
        print("✓ Rerun visualization initialized")
    else:
        print("⚠  Rerun visualization disabled in config")
        return

    # Load dataset configuration
    dataset_cfg = cfg.get('dataset', {})
    dataset_path = dataset_cfg.get('dataset_path', f"{cfg.data_root}/Replica/room0")
    stride = dataset_cfg.get('stride', 10)
    max_frames = dataset_cfg.get('max_frames', 10)

    print()
    print(f"📂 Loading Replica dataset from: {dataset_path}")
    print(f"   Stride: {stride}, Max frames: {max_frames}")
    print()

    # Create dataset
    try:
        dataset = ReplicaDataset(
            dataset_path=dataset_path,
            stride=stride,
            start=0,
            end=max_frames * stride if max_frames else -1,
            height=480,  # Resize to smaller resolution for faster processing
            width=640,
            device=cfg.device,
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   Make sure Replica dataset exists at: {dataset_path}")
        return

    print(f"✓ Dataset loaded: {len(dataset)} frames")
    print()

    # Get camera parameters
    cam_params = dataset.get_camera_params()
    print("📷 Camera parameters:")
    print(f"   Resolution: {cam_params['width']}x{cam_params['height']}")
    print(f"   Focal length: fx={cam_params['fx']:.1f}, fy={cam_params['fy']:.1f}")
    print(f"   Principal point: cx={cam_params['cx']:.1f}, cy={cam_params['cy']:.1f}")
    print()

    # Process frames
    print(f"🎬 Processing {len(dataset)} frames...")
    print()

    prev_pose = None

    for frame_idx in range(len(dataset)):
        # Load frame
        color, depth, intrinsics, pose = dataset[frame_idx]

        print(f"Frame {frame_idx:03d}:")
        print(f"  Color shape: {tuple(color.shape)}, dtype: {color.dtype}")
        print(f"  Depth shape: {tuple(depth.shape)}, dtype: {depth.dtype}, "
              f"range: [{depth.min():.2f}, {depth.max():.2f}] m")
        print(f"  Pose: {pose[:3, 3].tolist()}")

        # Log RGB image
        rerun_utils.log_rgb_image("world/camera/rgb", color)

        # Log depth image
        rerun_utils.log_depth_image("world/camera/depth", depth, meter=1.0)

        # Log camera pose
        rerun_utils.log_camera_pose(
            "world/camera",
            pose,
            intrinsics,
            cam_params['width'],
            cam_params['height']
        )

        # Log camera trajectory
        if prev_pose is not None:
            rerun_utils.log_camera_trajectory(
                f"world/camera_trajectory/segment_{frame_idx}",
                prev_pose,
                pose,
                color=[255, 0, 0]  # Red trajectory
            )
        prev_pose = pose.clone()

        # Convert depth to point cloud
        points, colors = rerun_utils.depth_to_point_cloud(
            depth,
            intrinsics,
            color=color,
            max_depth=5.0  # Only show points within 5 meters
        )

        print(f"  Point cloud: {points.shape[0]} points")

        # Log point cloud (transformed to world coordinates)
        points_homogeneous = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
        points_world = (pose @ points_homogeneous.T).T[:, :3]

        rerun_utils.log_point_cloud(
            "world/point_cloud",
            points_world,
            colors=colors
        )

        print()

    # Log completion message
    completion_text = f"""
# Demo 1.2 Complete! ✓

Successfully loaded and visualized {len(dataset)} frames from Replica dataset.

## What you're seeing:
- **RGB images**: Camera view at each frame
- **Depth images**: Depth maps in meters (darker = closer)
- **3D Point Cloud**: Reconstructed 3D points from RGB-D data
- **Camera trajectory**: Red line showing camera path (starts at origin)

## Controls:
- Use mouse to rotate, pan, and zoom the 3D view
- Use the timeline at the bottom to scrub through frames
- Toggle visibility of entities in the left panel

## Next Steps:
Run demo_2_1 to see object segmentation with SAM!

---
*Frame count: {len(dataset)}*
*Dataset: {dataset_path}*
    """

    rr.log("world/status", rr.TextDocument(completion_text, media_type=rr.MediaType.MARKDOWN))

    print("=" * 70)
    print("✅ Demo 1.2 completed successfully!")
    print("=" * 70)
    print()
    print("📊 Check the Rerun viewer to explore the data")
    print("   - RGB images at 'world/camera/rgb'")
    print("   - Depth images at 'world/camera/depth'")
    print("   - 3D point cloud at 'world/point_cloud'")
    print("   - Camera trajectory at 'world/camera_trajectory'")
    print()
    print("Next: Run demo_2_1_sam_segmentation.py for object detection")
    print()


if __name__ == "__main__":
    main()
