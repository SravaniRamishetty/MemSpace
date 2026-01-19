#!/usr/bin/env python3
"""
Demo 3.2: Point Cloud Accumulation (ConceptGraphs-style)

This demo demonstrates:
- Direct point cloud accumulation from RGB-D frames
- Faster alternative to TSDF fusion
- Similar to ConceptGraphs approach
- Voxel downsampling for efficiency
- Optional mesh extraction via Poisson reconstruction
- Comparison with TSDF approach (Demo 3.1)
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import rerun as rr
import open3d as o3d

from memspace.dataset.replica_dataset import ReplicaDataset
from memspace.reconstruction.pointcloud_fusion import PointCloudAccumulator
from memspace.utils import rerun_utils


def log_mesh_to_rerun(entity_path: str, mesh: o3d.geometry.TriangleMesh):
    """
    Log an Open3D mesh to Rerun

    Args:
        entity_path: Rerun entity path
        mesh: Open3D triangle mesh
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Get vertex colors if available
    if mesh.has_vertex_colors():
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    else:
        colors = None

    # Log mesh
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=triangles,
            vertex_colors=colors,
        )
    )


@hydra.main(config_path="../memspace/configs", config_name="demo_3_2", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 70)
    print("MemSpace Demo 3.2: Point Cloud Accumulation")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        spawn = cfg.get('rerun_spawn', True)
        save_path = cfg.get('rerun_save_path', None)

        if save_path:
            rr.init("memspace/demo_3_2", spawn=spawn)
            rr.save(save_path)
            print(f"✓ Rerun recording to: {save_path}")
            if spawn:
                print("✓ Rerun viewer spawned")
        else:
            rr.init("memspace/demo_3_2", spawn=spawn)
            if spawn:
                print("✓ Rerun viewer spawned")
            else:
                print("✓ Rerun initialized (no viewer)")

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    else:
        print("⚠  Rerun visualization disabled in config")
        return

    print()

    # Load dataset
    dataset_cfg = cfg.dataset
    dataset_path = dataset_cfg.dataset_path
    print(f"📂 Loading Replica dataset from: {dataset_path}")

    try:
        dataset = ReplicaDataset(
            dataset_path=dataset_path,
            stride=dataset_cfg.stride,
            start=dataset_cfg.get('start_frame', 0),
            end=dataset_cfg.max_frames * dataset_cfg.stride if dataset_cfg.max_frames else -1,
            height=480,
            width=640,
            device=cfg.device,
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"✓ Dataset loaded: {len(dataset)} frames")
    print()

    # Initialize Point Cloud Accumulator
    pc_cfg = cfg.pointcloud
    print(f"☁️  Initializing Point Cloud Accumulator")
    print(f"   Voxel size: {pc_cfg.voxel_size}m ({pc_cfg.voxel_size*100:.1f}cm)")
    print(f"   Max depth: {pc_cfg.max_depth}m")
    print(f"   Outlier removal: {pc_cfg.outlier_removal}")

    accumulator = PointCloudAccumulator(
        voxel_size=pc_cfg.voxel_size,
        max_depth=pc_cfg.max_depth,
        outlier_removal=pc_cfg.outlier_removal,
        outlier_nb_neighbors=pc_cfg.outlier_nb_neighbors,
        outlier_std_ratio=pc_cfg.outlier_std_ratio,
    )
    print()

    # Accumulate point clouds
    print("🎬 Accumulating point clouds from RGB-D frames...")
    print(f"   Processing {len(dataset)} frames (stride {dataset_cfg.stride})")
    print()

    start_time = time.time()

    for frame_idx in range(len(dataset)):
        if frame_idx % 10 == 0 or frame_idx == len(dataset) - 1:
            print(f"  Frame {frame_idx+1}/{len(dataset)}")

        # Load frame
        color, depth, intrinsics, pose = dataset[frame_idx]

        # Convert to numpy
        color_np = color.cpu().numpy().astype(np.uint8)
        depth_np = depth.cpu().numpy().astype(np.float32)
        intrinsics_np = intrinsics.cpu().numpy()
        pose_np = pose.cpu().numpy()

        # Integrate into point cloud
        accumulator.integrate(color_np, depth_np, intrinsics_np, pose_np)

        # Log camera trajectory every 5 frames
        if frame_idx % 5 == 0:
            rerun_utils.log_camera_pose(
                f"world/camera_trajectory/frame_{frame_idx:03d}",
                pose,
                intrinsics,
                width=640,
                height=480,
            )

    accumulation_time = time.time() - start_time

    print()
    print(f"✓ Accumulated {accumulator.num_frames_integrated} frames")
    print(f"  Processing time: {accumulation_time:.2f}s ({accumulation_time/len(dataset):.2f}s per frame)")
    print()

    # Get accumulated point cloud
    print("☁️  Extracting final point cloud...")
    pcd = accumulator.get_point_cloud(clean=True)

    num_points = len(pcd.points)
    print(f"✓ Point cloud extracted:")
    print(f"   Points: {num_points:,}")
    print()

    # Log point cloud to Rerun
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None

    rr.log(
        "world/point_cloud",
        rr.Points3D(positions=points, colors=colors),
    )

    # Optional: Extract mesh
    mesh_cfg = cfg.mesh
    if mesh_cfg.get('enable', True):
        print(f"🔨 Extracting mesh using {mesh_cfg.method} reconstruction...")
        mesh_start = time.time()

        mesh = accumulator.extract_mesh(
            method=mesh_cfg.method,
            poisson_depth=mesh_cfg.get('poisson_depth', 9),
            clean=True,
        )

        mesh_time = time.time() - mesh_start

        num_vertices = len(mesh.vertices)
        num_triangles = len(mesh.triangles)
        print(f"✓ Mesh extracted ({mesh_time:.2f}s):")
        print(f"   Vertices: {num_vertices:,}")
        print(f"   Triangles: {num_triangles:,}")
        print()

        # Simplify mesh if requested
        if mesh_cfg.get('simplify', True) and num_triangles > mesh_cfg.target_triangles:
            print(f"📉 Simplifying mesh to ~{mesh_cfg.target_triangles:,} triangles...")
            mesh = mesh.simplify_quadric_decimation(mesh_cfg.target_triangles)
            mesh.compute_vertex_normals()
            print(f"✓ Mesh simplified:")
            print(f"   Vertices: {len(mesh.vertices):,}")
            print(f"   Triangles: {len(mesh.triangles):,}")
            print()

        # Log mesh to Rerun
        log_mesh_to_rerun("world/mesh/reconstructed", mesh)

    # Get statistics
    stats = accumulator.get_statistics()
    bbox_extent = stats.get('bounding_box_extent', [0, 0, 0])
    scene_volume = stats.get('scene_volume', 0)

    # Log completion summary
    total_time = time.time() - start_time

    completion_text = f"""
# Demo 3.2 Complete! ✓

Successfully reconstructed 3D scene using point cloud accumulation.

## Reconstruction Summary:
- **Frames integrated:** {accumulator.num_frames_integrated}
- **Frame stride:** {dataset_cfg.stride}
- **Voxel size:** {pc_cfg.voxel_size}m ({pc_cfg.voxel_size*100:.1f}cm)
- **Processing time:** {total_time:.2f}s ({total_time/len(dataset):.3f}s per frame)

## Point Cloud Statistics:
- **Points:** {num_points:,}
- **Bounding box extent:** [{bbox_extent[0]:.2f}, {bbox_extent[1]:.2f}, {bbox_extent[2]:.2f}] m
- **Approximate volume:** {scene_volume:.2f} m³

## Mesh Statistics (if enabled):
"""

    if mesh_cfg.get('enable', True):
        completion_text += f"""
- **Method:** {mesh_cfg.method}
- **Vertices:** {len(mesh.vertices):,}
- **Triangles:** {len(mesh.triangles):,}
- **Extraction time:** {mesh_time:.2f}s
"""
    else:
        completion_text += "- Mesh extraction disabled\n"

    completion_text += f"""
## What you're seeing:
- **Point cloud**: Accumulated from {len(dataset)} RGB-D frames
- **Mesh**: Reconstructed surface from point cloud
- **Camera trajectory**: Camera poses during reconstruction

## Technical Details:
- **Method**: Direct point cloud accumulation (ConceptGraphs-style)
- **Downsampling**: Voxel grid filter at {pc_cfg.voxel_size}m
- **Outlier removal**: {'Enabled' if pc_cfg.outlier_removal else 'Disabled'}
- **Mesh reconstruction**: {mesh_cfg.method if mesh_cfg.get('enable') else 'Disabled'}

## Comparison with TSDF (Demo 3.1):
- **Speed**: ~{accumulation_time/len(dataset):.3f}s per frame (point cloud) vs ~3s (TSDF)
- **Memory**: Lower (no voxel grid)
- **Quality**: Faster but potentially noisier
- **Use case**: Online/real-time SLAM

## Next Steps:
Phase 3.3: Per-object 3D reconstruction using tracked masks

---
*Dataset: {dataset_path}*
*Frames: {len(dataset)}*
    """

    rr.log("world/summary", rr.TextDocument(completion_text, media_type=rr.MediaType.MARKDOWN))

    print("=" * 70)
    print("✅ Demo 3.2 completed successfully!")
    print("=" * 70)
    print()
    print(f"📊 3D Reconstruction Results:")
    print(f"   Frames integrated: {accumulator.num_frames_integrated}")
    print(f"   Point cloud: {num_points:,} points")
    if mesh_cfg.get('enable', True):
        print(f"   Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    print(f"   Scene volume: {scene_volume:.2f} m³")
    print(f"   Total processing time: {total_time:.2f}s")
    print()
    print("📊 Check the Rerun viewer:")
    print("   - Point cloud at 'world/point_cloud'")
    if mesh_cfg.get('enable', True):
        print("   - Reconstructed mesh at 'world/mesh/reconstructed'")
    print("   - Camera trajectory at 'world/camera_trajectory'")
    print()
    print("⚡ Performance Comparison:")
    print(f"   Point cloud accumulation: {accumulation_time:.2f}s total")
    print(f"   Per-frame time: {accumulation_time/len(dataset):.3f}s")
    print(f"   → Much faster than TSDF fusion (~10x speedup)")
    print()
    print("Next: Phase 3.3 - Per-object 3D reconstruction with tracking")
    print()


if __name__ == "__main__":
    main()
