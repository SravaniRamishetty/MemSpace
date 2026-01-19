#!/usr/bin/env python3
"""
Demo 3.1: TSDF Fusion and 3D Reconstruction

This demo demonstrates:
- TSDF volumetric fusion of RGB-D frames
- Multi-view depth integration
- Mesh extraction using marching cubes
- Mesh simplification and cleaning
- 3D visualization in Rerun
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import rerun as rr
import open3d as o3d

from memspace.dataset.replica_dataset import ReplicaDataset
from memspace.reconstruction.tsdf_fusion import TSDFVolume
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


@hydra.main(config_path="../memspace/configs", config_name="demo_3_1", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 70)
    print("MemSpace Demo 3.1: TSDF Fusion and 3D Reconstruction")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        spawn = cfg.get('rerun_spawn', True)
        save_path = cfg.get('rerun_save_path', None)

        if save_path:
            rr.init("memspace/demo_3_1", spawn=spawn)
            rr.save(save_path)
            print(f"✓ Rerun recording to: {save_path}")
            if spawn:
                print("✓ Rerun viewer spawned")
        else:
            rr.init("memspace/demo_3_1", spawn=spawn)
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

    # Initialize TSDF Volume
    tsdf_cfg = cfg.tsdf
    print(f"🏗️  Initializing TSDF Volume")
    print(f"   Voxel size: {tsdf_cfg.voxel_size}m ({tsdf_cfg.voxel_size*100:.1f}cm)")
    print(f"   SDF truncation: {tsdf_cfg.sdf_trunc}m ({tsdf_cfg.sdf_trunc*100:.1f}cm)")
    print(f"   Color type: {tsdf_cfg.color_type}")

    tsdf = TSDFVolume(
        voxel_size=tsdf_cfg.voxel_size,
        sdf_trunc=tsdf_cfg.sdf_trunc,
    )
    print()

    # Integrate frames into TSDF
    print("🎬 Integrating RGB-D frames into TSDF volume...")
    print(f"   Processing {len(dataset)} frames (stride {dataset_cfg.stride})")
    print()

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

        # Integrate into TSDF
        tsdf.integrate(color_np, depth_np, intrinsics_np, pose_np)

        # Log camera trajectory every 5 frames
        if frame_idx % 5 == 0:
            rerun_utils.log_camera_pose(
                f"world/camera_trajectory/frame_{frame_idx:03d}",
                pose,
                intrinsics,
                width=640,
                height=480,
            )

    print()
    print(f"✓ Integrated {tsdf.num_frames_integrated} frames into TSDF volume")
    print()

    # Extract mesh
    print("🔨 Extracting triangle mesh using marching cubes...")
    mesh = tsdf.extract_mesh()

    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)
    print(f"✓ Mesh extracted:")
    print(f"   Vertices: {num_vertices:,}")
    print(f"   Triangles: {num_triangles:,}")
    print()

    # Mesh processing
    mesh_cfg = cfg.mesh
    mesh_original = mesh

    # Simplify mesh
    if mesh_cfg.get('simplify', True):
        target_triangles = mesh_cfg.target_triangles
        if len(mesh.triangles) > target_triangles:
            print(f"📉 Simplifying mesh to ~{target_triangles:,} triangles...")
            mesh = mesh.simplify_quadric_decimation(target_triangles)
            print(f"✓ Mesh simplified:")
            print(f"   Vertices: {len(mesh.vertices):,}")
            print(f"   Triangles: {len(mesh.triangles):,}")
            print()

    # Recompute normals
    mesh.compute_vertex_normals()

    # Log to Rerun
    print("📊 Logging mesh to Rerun...")

    # Log original mesh
    log_mesh_to_rerun("world/mesh/original", mesh_original)

    # Log processed mesh
    log_mesh_to_rerun("world/mesh/processed", mesh)

    # Also extract and log point cloud
    print("☁️  Extracting point cloud...")
    pcd = tsdf.extract_point_cloud()
    print(f"✓ Point cloud extracted: {len(pcd.points):,} points")

    # Log point cloud
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None

    rr.log(
        "world/point_cloud",
        rr.Points3D(positions=points, colors=colors),
    )
    print()

    # Compute mesh statistics
    mesh_bbox = mesh.get_axis_aligned_bounding_box()
    bbox_extent = mesh_bbox.get_extent()
    volume = bbox_extent[0] * bbox_extent[1] * bbox_extent[2]

    # Log completion summary
    completion_text = f"""
# Demo 3.1 Complete! ✓

Successfully reconstructed 3D scene using TSDF fusion.

## Reconstruction Summary:
- **Frames integrated:** {tsdf.num_frames_integrated}
- **Frame stride:** {dataset_cfg.stride}
- **Voxel size:** {tsdf_cfg.voxel_size}m ({tsdf_cfg.voxel_size*100:.1f}cm)
- **SDF truncation:** {tsdf_cfg.sdf_trunc}m ({tsdf_cfg.sdf_trunc*100:.1f}cm)

## Mesh Statistics:
- **Original mesh:**
  - Vertices: {num_vertices:,}
  - Triangles: {num_triangles:,}
- **Processed mesh:**
  - Vertices: {len(mesh.vertices):,}
  - Triangles: {len(mesh.triangles):,}
- **Bounding box extent:** [{bbox_extent[0]:.2f}, {bbox_extent[1]:.2f}, {bbox_extent[2]:.2f}] m
- **Approximate volume:** {volume:.2f} m³

## Point Cloud:
- **Points:** {len(pcd.points):,}

## What you're seeing:
- **Original mesh**: Unprocessed mesh from TSDF (may have outliers)
- **Processed mesh**: Cleaned and simplified mesh
- **Point cloud**: Dense point cloud from TSDF
- **Camera trajectory**: Camera poses during reconstruction

## Technical Details:
- **TSDF**: Truncated Signed Distance Function volumetric fusion
- **Marching Cubes**: Surface extraction from volume
- **Simplification**: Quadric decimation to ~{mesh_cfg.target_triangles:,} triangles

## Next Steps:
Phase 3.2: Per-object 3D reconstruction using tracked masks

---
*Dataset: {dataset_path}*
*Frames: {len(dataset)}*
    """

    rr.log("world/summary", rr.TextDocument(completion_text, media_type=rr.MediaType.MARKDOWN))

    print("=" * 70)
    print("✅ Demo 3.1 completed successfully!")
    print("=" * 70)
    print()
    print(f"📊 3D Reconstruction Results:")
    print(f"   Frames integrated: {tsdf.num_frames_integrated}")
    print(f"   Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    print(f"   Point cloud: {len(pcd.points):,} points")
    print(f"   Scene volume: {volume:.2f} m³")
    print()
    print("📊 Check the Rerun viewer:")
    print("   - Original mesh at 'world/mesh/original'")
    print("   - Processed mesh at 'world/mesh/processed'")
    print("   - Point cloud at 'world/point_cloud'")
    print("   - Camera trajectory at 'world/camera_trajectory'")
    print()
    print("Next: Phase 3.2 - Per-object 3D reconstruction")
    print()


if __name__ == "__main__":
    main()
