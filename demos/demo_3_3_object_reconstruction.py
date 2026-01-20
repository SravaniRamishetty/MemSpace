#!/usr/bin/env python3
"""
Demo 3.3: Per-Object 3D Reconstruction

This demo demonstrates:
- Integration of object tracking (Phase 2.3) with 3D reconstruction (Phase 3.2)
- Per-object point cloud accumulation using masks
- Tracking objects across frames while building their 3D models
- Visualizing each tracked object's 3D point cloud
- Complete pipeline: SAM → CLIP → Tracking → 3D Reconstruction
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

from memspace.dataset import get_dataset
from memspace.models.sam_wrapper import SAMWrapper
from memspace.models.clip_wrapper import CLIPWrapper
from memspace.scenegraph.object_tracker import ObjectTracker
from memspace.reconstruction.object_reconstruction import ObjectReconstructor
from memspace.utils import rerun_utils
from memspace.utils.mask_utils import merge_overlapping_masks


@hydra.main(config_path="../memspace/configs", config_name="demo_3_3", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 70)
    print("MemSpace Demo 3.3: Per-Object 3D Reconstruction")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        spawn = cfg.get('rerun_spawn', True)
        save_path = cfg.get('rerun_save_path', None)

        if save_path:
            rr.init("memspace/demo_3_3", spawn=spawn)
            rr.save(save_path)
            print(f"✓ Rerun recording to: {save_path}")
            if spawn:
                print("✓ Rerun viewer spawned")
        else:
            rr.init("memspace/demo_3_3", spawn=spawn)
            if spawn:
                print("✓ Rerun viewer spawned")
            else:
                print("✓ Rerun initialized (no viewer)")

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    else:
        print("⚠  Rerun visualization disabled in config")
        return

    print()

    # Initialize SAM model
    print(f"🤖 Initializing SAM model: {cfg.model.sam.model_type}")
    sam_model = SAMWrapper(
        model_type=cfg.model.sam.model_type,
        device=cfg.device,
    )
    print()

    # Initialize CLIP model
    print(f"🤖 Initializing CLIP model: {cfg.model.clip.model_name}")
    clip_model = CLIPWrapper(
        model_name=cfg.model.clip.model_name,
        pretrained=cfg.model.clip.pretrained,
        device=cfg.device,
    )
    print()

    # Initialize Object Tracker
    track_cfg = cfg.tracking
    print(f"🎯 Initializing Object Tracker")
    tracker = ObjectTracker(
        sim_threshold=track_cfg.sim_threshold,
        spatial_weight=track_cfg.spatial_weight,
        clip_weight=track_cfg.clip_weight,
        max_missing_frames=track_cfg.max_missing_frames,
        min_observations=track_cfg.min_observations,
    )
    print()

    # Initialize Object Reconstructor
    recon_cfg = cfg.object_reconstruction
    print(f"🏗️  Initializing Object Reconstructor")
    print(f"   Voxel size: {recon_cfg.voxel_size}m")
    print(f"   Min points per object: {recon_cfg.min_points}")
    reconstructor = ObjectReconstructor(
        voxel_size=recon_cfg.voxel_size,
        max_depth=recon_cfg.max_depth,
        min_points=recon_cfg.min_points,
        outlier_removal=recon_cfg.outlier_removal,
        outlier_nb_neighbors=recon_cfg.outlier_nb_neighbors,
        outlier_std_ratio=recon_cfg.outlier_std_ratio,
    )
    print()

    # Load dataset
    print(f"📂 Loading  dataset from: {cfg.dataset.dataset_path}")

    try:
        dataset = get_dataset(cfg.dataset, device=cfg.device)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"✓ Dataset loaded: {len(dataset)} frames")
    print()

    # Get settings
    seg_cfg = cfg.segmentation
    clip_cfg = cfg.clip_features

    print("🎬 Processing frames with tracking + 3D reconstruction...")
    print(f"   Frames: {len(dataset)}")
    print(f"   Stride: {cfg.dataset.stride}")
    print()

    start_time = time.time()

    for frame_idx in range(len(dataset)):
        if frame_idx % 10 == 0 or frame_idx == len(dataset) - 1:
            print(f"Frame {frame_idx+1}/{len(dataset)}")

        # Load frame
        color, depth, intrinsics, pose = dataset[frame_idx]
        color_np = color.cpu().numpy().astype(np.uint8)
        depth_np = depth.cpu().numpy().astype(np.float32)

        # Run SAM segmentation
        masks, boxes, scores = sam_model.generate_masks(
            color_np,
            min_mask_region_area=seg_cfg.min_mask_area,
        )

        if len(masks) == 0:
            # Update tracker with empty detections
            tracker.update(frame_idx, np.array([]), np.array([]), np.array([]), np.array([]))
            continue

        # Merge overlapping masks
        if seg_cfg.get('merge_masks', True):
            masks, boxes, scores = merge_overlapping_masks(
                masks, boxes, scores,
                iou_threshold=seg_cfg.get('merge_iou_threshold', 0.5),
                containment_threshold=seg_cfg.get('merge_containment_threshold', 0.85),
            )

        # Limit number of masks
        if len(masks) > seg_cfg.max_masks_per_frame:
            sorted_idx = np.argsort(scores)[::-1][:seg_cfg.max_masks_per_frame]
            masks = masks[sorted_idx]
            boxes = boxes[sorted_idx]
            scores = scores[sorted_idx]

        # Extract CLIP features
        crops, features = clip_model.extract_mask_features(
            color_np,
            boxes,
            padding=clip_cfg.padding,
            batch_size=clip_cfg.batch_size,
        )

        # Update tracker
        object_ids, match_types = tracker.update(
            frame_idx=frame_idx,
            masks=masks,
            bboxes=boxes,
            scores=scores,
            clip_features=features,
        )

        # Integrate into per-object 3D reconstruction
        reconstructor.integrate_frame(
            object_ids=object_ids,
            masks=masks,
            color=color_np,
            depth=depth_np,
            intrinsics=intrinsics.cpu().numpy(),
            extrinsics=pose.cpu().numpy(),
        )

        # Log camera trajectory every 5 frames
        if frame_idx % 5 == 0:
            rerun_utils.log_camera_pose(
                f"world/camera_trajectory/frame_{frame_idx:03d}",
                pose,
                intrinsics,
                width=640,
                height=480,
            )

    total_time = time.time() - start_time

    print()
    print(f"✓ Processed {len(dataset)} frames in {total_time:.2f}s ({total_time/len(dataset):.3f}s per frame)")
    print()

    # Get final statistics
    tracker_stats = tracker.get_statistics()
    recon_stats = reconstructor.get_statistics()

    print(f"📊 Tracking Statistics:")
    print(f"   Total objects tracked: {tracker_stats['total_objects']}")
    print(f"   Confirmed objects: {tracker_stats['confirmed_objects']}")
    print(f"   Active objects: {tracker_stats['active_objects']}")
    print()

    print(f"🏗️  Reconstruction Statistics:")
    print(f"   Objects with 3D data: {recon_stats['num_objects']}")

    # Get valid objects (with enough points)
    valid_objects = reconstructor.filter_objects_by_points(recon_cfg.min_points)
    print(f"   Valid objects (≥{recon_cfg.min_points} points): {len(valid_objects)}")
    print()

    # Visualize object point clouds
    vis_cfg = cfg.visualization
    if vis_cfg.show_individual_objects:
        print(f"☁️  Logging per-object point clouds to Rerun...")

        objects_to_show = valid_objects[:vis_cfg.max_objects_to_show]

        for obj_id in objects_to_show:
            pcd = reconstructor.get_object_cloud(obj_id, clean=True)

            if pcd is None or len(pcd.points) == 0:
                continue

            # Get object info from tracker
            obj = tracker.objects[obj_id]

            # Get consistent color for this object
            color_obj = obj.get_color()

            # Log point cloud
            points = np.asarray(pcd.points)
            colors = np.tile(color_obj, (len(points), 1))  # Use object color

            rr.log(
                f"world/objects/obj_{obj_id:03d}/pointcloud",
                rr.Points3D(positions=points, colors=colors),
            )

            # Log object statistics
            obj_stats = recon_stats['objects'][obj_id]
            stats_text = f"""
**Object {obj_id}**
- Points: {obj_stats['num_points']:,}
- Frames observed: {obj_stats['num_frames']}
- Tracking: {obj.num_observations} observations
- First seen: Frame {obj.first_seen_frame}
- Last seen: Frame {obj.last_seen_frame}
- Status: {obj.status.value}
"""
            if 'volume' in obj_stats:
                stats_text += f"- Volume: {obj_stats['volume']:.4f} m³\n"

            rr.log(
                f"world/objects/obj_{obj_id:03d}/info",
                rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN)
            )

        print(f"✓ Logged {len(objects_to_show)} object point clouds")
        print()

    # Log completion summary
    completion_text = f"""
# Demo 3.3 Complete! ✓

Successfully tracked and reconstructed {len(valid_objects)} objects in 3D.

## Pipeline Summary:
1. **SAM Segmentation**: Detected objects in each frame
2. **CLIP Embeddings**: Extracted visual features
3. **Object Tracking**: Associated detections across frames
4. **3D Reconstruction**: Built per-object point clouds

## Tracking Results:
- **Total objects tracked:** {tracker_stats['total_objects']}
- **Confirmed objects:** {tracker_stats['confirmed_objects']}
- **Total detections:** {tracker_stats['total_detections']}
- **Successful matches:** {tracker_stats['total_matches']}
- **Match rate:** {tracker_stats['total_matches']/tracker_stats['total_detections']*100:.1f}%

## 3D Reconstruction Results:
- **Objects with 3D data:** {recon_stats['num_objects']}
- **Valid objects:** {len(valid_objects)} (≥{recon_cfg.min_points} points)
- **Voxel size:** {recon_cfg.voxel_size}m

## Top Objects by Points:
"""

    # Sort objects by point count
    sorted_objects = sorted(
        [(obj_id, recon_stats['objects'][obj_id]) for obj_id in valid_objects],
        key=lambda x: x[1]['num_points'],
        reverse=True
    )

    for i, (obj_id, obj_stats) in enumerate(sorted_objects[:10], 1):
        obj = tracker.objects[obj_id]
        completion_text += f"""
{i}. **Object {obj_id}**: {obj_stats['num_points']:,} points, {obj_stats['num_frames']} frames
"""

    completion_text += f"""
## Performance:
- **Total time:** {total_time:.2f}s
- **Per-frame time:** {total_time/len(dataset):.3f}s
- **Frames processed:** {len(dataset)}

## What you're seeing:
- **Individual object point clouds**: Each tracked object in 3D
- **Camera trajectory**: Camera poses during capture
- **Object info**: Statistics for each object

## Key Features:
- Per-object 3D reconstruction
- Masked depth integration (only object pixels)
- Consistent object IDs across frames
- Real-time capable (~{total_time/len(dataset):.3f}s per frame)

## Next Steps:
- Object-level mesh extraction
- Semantic scene graphs
- Natural language queries

---
*Dataset: {cfg.dataset.dataset_path}*
*Frames: {len(dataset)}*
*Valid objects: {len(valid_objects)}*
    """

    rr.log("world/summary", rr.TextDocument(completion_text, media_type=rr.MediaType.MARKDOWN))

    print("=" * 70)
    print("✅ Demo 3.3 completed successfully!")
    print("=" * 70)
    print()
    print(f"📊 Final Results:")
    print(f"   Frames processed: {len(dataset)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Per-frame time: {total_time/len(dataset):.3f}s")
    print(f"   Objects tracked: {tracker_stats['total_objects']}")
    print(f"   Objects with 3D data: {len(valid_objects)}")
    print()
    print("📊 Check the Rerun viewer:")
    print("   - Object point clouds at 'world/objects/obj_XXX/pointcloud'")
    print("   - Object info at 'world/objects/obj_XXX/info'")
    print("   - Camera trajectory at 'world/camera_trajectory'")
    print()
    print("🎯 Complete pipeline demonstrated:")
    print("   SAM → CLIP → Tracking → 3D Reconstruction")
    print()


if __name__ == "__main__":
    main()
