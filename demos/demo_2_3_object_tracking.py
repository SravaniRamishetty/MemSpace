#!/usr/bin/env python3
"""
Demo 2.3: Multi-frame Object Tracking

This demo demonstrates:
- Running SAM segmentation across multiple frames
- Extracting CLIP embeddings for each mask
- Tracking objects across frames using spatial + semantic similarity
- Visualizing object tracks with persistent IDs in Rerun
- Managing object lifecycle (Active/Missing/Inactive)
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
import cv2
from typing import List

from memspace.dataset import get_dataset
from memspace.models.sam_wrapper import SAMWrapper
from memspace.models.clip_wrapper import CLIPWrapper
from memspace.scenegraph.object_tracker import ObjectTracker
from memspace.scenegraph.object_instance import ObjectStatus
from memspace.utils import rerun_utils
from memspace.utils.mask_utils import merge_overlapping_masks


def visualize_tracked_masks(
    image: np.ndarray,
    masks: np.ndarray,
    object_ids: List[int],
    tracker: ObjectTracker,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create visualization with tracked masks colored by object ID

    Args:
        image: RGB image (H, W, 3), uint8
        masks: Binary masks (N, H, W)
        object_ids: Object IDs for each mask
        tracker: ObjectTracker instance
        alpha: Transparency of masks

    Returns:
        Visualization image (H, W, 3), uint8
    """
    vis = image.copy()

    for mask, obj_id in zip(masks, object_ids):
        # Get object from tracker
        obj = tracker.objects[obj_id] if obj_id < len(tracker.objects) else None

        if obj is None:
            continue

        # Get consistent color based on object ID
        color = (obj.get_color() * 255).astype(np.uint8)

        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color

        # Blend with original image
        vis = cv2.addWeighted(vis, 1.0, colored_mask, alpha, 0)

        # Draw bounding box
        bbox = obj.current_bbox.astype(int)
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color.tolist(), 2)

        # Draw object ID and status
        label = f"ID:{obj_id}"
        if obj.is_missing():
            label += " (MISSING)"
        cv2.putText(vis, label, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

    return vis


@hydra.main(config_path="../memspace/configs", config_name="demo_2_3", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 70)
    print("MemSpace Demo 2.3: Multi-frame Object Tracking")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        # Initialize with optional recording
        spawn = cfg.get('rerun_spawn', True)
        save_path = cfg.get('rerun_save_path', None)

        if save_path:
            # Save to RRD file
            rr.init("memspace/demo_2_3", spawn=spawn)
            rr.save(save_path)
            print(f"✓ Rerun recording to: {save_path}")
            if spawn:
                print("✓ Rerun viewer spawned")
        else:
            # Just spawn viewer
            rr.init("memspace/demo_2_3", spawn=spawn)
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
    print(f"   Similarity threshold: {track_cfg.sim_threshold}")
    print(f"   Spatial weight: {track_cfg.spatial_weight}")
    print(f"   CLIP weight: {track_cfg.clip_weight}")
    print(f"   Max missing frames: {track_cfg.max_missing_frames}")
    tracker = ObjectTracker(
        sim_threshold=track_cfg.sim_threshold,
        spatial_weight=track_cfg.spatial_weight,
        clip_weight=track_cfg.clip_weight,
        max_missing_frames=track_cfg.max_missing_frames,
        min_observations=track_cfg.min_observations,
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

    print("🎬 Processing frames with tracking...")
    print(f"   Frames: {len(dataset)}")
    print(f"   Stride: {cfg.dataset.stride}")
    print()

    for frame_idx in range(len(dataset)):
        print(f"Frame {frame_idx:03d}:")

        # Load frame
        color, depth, intrinsics, pose = dataset[frame_idx]
        color_np = color.cpu().numpy().astype(np.uint8)

        # Run SAM segmentation
        print(f"  Running SAM segmentation...")
        masks, boxes, scores = sam_model.generate_masks(
            color_np,
            min_mask_region_area=seg_cfg.min_mask_area,
        )

        num_masks_initial = len(masks)
        print(f"  Found {num_masks_initial} masks (before merging)")

        if num_masks_initial == 0:
            print(f"  ⚠  No masks found, marking all objects as missing")
            # Update tracker with empty detections
            tracker.update(frame_idx, np.array([]), np.array([]), np.array([]), np.array([]))
            print()
            continue

        # Merge overlapping masks
        if seg_cfg.get('merge_masks', True):
            masks, boxes, scores = merge_overlapping_masks(
                masks, boxes, scores,
                iou_threshold=seg_cfg.get('merge_iou_threshold', 0.5),
                containment_threshold=seg_cfg.get('merge_containment_threshold', 0.85),
            )
            num_masks = len(masks)
            print(f"  After merging: {num_masks} masks ({num_masks_initial - num_masks} merged)")

        # Limit number of masks
        if num_masks > seg_cfg.max_masks_per_frame:
            sorted_idx = np.argsort(scores)[::-1][:seg_cfg.max_masks_per_frame]
            masks = masks[sorted_idx]
            boxes = boxes[sorted_idx]
            scores = scores[sorted_idx]
            num_masks = seg_cfg.max_masks_per_frame

        # Extract CLIP features
        print(f"  Extracting CLIP features for {num_masks} masks...")
        crops, features = clip_model.extract_mask_features(
            color_np,
            boxes,
            padding=clip_cfg.padding,
            batch_size=clip_cfg.batch_size,
        )

        # Update tracker
        print(f"  Updating tracker...")
        object_ids, match_types = tracker.update(
            frame_idx=frame_idx,
            masks=masks,
            bboxes=boxes,
            scores=scores,
            clip_features=features,
        )

        # Count matches
        num_matched = sum(match_types)
        num_new = len(match_types) - num_matched

        print(f"  Tracked: {num_matched} matched, {num_new} new")

        # Get tracker statistics
        stats = tracker.get_statistics()
        print(f"  Objects: {stats['active_objects']} active, "
              f"{stats['total_objects']} total, "
              f"{stats['confirmed_objects']} confirmed")

        # Log camera pose (needed for 2D images in 3D view)
        rerun_utils.log_camera_pose(
            "world/camera",
            pose,
            intrinsics,
            width=color_np.shape[1],
            height=color_np.shape[0],
        )

        # Log to Rerun
        rerun_utils.log_rgb_image("world/camera/rgb", color)

        # Create tracked visualization
        tracked_vis = visualize_tracked_masks(color_np, masks, object_ids, tracker, alpha=0.4)
        rerun_utils.log_rgb_image(
            "world/camera/tracking_overlay",
            torch.from_numpy(tracked_vis)
        )

        # Log individual tracked objects
        for mask_idx, (mask, obj_id, match_type) in enumerate(zip(masks, object_ids, match_types)):
            obj = tracker.objects[obj_id]
            color_obj = (obj.get_color() * 255).astype(np.uint8)

            # Create colored mask
            color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            color_mask[mask > 0] = color_obj

            # Log mask
            rr.log(
                f"world/tracked_objects/obj_{obj_id:03d}",
                rr.Image(color_mask),
            )

        # Log tracking statistics
        track_stats_text = f"""
## Frame {frame_idx} - Tracking Statistics

**Detections:**
- Total masks: {num_masks}
- Matched to existing: {num_matched}
- New objects: {num_new}

**Tracker State:**
- Total objects: {stats['total_objects']}
- Active objects: {stats['active_objects']}
- Confirmed objects: {stats['confirmed_objects']}
- Total matches: {stats['total_matches']}

**Object Details:**
"""
        # Add details for each active object
        active_objects = tracker.get_active_objects()
        for obj in active_objects[:10]:  # Limit to top 10
            track_stats_text += f"\n- **ID {obj.object_id}**: {obj.num_observations} obs, age {obj.get_age()}"

        rr.log("world/tracking_stats", rr.TextDocument(track_stats_text, media_type=rr.MediaType.MARKDOWN))

        print()

    # Final summary
    final_stats = tracker.get_statistics()

    completion_text = f"""
# Demo 2.3 Complete! ✓

Successfully tracked objects across {len(dataset)} frames.

## Tracking Summary:
- **Total objects tracked:** {final_stats['total_objects']}
- **Confirmed objects:** {final_stats['confirmed_objects']} (≥{track_cfg.min_observations} observations)
- **Active objects:** {final_stats['active_objects']}
- **Total detections:** {final_stats['total_detections']}
- **Successful matches:** {final_stats['total_matches']}
- **Match rate:** {final_stats['total_matches']/final_stats['total_detections']*100:.1f}%

## Tracking Configuration:
- **Similarity threshold:** {track_cfg.sim_threshold}
- **Spatial weight:** {track_cfg.spatial_weight}
- **CLIP weight:** {track_cfg.clip_weight}
- **Max missing frames:** {track_cfg.max_missing_frames}

## Object Lifecycle:
- **Active**: Currently being observed
- **Missing**: Not seen for <{track_cfg.max_missing_frames} frames
- **Inactive**: Lost (≥{track_cfg.max_missing_frames} frames missing)

## Top Objects by Observations:
"""

    # Sort objects by observations
    sorted_objects = sorted(tracker.get_all_objects(), key=lambda x: x.num_observations, reverse=True)
    for i, obj in enumerate(sorted_objects[:10], 1):
        completion_text += f"\n{i}. **ID {obj.object_id}**: {obj.num_observations} observations "
        completion_text += f"(frames {obj.first_seen_frame}-{obj.last_seen_frame}, "
        completion_text += f"status: {obj.status.value})"

    completion_text += f"""

## What you're seeing:
- **RGB images**: Original camera frames
- **Tracking overlay**: Masks colored by object ID with bounding boxes
- **Individual tracked objects**: Each object with persistent ID
- **Tracking statistics**: Per-frame and overall statistics

## Key Features:
- Multi-frame object association
- Spatial + semantic similarity matching
- Object lifecycle management (Active/Missing/Inactive)
- Persistent object IDs across frames

## Next Steps:
Phase 3.1: 3D reconstruction (TSDF fusion, mesh extraction)

---
*Frames processed: {len(dataset)}*
*Dataset: {cfg.dataset.dataset_path}*
    """

    rr.log("world/summary", rr.TextDocument(completion_text, media_type=rr.MediaType.MARKDOWN))

    print("=" * 70)
    print("✅ Demo 2.3 completed successfully!")
    print("=" * 70)
    print()
    print(f"📊 Tracking Results:")
    print(f"   Total objects: {final_stats['total_objects']}")
    print(f"   Confirmed objects: {final_stats['confirmed_objects']}")
    print(f"   Active objects: {final_stats['active_objects']}")
    print(f"   Total detections: {final_stats['total_detections']}")
    print(f"   Successful matches: {final_stats['total_matches']}")
    print(f"   Match rate: {final_stats['total_matches']/final_stats['total_detections']*100:.1f}%")
    print()
    print("📊 Check the Rerun viewer:")
    print("   - Original RGB at 'world/camera/rgb'")
    print("   - Tracking overlay at 'world/camera/tracking_overlay'")
    print("   - Tracked objects at 'world/tracked_objects/obj_XXX'")
    print()
    print("Next: Phase 3.1 - 3D reconstruction with TSDF fusion")
    print()


if __name__ == "__main__":
    main()
