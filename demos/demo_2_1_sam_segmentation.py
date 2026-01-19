#!/usr/bin/env python3
"""
Demo 2.1: SAM Segmentation

This demo demonstrates:
- Loading RGB images from Replica dataset
- Running SAM (Segment Anything Model) for class-agnostic segmentation
- Visualizing segmentation masks in Rerun
- Showing individual mask instances with colors
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

from memspace.dataset.replica_dataset import ReplicaDataset
from memspace.models.sam_wrapper import SAMWrapper
from memspace.utils import rerun_utils
from memspace.utils.mask_utils import merge_overlapping_masks, filter_masks_by_depth


def visualize_masks_overlay(image: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create visualization with colored masks overlaid on image

    Args:
        image: RGB image (H, W, 3), uint8
        masks: Binary masks (N, H, W), bool or uint8
        alpha: Transparency of masks

    Returns:
        Visualization image (H, W, 3), uint8
    """
    vis = image.copy()

    # Generate random colors for each mask
    np.random.seed(42)  # For consistent colors
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

    for idx, mask in enumerate(masks):
        color = colors[idx]

        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color

        # Blend with original image
        vis = cv2.addWeighted(vis, 1.0, colored_mask, alpha, 0)

    return vis


@hydra.main(config_path="../memspace/configs", config_name="demo_2_1", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 70)
    print("MemSpace Demo 2.1: SAM Segmentation")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        # Initialize with optional recording
        spawn = cfg.get('rerun_spawn', True)
        save_path = cfg.get('rerun_save_path', None)

        if save_path:
            # Save to RRD file
            rr.init("memspace/demo_2_1", spawn=spawn)
            rr.save(save_path)
            print(f"✓ Rerun recording to: {save_path}")
            if spawn:
                print("✓ Rerun viewer spawned")
        else:
            # Just spawn viewer
            rr.init("memspace/demo_2_1", spawn=spawn)
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
    print(f"🤖 Initializing SAM model: {cfg.model.model_type}")
    sam_model = SAMWrapper(
        model_type=cfg.model.model_type,
        device=cfg.device,
    )
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

    # Get segmentation settings
    seg_cfg = cfg.segmentation
    min_mask_area = seg_cfg.min_mask_area
    max_masks = seg_cfg.max_masks_per_frame

    print("🎬 Processing frames with SAM segmentation...")
    print(f"   Min mask area: {min_mask_area} pixels")
    print(f"   Max masks per frame: {max_masks}")
    print()

    total_masks = 0

    for frame_idx in range(len(dataset)):
        print(f"Frame {frame_idx:03d}:")

        # Load frame
        color, depth, intrinsics, pose = dataset[frame_idx]

        # Convert to numpy for SAM (expects uint8)
        color_np = color.cpu().numpy().astype(np.uint8)

        # Run SAM segmentation
        print(f"  Running SAM segmentation...")
        masks, boxes, scores = sam_model.generate_masks(
            color_np,
            min_mask_region_area=min_mask_area,
        )

        num_masks = len(masks)
        print(f"  Found {num_masks} masks (before merging)")

        if num_masks == 0:
            print(f"  ⚠  No masks found, skipping frame")
            print()
            continue

        # Merge overlapping masks to reduce over-segmentation
        if seg_cfg.get('merge_masks', True):
            print(f"  Merging overlapping masks...")
            masks, boxes, scores = merge_overlapping_masks(
                masks, boxes, scores,
                iou_threshold=seg_cfg.get('merge_iou_threshold', 0.5),
                containment_threshold=seg_cfg.get('merge_containment_threshold', 0.85),
            )
            num_masks_merged = len(masks)
            print(f"  After merging: {num_masks_merged} masks ({num_masks - num_masks_merged} merged)")
            num_masks = num_masks_merged

        # Filter by depth consistency (optional)
        if seg_cfg.get('filter_by_depth', False):
            print(f"  Filtering masks by depth consistency...")
            depth_np = depth.cpu().numpy()
            keep_mask = filter_masks_by_depth(
                masks, depth_np,
                max_depth_variance=seg_cfg.get('max_depth_variance', 0.5),
            )
            masks = masks[keep_mask]
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            num_masks_filtered = len(masks)
            print(f"  After depth filtering: {num_masks_filtered} masks ({num_masks - num_masks_filtered} removed)")
            num_masks = num_masks_filtered

        # Limit number of masks
        if num_masks > max_masks:
            print(f"  Limiting to top {max_masks} masks by score")
            # Sort by score descending
            sorted_idx = np.argsort(scores)[::-1][:max_masks]
            masks = masks[sorted_idx]
            boxes = boxes[sorted_idx]
            scores = scores[sorted_idx]
            num_masks = max_masks

        total_masks += num_masks

        # Log camera pose (needed for 2D images in 3D view)
        rerun_utils.log_camera_pose(
            "world/camera",
            pose,
            intrinsics,
            width=color_np.shape[1],
            height=color_np.shape[0],
        )

        # Log RGB image
        rerun_utils.log_rgb_image("world/camera/rgb", color)

        # Create and log mask visualization
        mask_vis = visualize_masks_overlay(color_np, masks, alpha=0.4)
        rerun_utils.log_rgb_image(
            "world/camera/segmentation_overlay",
            torch.from_numpy(mask_vis)
        )

        # Log individual masks with different colors
        for mask_idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # Create colored mask
            color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

            # Generate consistent color for this mask
            np.random.seed(42 + mask_idx)
            mask_color = np.random.randint(0, 255, size=3, dtype=np.uint8)

            color_mask[mask > 0] = mask_color

            # Log mask as image
            rr.log(
                f"world/masks/mask_{mask_idx:03d}",
                rr.Image(color_mask),
            )

        # Log mask statistics
        stats_text = f"""
## Frame {frame_idx}

**Segmentation Results:**
- Number of masks: {num_masks}
- Total pixels segmented: {sum(mask.sum() for mask in masks):,}
- Average mask size: {np.mean([mask.sum() for mask in masks]):.1f} pixels
- Score range: [{scores.min():.3f}, {scores.max():.3f}]

**Top 5 masks by size:**
"""
        mask_areas = [mask.sum() for mask in masks]
        top5_idx = np.argsort(mask_areas)[::-1][:5]
        for i, idx in enumerate(top5_idx, 1):
            stats_text += f"\n{i}. Mask {idx}: {mask_areas[idx]:,} pixels (score: {scores[idx]:.3f})"

        rr.log("world/stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))

        print(f"  Avg mask size: {np.mean(mask_areas):.1f} pixels")
        print()

    # Log completion summary
    completion_text = f"""
# Demo 2.1 Complete! ✓

Successfully segmented {len(dataset)} frames using SAM ({cfg.model.model_type}).

## Summary:
- **Total masks generated:** {total_masks}
- **Average masks per frame:** {total_masks / len(dataset):.1f}
- **Model:** {cfg.model.model_type}
- **Min mask area:** {min_mask_area} pixels

## What you're seeing:
- **RGB images**: Original camera frames
- **Segmentation overlay**: Colored masks overlaid on RGB
- **Individual masks**: Each mask shown separately with unique color

## Key Features:
- Class-agnostic segmentation (no predefined categories)
- Automatic mask generation (no manual prompts needed)
- Instance-level segmentation (each object gets separate mask)

## Next Steps:
Run demo_2_2 to extract CLIP embeddings for each mask!

---
*Frames processed: {len(dataset)}*
*Dataset: {dataset_path}*
    """

    rr.log("world/summary", rr.TextDocument(completion_text, media_type=rr.MediaType.MARKDOWN))

    print("=" * 70)
    print("✅ Demo 2.1 completed successfully!")
    print("=" * 70)
    print()
    print(f"📊 Segmented {total_masks} objects across {len(dataset)} frames")
    print(f"   Average: {total_masks / len(dataset):.1f} masks per frame")
    print()
    print("📊 Check the Rerun viewer:")
    print("   - Original RGB at 'world/camera/rgb'")
    print("   - Segmentation overlay at 'world/camera/segmentation_overlay'")
    print("   - Individual masks at 'world/masks/mask_XXX'")
    print()
    print("Next: Run demo_2_2_clip_embeddings.py to add semantic features")
    print()


if __name__ == "__main__":
    main()
