#!/usr/bin/env python3
"""
Demo 2.2: CLIP Embeddings

This demo demonstrates:
- Running SAM segmentation (from Demo 2.1)
- Extracting CLIP embeddings for each segmented mask
- Visualizing crops and embeddings in Rerun
- Computing embedding similarity for semantic grouping
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from memspace.dataset import get_dataset
from memspace.models.sam_wrapper import SAMWrapper
from memspace.models.clip_wrapper import CLIPWrapper
from memspace.utils import rerun_utils
from memspace.utils.mask_utils import merge_overlapping_masks


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


@hydra.main(config_path="../memspace/configs", config_name="demo_2_2", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 70)
    print("MemSpace Demo 2.2: CLIP Embeddings")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        # Initialize with optional recording
        spawn = cfg.get('rerun_spawn', True)
        save_path = cfg.get('rerun_save_path', None)

        if save_path:
            # Save to RRD file
            rr.init("memspace/demo_2_2", spawn=spawn)
            rr.save(save_path)
            print(f"✓ Rerun recording to: {save_path}")
            if spawn:
                print("✓ Rerun viewer spawned")
        else:
            # Just spawn viewer
            rr.init("memspace/demo_2_2", spawn=spawn)
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
    print(f"🤖 Initializing CLIP model: {cfg.model.clip.model_name} ({cfg.model.clip.pretrained})")
    clip_model = CLIPWrapper(
        model_name=cfg.model.clip.model_name,
        pretrained=cfg.model.clip.pretrained,
        device=cfg.device,
    )
    embedding_dim = clip_model.get_embedding_dim()
    print(f"   Embedding dimension: {embedding_dim}")
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
    min_mask_area = seg_cfg.min_mask_area
    max_masks = seg_cfg.max_masks_per_frame

    print("🎬 Processing frames with SAM + CLIP...")
    print(f"   Min mask area: {min_mask_area} pixels")
    print(f"   Max masks per frame: {max_masks}")
    print(f"   CLIP padding: {clip_cfg.padding} pixels")
    print()

    total_masks = 0
    all_embeddings = []
    all_mask_ids = []

    for frame_idx in range(len(dataset)):
        print(f"Frame {frame_idx:03d}:")

        # Load frame
        color, depth, intrinsics, pose = dataset[frame_idx]

        # Convert to numpy for SAM and CLIP (expects uint8)
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

        # Merge overlapping masks
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

        # Limit number of masks
        if num_masks > max_masks:
            print(f"  Limiting to top {max_masks} masks by score")
            sorted_idx = np.argsort(scores)[::-1][:max_masks]
            masks = masks[sorted_idx]
            boxes = boxes[sorted_idx]
            scores = scores[sorted_idx]
            num_masks = max_masks

        # Extract CLIP features
        print(f"  Extracting CLIP features for {num_masks} masks...")
        crops, features = clip_model.extract_mask_features(
            color_np,
            boxes,
            padding=clip_cfg.padding,
            batch_size=clip_cfg.batch_size,
        )
        print(f"  ✓ Extracted {len(features)} embeddings of dimension {embedding_dim}")

        total_masks += num_masks

        # Store embeddings for later visualization
        all_embeddings.append(features)
        all_mask_ids.extend([(frame_idx, i) for i in range(num_masks)])

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

        # Log individual mask crops with CLIP features
        for mask_idx, (mask, box, score, crop, feat) in enumerate(zip(masks, boxes, scores, crops, features)):
            # Log mask
            color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            np.random.seed(42 + mask_idx)
            mask_color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            color_mask[mask > 0] = mask_color

            rr.log(
                f"world/masks/mask_{mask_idx:03d}",
                rr.Image(color_mask),
            )

            # Log crop
            crop_np = np.array(crop)
            rr.log(
                f"world/crops/crop_{mask_idx:03d}",
                rr.Image(crop_np),
            )

        # Compute pairwise similarities within frame
        if num_masks > 1:
            # Normalize features (should already be normalized)
            features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

            # Compute similarity matrix
            similarity_matrix = features_norm @ features_norm.T

            # Find most similar pairs
            similarity_text = f"\n## Frame {frame_idx} - Mask Similarities\n\n"
            for i in range(num_masks):
                for j in range(i + 1, num_masks):
                    sim = similarity_matrix[i, j]
                    if sim > 0.7:  # Only show high similarities
                        similarity_text += f"- Mask {i} ↔ Mask {j}: {sim:.3f}\n"

            if len(similarity_text.split('\n')) > 4:
                rr.log(f"world/frame_{frame_idx}/similarities",
                       rr.TextDocument(similarity_text, media_type=rr.MediaType.MARKDOWN))

        # Log mask statistics
        stats_text = f"""
## Frame {frame_idx}

**Segmentation Results:**
- Number of masks: {num_masks}
- Total pixels segmented: {sum(mask.sum() for mask in masks):,}
- Average mask size: {np.mean([mask.sum() for mask in masks]):.1f} pixels

**CLIP Features:**
- Embedding dimension: {embedding_dim}
- Feature norm range: [{np.linalg.norm(features, axis=1).min():.3f}, {np.linalg.norm(features, axis=1).max():.3f}]

**Top 5 masks by size:**
"""
        mask_areas = [mask.sum() for mask in masks]
        top5_idx = np.argsort(mask_areas)[::-1][:5]
        for i, idx in enumerate(top5_idx, 1):
            stats_text += f"\n{i}. Mask {idx}: {mask_areas[idx]:,} pixels (score: {scores[idx]:.3f})"

        rr.log("world/stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))

        print(f"  Avg mask size: {np.mean(mask_areas):.1f} pixels")
        print()

    # Visualize embedding space
    if clip_cfg.get('visualize_embeddings', True) and len(all_embeddings) > 0:
        print("📊 Visualizing embedding space...")

        # Concatenate all embeddings
        all_features = np.concatenate(all_embeddings, axis=0)
        print(f"   Total embeddings: {len(all_features)}")

        # Reduce to 2D using PCA
        if len(all_features) > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(all_features)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"   PCA explained variance: {explained_var:.2%}")

            # Color by frame
            colors_by_frame = np.array([frame_idx for frame_idx, _ in all_mask_ids])

            # Log as scatter plot text
            embedding_viz_text = f"""
# CLIP Embedding Space Visualization (PCA)

**Total masks:** {len(all_features)}
**Explained variance:** {explained_var:.2%}

PCA reduces {embedding_dim}D embeddings to 2D for visualization.
Closer points indicate more similar visual features.

Frame colors:
"""
            for frame_idx in range(len(dataset)):
                num_in_frame = sum(1 for fid, _ in all_mask_ids if fid == frame_idx)
                if num_in_frame > 0:
                    embedding_viz_text += f"- Frame {frame_idx}: {num_in_frame} masks\n"

            rr.log("world/embeddings/info",
                   rr.TextDocument(embedding_viz_text, media_type=rr.MediaType.MARKDOWN))

    # Log completion summary
    completion_text = f"""
# Demo 2.2 Complete! ✓

Successfully extracted CLIP embeddings for {total_masks} masks across {len(dataset)} frames.

## Summary:
- **Total masks:** {total_masks}
- **Average masks per frame:** {total_masks / len(dataset):.1f}
- **SAM model:** {cfg.model.sam.model_type}
- **CLIP model:** {cfg.model.clip.model_name} ({cfg.model.clip.pretrained})
- **Embedding dimension:** {embedding_dim}

## What you're seeing:
- **RGB images**: Original camera frames
- **Segmentation overlay**: Colored masks overlaid on RGB
- **Individual masks**: Each mask shown separately
- **Crops**: Bounding box crops used for CLIP
- **Embeddings**: Visual feature vectors for each mask

## Key Features:
- Class-agnostic segmentation (no predefined categories)
- Dense visual embeddings for zero-shot recognition
- Semantic similarity via cosine distance
- Ready for natural language queries

## Next Steps:
Run demo_2_3 to track objects across multiple frames!

---
*Frames processed: {len(dataset)}*
*Dataset: {cfg.dataset.dataset_path}*
    """

    rr.log("world/summary", rr.TextDocument(completion_text, media_type=rr.MediaType.MARKDOWN))

    print("=" * 70)
    print("✅ Demo 2.2 completed successfully!")
    print("=" * 70)
    print()
    print(f"📊 Extracted CLIP embeddings for {total_masks} objects across {len(dataset)} frames")
    print(f"   Average: {total_masks / len(dataset):.1f} masks per frame")
    print(f"   Embedding dimension: {embedding_dim}")
    print()
    print("📊 Check the Rerun viewer:")
    print("   - Original RGB at 'world/camera/rgb'")
    print("   - Segmentation overlay at 'world/camera/segmentation_overlay'")
    print("   - Individual masks at 'world/masks/mask_XXX'")
    print("   - Mask crops at 'world/crops/crop_XXX'")
    print()
    print("Next: Run demo_2_3_object_tracking.py to associate masks across frames")
    print()


if __name__ == "__main__":
    main()
