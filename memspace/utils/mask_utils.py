"""
Utility functions for mask processing and merging
"""

import numpy as np
from typing import Tuple, List
from scipy.ndimage import label as connected_components


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks

    Args:
        mask1: Binary mask (H, W)
        mask2: Binary mask (H, W)

    Returns:
        IoU score in [0, 1]
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def merge_overlapping_masks(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge overlapping masks to reduce over-segmentation

    Strategy:
    1. Sort masks by score (keep higher quality masks)
    2. For each mask, check if it overlaps significantly with existing masks
    3. If IoU > threshold OR one contains the other, merge or skip

    Args:
        masks: Binary masks (N, H, W)
        boxes: Bounding boxes (N, 4) in xyxy format
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for merging (default: 0.5)
        containment_threshold: If one mask is X% inside another, merge (default: 0.85)

    Returns:
        merged_masks: Merged masks (M, H, W) where M <= N
        merged_boxes: Corresponding boxes (M, 4)
        merged_scores: Corresponding scores (M,) - max score of merged group
    """
    if len(masks) == 0:
        return masks, boxes, scores

    # Sort by score descending (keep better masks first)
    sorted_idx = np.argsort(scores)[::-1]
    masks = masks[sorted_idx]
    boxes = boxes[sorted_idx]
    scores = scores[sorted_idx]

    merged_masks = []
    merged_boxes = []
    merged_scores = []

    used = np.zeros(len(masks), dtype=bool)

    for i in range(len(masks)):
        if used[i]:
            continue

        # Start with current mask
        current_mask = masks[i].astype(bool)
        current_score = scores[i]

        # Find masks to merge with this one
        to_merge = [i]

        for j in range(i + 1, len(masks)):
            if used[j]:
                continue

            # Compute IoU
            iou = compute_mask_iou(current_mask, masks[j])

            # Compute containment (is mask j mostly inside current_mask?)
            intersection = np.logical_and(current_mask, masks[j]).sum()
            containment_j_in_current = intersection / masks[j].sum() if masks[j].sum() > 0 else 0
            containment_current_in_j = intersection / current_mask.sum() if current_mask.sum() > 0 else 0

            # Merge if:
            # 1. High IoU (very overlapping)
            # 2. One is mostly contained in the other (one is a fragment)
            if (iou > iou_threshold or
                containment_j_in_current > containment_threshold or
                containment_current_in_j > containment_threshold):

                # Merge mask j into current
                current_mask = np.logical_or(current_mask, masks[j])
                current_score = max(current_score, scores[j])
                to_merge.append(j)
                used[j] = True

        # Mark as used
        used[i] = True

        # Compute bounding box for merged mask
        ys, xs = np.where(current_mask)
        if len(ys) > 0:
            merged_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        else:
            merged_box = boxes[i]  # Fallback

        merged_masks.append(current_mask)
        merged_boxes.append(merged_box)
        merged_scores.append(current_score)

    # Convert to arrays
    merged_masks = np.array(merged_masks, dtype=np.uint8)
    merged_boxes = np.array(merged_boxes, dtype=np.float32)
    merged_scores = np.array(merged_scores, dtype=np.float32)

    return merged_masks, merged_boxes, merged_scores


def remove_small_holes(mask: np.ndarray, min_hole_area: int = 100) -> np.ndarray:
    """
    Remove small holes inside masks

    Args:
        mask: Binary mask (H, W)
        min_hole_area: Minimum hole size to keep (pixels)

    Returns:
        Mask with small holes filled
    """
    # Find connected components in inverted mask (holes)
    inverted = ~mask.astype(bool)
    labeled, num_features = connected_components(inverted)

    # Fill small holes
    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        if region_mask.sum() < min_hole_area:
            mask[region_mask] = True

    return mask


def filter_masks_by_depth(
    masks: np.ndarray,
    depth: np.ndarray,
    max_depth_variance: float = 0.5,
) -> np.ndarray:
    """
    Filter out masks that span multiple depth layers (likely over-segmented)

    Args:
        masks: Binary masks (N, H, W)
        depth: Depth image (H, W) in meters
        max_depth_variance: Max allowed depth std deviation within mask (meters)

    Returns:
        Boolean array indicating which masks to keep (N,)
    """
    keep = np.ones(len(masks), dtype=bool)

    for i, mask in enumerate(masks):
        # Get depth values in this mask
        mask_depths = depth[mask > 0]

        # Skip if no valid depth
        if len(mask_depths) == 0 or mask_depths.max() == 0:
            continue

        # Filter out zero/invalid depths
        valid_depths = mask_depths[mask_depths > 0]

        if len(valid_depths) == 0:
            continue

        # Compute depth variance
        depth_std = valid_depths.std()

        # If variance is too high, mask spans multiple depth layers
        if depth_std > max_depth_variance:
            keep[i] = False

    return keep
