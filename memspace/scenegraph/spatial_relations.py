"""
Spatial relationship detection between 3D objects.

This module provides geometric-based spatial relationship detection without requiring LLMs.
Inspired by ConceptGraphs' approach but using pure geometric reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import faiss


class RelationType(Enum):
    """Types of spatial relationships between objects."""
    ON = "on"           # Object A is on top of object B
    IN = "in"           # Object A is inside object B
    NEAR = "near"       # Object A is near object B
    ABOVE = "above"     # Object A is above object B (but not necessarily "on")
    BELOW = "below"     # Object A is below object B
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FRONT_OF = "front_of"
    BEHIND = "behind"
    NONE = "none"       # No clear spatial relationship


@dataclass
class SpatialRelation:
    """Represents a spatial relationship between two objects."""
    object_a_id: int
    object_b_id: int
    relation_type: RelationType
    confidence: float  # 0.0 to 1.0
    metadata: Optional[Dict] = None


def compute_bbox_overlap_faiss(
    objects_a: List[Dict],
    objects_b: Optional[List[Dict]] = None,
    distance_threshold: float = 0.025,
) -> np.ndarray:
    """
    Compute overlap matrix between object point clouds using FAISS.

    Based on ConceptGraphs' compute_overlap_matrix_general() approach.
    Uses FAISS for efficient nearest neighbor search.

    Args:
        objects_a: List of objects with 'pcd' (Open3D point cloud)
        objects_b: Optional second list. If None, computes self-overlap of objects_a
        distance_threshold: Points closer than this are considered overlapping (meters)

    Returns:
        overlap_matrix: Shape (len(objects_a), len(objects_b))
                       Values are ratios of overlapping points [0.0, 1.0]
    """
    # Self-comparison if objects_b not provided
    same_objects = objects_b is None
    objects_b = objects_a if same_objects else objects_b

    len_a = len(objects_a)
    len_b = len(objects_b)
    overlap_matrix = np.zeros((len_a, len_b))

    # Convert point clouds to numpy arrays
    points_a = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_a]
    points_b = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_b]

    # Create FAISS indices for objects_a (for efficient search)
    indices_a = []
    for i, points_arr in enumerate(points_a):
        if points_arr.shape[0] == 0:
            indices_a.append(None)
            continue

        # Create FAISS index
        index = faiss.IndexFlatL2(3)  # 3D points

        # Ensure C-contiguous
        if not points_arr.flags['C_CONTIGUOUS']:
            points_arr = np.ascontiguousarray(points_arr)

        index.add(points_arr)
        indices_a.append(index)

    # For each object in objects_b, find how many of its points are near objects_a
    threshold_squared = distance_threshold ** 2

    for j, points_b_arr in enumerate(points_b):
        if points_b_arr.shape[0] == 0:
            continue

        # Ensure C-contiguous
        if not points_b_arr.flags['C_CONTIGUOUS']:
            points_b_arr = np.ascontiguousarray(points_b_arr)

        for i, index_a in enumerate(indices_a):
            if index_a is None:
                continue

            # Skip self-comparison in same_objects mode
            if same_objects and i == j:
                continue

            # Search for nearest neighbor of each point in b within object a
            # k=1 means find single nearest neighbor
            distances, _ = index_a.search(points_b_arr, k=1)

            # Count points in b that are within threshold of a
            num_close = np.sum(distances[:, 0] < threshold_squared)

            # Overlap ratio: fraction of b's points near a
            overlap_ratio = num_close / points_b_arr.shape[0]
            overlap_matrix[i, j] = overlap_ratio

    return overlap_matrix


def detect_on_relationship(
    bbox_a: np.ndarray,
    bbox_b: np.ndarray,
    height_tolerance: float = 0.05,
    overlap_threshold: float = 0.3,
) -> Tuple[bool, float]:
    """
    Detect if object A is "on" object B.

    Criteria:
    1. A's bottom is at approximately B's top (within height_tolerance)
    2. A's center is horizontally within B's footprint (XY overlap)

    Args:
        bbox_a: [center_x, center_y, center_z, width, height, depth]
        bbox_b: Same format
        height_tolerance: Allowed gap between A's bottom and B's top (meters)
        overlap_threshold: Minimum XY overlap ratio required

    Returns:
        (is_on, confidence): Boolean and confidence score [0.0, 1.0]
    """
    # Extract centers and extents
    center_a = bbox_a[:3]
    extent_a = bbox_a[3:]
    center_b = bbox_b[:3]
    extent_b = bbox_b[3:]

    # Compute bottom of A and top of B
    bottom_a = center_a[2] - extent_a[2] / 2
    top_b = center_b[2] + extent_b[2] / 2

    # Check vertical alignment: A's bottom should be at B's top
    vertical_gap = abs(bottom_a - top_b)
    if vertical_gap > height_tolerance:
        return False, 0.0

    # Check horizontal overlap (XY plane)
    # Compute min/max for both objects in XY
    min_a_xy = center_a[:2] - extent_a[:2] / 2
    max_a_xy = center_a[:2] + extent_a[:2] / 2
    min_b_xy = center_b[:2] - extent_b[:2] / 2
    max_b_xy = center_b[:2] + extent_b[:2] / 2

    # Compute intersection
    intersection_min = np.maximum(min_a_xy, min_b_xy)
    intersection_max = np.minimum(max_a_xy, max_b_xy)

    # Check if intersection exists
    if np.any(intersection_max <= intersection_min):
        return False, 0.0

    # Compute intersection area
    intersection_area = np.prod(intersection_max - intersection_min)
    area_a = np.prod(extent_a[:2])

    # Overlap ratio: how much of A's footprint is over B
    overlap_ratio = intersection_area / area_a if area_a > 0 else 0.0

    if overlap_ratio < overlap_threshold:
        return False, 0.0

    # Confidence based on vertical alignment and overlap
    vertical_confidence = 1.0 - min(vertical_gap / height_tolerance, 1.0)
    overlap_confidence = min(overlap_ratio / overlap_threshold, 1.0)
    confidence = (vertical_confidence + overlap_confidence) / 2

    return True, confidence


def detect_in_relationship(
    bbox_a: np.ndarray,
    bbox_b: np.ndarray,
    containment_threshold: float = 0.8,
) -> Tuple[bool, float]:
    """
    Detect if object A is "in" object B (containment).

    Criteria:
    - A's bounding box is mostly contained within B's bounding box

    Args:
        bbox_a: [center_x, center_y, center_z, width, height, depth]
        bbox_b: Same format
        containment_threshold: Minimum volume overlap ratio (A inside B)

    Returns:
        (is_in, confidence): Boolean and confidence score
    """
    center_a = bbox_a[:3]
    extent_a = bbox_a[3:]
    center_b = bbox_b[:3]
    extent_b = bbox_b[3:]

    # Compute min/max corners
    min_a = center_a - extent_a / 2
    max_a = center_a + extent_a / 2
    min_b = center_b - extent_b / 2
    max_b = center_b + extent_b / 2

    # Check if A is contained in B
    # A's min must be >= B's min, and A's max must be <= B's max
    contained_min = min_a >= min_b
    contained_max = max_a <= max_b

    if not (np.all(contained_min) and np.all(contained_max)):
        # Not fully contained, check partial containment
        # Compute intersection volume
        intersection_min = np.maximum(min_a, min_b)
        intersection_max = np.minimum(max_a, max_b)

        # Check if intersection exists
        if np.any(intersection_max <= intersection_min):
            return False, 0.0

        # Compute volumes
        intersection_volume = np.prod(intersection_max - intersection_min)
        volume_a = np.prod(extent_a)

        # Containment ratio: how much of A is inside B
        containment_ratio = intersection_volume / volume_a if volume_a > 0 else 0.0

        if containment_ratio < containment_threshold:
            return False, 0.0

        confidence = containment_ratio
        return True, confidence

    # Fully contained
    return True, 1.0


def detect_near_relationship(
    bbox_a: np.ndarray,
    bbox_b: np.ndarray,
    distance_threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Detect if objects are "near" each other.

    Args:
        bbox_a: [center_x, center_y, center_z, width, height, depth]
        bbox_b: Same format
        distance_threshold: Maximum center-to-center distance (meters)

    Returns:
        (is_near, confidence): Boolean and confidence score
    """
    center_a = bbox_a[:3]
    center_b = bbox_b[:3]

    # Compute center-to-center distance
    distance = np.linalg.norm(center_a - center_b)

    if distance > distance_threshold:
        return False, 0.0

    # Confidence inversely proportional to distance
    confidence = 1.0 - (distance / distance_threshold)
    return True, confidence


def detect_directional_relation(
    bbox_a: np.ndarray,
    bbox_b: np.ndarray,
    direction_threshold: float = 0.3,
) -> Tuple[Optional[RelationType], float]:
    """
    Detect directional relationship (left_of, right_of, above, below, etc.).

    Args:
        bbox_a: [center_x, center_y, center_z, width, height, depth]
        bbox_b: Same format
        direction_threshold: Minimum offset to consider directional

    Returns:
        (relation_type, confidence): Direction type and confidence
    """
    center_a = bbox_a[:3]
    center_b = bbox_b[:3]

    # Compute offset vector from B to A
    offset = center_a - center_b

    # Find dominant axis
    abs_offset = np.abs(offset)
    dominant_axis = np.argmax(abs_offset)

    # Check if offset is significant enough
    if abs_offset[dominant_axis] < direction_threshold:
        return None, 0.0

    # Determine direction
    confidence = min(abs_offset[dominant_axis] / direction_threshold, 1.0)

    if dominant_axis == 0:  # X axis
        return RelationType.RIGHT_OF if offset[0] > 0 else RelationType.LEFT_OF, confidence
    elif dominant_axis == 1:  # Y axis
        return RelationType.FRONT_OF if offset[1] > 0 else RelationType.BEHIND, confidence
    else:  # Z axis
        return RelationType.ABOVE if offset[2] > 0 else RelationType.BELOW, confidence


def compute_spatial_relationships(
    objects: List[Dict],
    overlap_matrix: Optional[np.ndarray] = None,
    overlap_threshold: float = 0.01,
    config: Optional[Dict] = None,
) -> List[SpatialRelation]:
    """
    Compute all spatial relationships between objects.

    Args:
        objects: List of SemanticObject dicts with 'bbox_3d'
        overlap_matrix: Pre-computed bbox overlap matrix (optional)
        overlap_threshold: Only check relationships if overlap > threshold
        config: Configuration dict for relationship detection parameters

    Returns:
        List of SpatialRelation objects
    """
    # Default config
    if config is None:
        config = {
            'on_height_tolerance': 0.05,
            'on_overlap_threshold': 0.3,
            'in_containment_threshold': 0.8,
            'near_distance_threshold': 0.5,
            'directional_threshold': 0.3,
        }

    relations = []
    num_objects = len(objects)

    # If no overlap matrix provided, assume all pairs need checking
    if overlap_matrix is None:
        overlap_matrix = np.ones((num_objects, num_objects))

    # Check each pair of objects
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            # Skip if no spatial overlap (optimization)
            if overlap_matrix[i, j] < overlap_threshold and overlap_matrix[j, i] < overlap_threshold:
                continue

            obj_a = objects[i]
            obj_b = objects[j]

            bbox_a = obj_a['bounding_box_3d']
            bbox_b = obj_b['bounding_box_3d']

            # Check "on" relationship (A on B)
            is_on_ab, conf_on_ab = detect_on_relationship(
                bbox_a, bbox_b,
                height_tolerance=config['on_height_tolerance'],
                overlap_threshold=config['on_overlap_threshold'],
            )
            if is_on_ab:
                relations.append(SpatialRelation(
                    object_a_id=obj_a['object_id'],
                    object_b_id=obj_b['object_id'],
                    relation_type=RelationType.ON,
                    confidence=conf_on_ab,
                    metadata={'direction': 'a_on_b'}
                ))

            # Check "on" relationship (B on A)
            is_on_ba, conf_on_ba = detect_on_relationship(
                bbox_b, bbox_a,
                height_tolerance=config['on_height_tolerance'],
                overlap_threshold=config['on_overlap_threshold'],
            )
            if is_on_ba:
                relations.append(SpatialRelation(
                    object_a_id=obj_b['object_id'],
                    object_b_id=obj_a['object_id'],
                    relation_type=RelationType.ON,
                    confidence=conf_on_ba,
                    metadata={'direction': 'b_on_a'}
                ))

            # Check "in" relationship (A in B)
            is_in_ab, conf_in_ab = detect_in_relationship(
                bbox_a, bbox_b,
                containment_threshold=config['in_containment_threshold'],
            )
            if is_in_ab:
                relations.append(SpatialRelation(
                    object_a_id=obj_a['object_id'],
                    object_b_id=obj_b['object_id'],
                    relation_type=RelationType.IN,
                    confidence=conf_in_ab,
                    metadata={'direction': 'a_in_b'}
                ))

            # Check "in" relationship (B in A)
            is_in_ba, conf_in_ba = detect_in_relationship(
                bbox_b, bbox_a,
                containment_threshold=config['in_containment_threshold'],
            )
            if is_in_ba:
                relations.append(SpatialRelation(
                    object_a_id=obj_b['object_id'],
                    object_b_id=obj_a['object_id'],
                    relation_type=RelationType.IN,
                    confidence=conf_in_ba,
                    metadata={'direction': 'b_in_a'}
                ))

            # Check "near" relationship (symmetric)
            is_near, conf_near = detect_near_relationship(
                bbox_a, bbox_b,
                distance_threshold=config['near_distance_threshold'],
            )
            if is_near:
                relations.append(SpatialRelation(
                    object_a_id=obj_a['object_id'],
                    object_b_id=obj_b['object_id'],
                    relation_type=RelationType.NEAR,
                    confidence=conf_near,
                    metadata={'symmetric': True}
                ))

            # Check directional relationships
            direction_ab, conf_dir_ab = detect_directional_relation(
                bbox_a, bbox_b,
                direction_threshold=config['directional_threshold'],
            )
            if direction_ab is not None:
                relations.append(SpatialRelation(
                    object_a_id=obj_a['object_id'],
                    object_b_id=obj_b['object_id'],
                    relation_type=direction_ab,
                    confidence=conf_dir_ab,
                ))

    return relations
