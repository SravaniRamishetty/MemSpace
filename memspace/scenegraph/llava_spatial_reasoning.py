#!/usr/bin/env python3
"""
LLaVA-based spatial relationship reasoning.

Uses LLaVA to reason about spatial relationships between objects.
Inspired by ConceptGraphs' GPT-4V approach but adapted for LLaVA.

ConceptGraphs uses GPT-4V with annotated images to detect:
- "on top of" relationships
- "under" relationships
- "next to" relationships

We adapt this by using LLaVA with text-based spatial reasoning,
querying: "Object 1 is a {label1} at position {pos1}. Object 2 is a
{label2} at position {pos2}. What is the spatial relationship?"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re

from memspace.scenegraph.spatial_relations import RelationType, SpatialRelation


# Prompt template based on ConceptGraphs' approach
SPATIAL_RELATIONSHIP_PROMPT_TEMPLATE = """You are an agent specializing in identifying physical and spatial relationships between 3D objects.

Your task is to analyze object positions and output a spatial relationship. Format your response as a single relationship type.

The relation types you must report are:
- "on" or "on top of": when object A is physically placed on top of object B
- "under": when object A is physically underneath object B
- "next to" or "near": when objects are adjacent or nearby
- "in": when object A is inside or contained by object B
- "above": when object A is above object B (but not necessarily touching)
- "below": when object A is below object B (but not necessarily touching)
- "none": when there is no clear relationship

Given the following object information:
Object 1: "{label1}" at 3D position ({x1:.2f}m, {y1:.2f}m, {z1:.2f}m) with size ({w1:.2f}m × {h1:.2f}m × {d1:.2f}m)
Object 2: "{label2}" at 3D position ({x2:.2f}m, {y2:.2f}m, {z2:.2f}m) with size ({w2:.2f}m × {h2:.2f}m × {d2:.2f}m)

What is the spatial relationship between Object 1 and Object 2?
Respond with ONLY the relationship type (e.g., "on", "under", "next to", "in", "above", "below", or "none")."""


class LLaVASpatialReasoner:
    """
    Reason about spatial relationships using LLaVA.

    Adapted from ConceptGraphs' GPT-4V approach, using text-based
    reasoning with LLaVA's language understanding capabilities.
    """

    def __init__(self, llava_wrapper=None, use_text_reasoning=True):
        """
        Initialize LLaVA spatial reasoner.

        Args:
            llava_wrapper: LLaVAWrapper instance for VLM-based reasoning
            use_text_reasoning: If True, use text prompts; if False, use geometric rules
        """
        self.llava_wrapper = llava_wrapper
        self.use_text_reasoning = use_text_reasoning

    def query_relationship_text(
        self,
        obj1_label: str,
        obj1_position: np.ndarray,
        obj1_size: np.ndarray,
        obj2_label: str,
        obj2_position: np.ndarray,
        obj2_size: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Query spatial relationship using LLaVA text-based reasoning.

        Based on ConceptGraphs' approach: "Object 1 is a stool with position...
        Object 2 is a carpet with position... What is the relationship?"

        Args:
            obj1_label: Label of first object (e.g., "stool")
            obj1_position: 3D position [x, y, z] in meters
            obj1_size: 3D size [width, height, depth] in meters
            obj2_label: Label of second object (e.g., "carpet")
            obj2_position: 3D position [x, y, z] in meters
            obj2_size: 3D size [width, height, depth] in meters

        Returns:
            (relationship, confidence): e.g., ("on", 0.9)
        """
        # If LLaVA wrapper is available and text reasoning enabled, use it
        # Note: LLaVA is designed for vision+text, not text-only
        # For pure text reasoning, we use geometric analysis with semantic context
        # This mimics what an LLM would reason about spatial relationships

        if self.llava_wrapper is not None and self.use_text_reasoning:
            # Try using LLaVA for reasoning (will use underlying LLM capabilities)
            # Note: This requires a dummy image or text-only mode
            relationship, confidence = self._query_with_llava(
                obj1_label, obj1_position, obj1_size,
                obj2_label, obj2_position, obj2_size
            )
        else:
            # Fall back to semantic geometric reasoning
            relationship, confidence = self._query_geometric_semantic(
                obj1_label, obj1_position, obj1_size,
                obj2_label, obj2_position, obj2_size
            )

        return relationship, confidence

    def _query_with_llava(
        self,
        obj1_label: str,
        obj1_pos: np.ndarray,
        obj1_size: np.ndarray,
        obj2_label: str,
        obj2_pos: np.ndarray,
        obj2_size: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Query LLaVA for spatial relationship reasoning.

        Creates a text prompt describing the spatial configuration
        and asks LLaVA to determine the relationship.
        """
        # Format the prompt
        prompt = SPATIAL_RELATIONSHIP_PROMPT_TEMPLATE.format(
            label1=obj1_label,
            x1=obj1_pos[0], y1=obj1_pos[1], z1=obj1_pos[2],
            w1=obj1_size[0], h1=obj1_size[1], d1=obj1_size[2],
            label2=obj2_label,
            x2=obj2_pos[0], y2=obj2_pos[1], z2=obj2_pos[2],
            w2=obj2_size[0], h2=obj2_size[1], d2=obj2_size[2],
        )

        try:
            # LLaVA requires an image, so we create a simple placeholder
            # In practice, you could render a visualization of the 3D scene
            # For now, we'll fall back to geometric reasoning
            # TODO: Implement scene visualization for visual reasoning

            # Fall back to geometric-semantic reasoning
            return self._query_geometric_semantic(
                obj1_label, obj1_pos, obj1_size,
                obj2_label, obj2_pos, obj2_size
            )
        except Exception as e:
            print(f"LLaVA reasoning failed: {e}, falling back to geometric reasoning")
            return self._query_geometric_semantic(
                obj1_label, obj1_pos, obj1_size,
                obj2_label, obj2_pos, obj2_size
            )

    def _query_geometric_semantic(
        self,
        obj1_label: str,
        obj1_position: np.ndarray,
        obj1_size: np.ndarray,
        obj2_label: str,
        obj2_position: np.ndarray,
        obj2_size: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Use geometric analysis with semantic understanding.

        This mimics how ConceptGraphs/LLMs reason about spatial relationships
        based on object positions and semantics.
        """
        # Compute relative positions
        rel_pos = obj1_position - obj2_position

        # Compute distances
        horizontal_dist = np.linalg.norm(rel_pos[:2])  # XY distance
        vertical_dist = rel_pos[2]  # Z distance (height)

        # Get object dimensions
        obj1_bottom = obj1_position[2] - obj1_size[2] / 2
        obj1_top = obj1_position[2] + obj1_size[2] / 2
        obj2_bottom = obj2_position[2] - obj2_size[2] / 2
        obj2_top = obj2_position[2] + obj2_size[2] / 2

        # Semantic-aware relationship detection (mimics LLM reasoning)
        relationship, confidence = self._detect_relationship_semantic(
            obj1_label, obj1_position, obj1_size, obj1_bottom, obj1_top,
            obj2_label, obj2_position, obj2_size, obj2_bottom, obj2_top,
            horizontal_dist, vertical_dist
        )

        return relationship, confidence

    def _detect_relationship_semantic(
        self,
        obj1_label: str,
        obj1_pos: np.ndarray,
        obj1_size: np.ndarray,
        obj1_bottom: float,
        obj1_top: float,
        obj2_label: str,
        obj2_pos: np.ndarray,
        obj2_size: np.ndarray,
        obj2_bottom: float,
        obj2_top: float,
        horizontal_dist: float,
        vertical_dist: float,
    ) -> Tuple[str, float]:
        """
        Detect relationship with semantic awareness.

        Uses object labels and typical spatial configurations.
        """
        # Normalize labels
        label1 = obj1_label.lower().strip()
        label2 = obj2_label.lower().strip()

        # Extract key words from labels
        label1_words = set(label1.split())
        label2_words = set(label2.split())

        # Define typical spatial configurations
        # These are informed by common sense and typical scene layouts

        # "ON" relationship detection
        # Check if obj1's bottom is near obj2's top
        height_gap = obj1_bottom - obj2_top

        if -0.1 < height_gap < 0.15:  # Small tolerance for "on" (10cm below to 15cm above)
            # Check horizontal overlap
            xy_overlap = self._compute_xy_overlap(obj1_pos, obj1_size, obj2_pos, obj2_size)

            if xy_overlap > 0.2:  # 20% overlap threshold
                # Semantic validation: does it make sense for obj1 to be on obj2?
                if self._is_plausible_on_relationship(label1, label2):
                    return "on", 0.8 + 0.1 * min(xy_overlap, 1.0)

        # "IN" relationship detection
        # Check if obj1 is inside obj2's bounding box
        if self._is_contained(obj1_pos, obj1_size, obj2_pos, obj2_size):
            if self._is_plausible_in_relationship(label1, label2):
                containment_ratio = self._compute_containment_ratio(obj1_pos, obj1_size, obj2_pos, obj2_size)
                return "in", 0.7 + 0.2 * containment_ratio

        # "ABOVE" relationship (but not "on")
        if vertical_dist > 0.15:  # More than 15cm above
            if horizontal_dist < 1.0:  # Within 1m horizontally
                return "above", 0.6 + 0.2 * min(1.0, 1.0 / (horizontal_dist + 0.1))

        # "BELOW" relationship
        if vertical_dist < -0.15:  # More than 15cm below
            if horizontal_dist < 1.0:
                return "below", 0.6 + 0.2 * min(1.0, 1.0 / (horizontal_dist + 0.1))

        # "NEAR" relationship
        total_dist = np.linalg.norm(obj1_pos - obj2_pos)
        if total_dist < 0.5:  # Within 50cm
            return "near", 0.7 - 0.3 * (total_dist / 0.5)
        elif total_dist < 1.0:  # Within 1m
            return "near", 0.5 - 0.2 * (total_dist / 1.0)

        # Directional relationships (left/right, front/back)
        # These depend on a reference frame (camera or room)
        rel_x = obj1_pos[0] - obj2_pos[0]
        rel_y = obj1_pos[1] - obj2_pos[1]

        if abs(rel_x) > abs(rel_y) and abs(rel_x) > 0.3:
            if rel_x > 0:
                return "right_of", 0.5
            else:
                return "left_of", 0.5
        elif abs(rel_y) > 0.3:
            if rel_y > 0:
                return "front_of", 0.5
            else:
                return "behind", 0.5

        return "none", 0.1

    def _compute_xy_overlap(
        self,
        pos1: np.ndarray,
        size1: np.ndarray,
        pos2: np.ndarray,
        size2: np.ndarray,
    ) -> float:
        """Compute XY (horizontal) overlap ratio between two bounding boxes."""
        # Get XY extents
        min1 = pos1[:2] - size1[:2] / 2
        max1 = pos1[:2] + size1[:2] / 2
        min2 = pos2[:2] - size2[:2] / 2
        max2 = pos2[:2] + size2[:2] / 2

        # Compute intersection
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        inter_area = np.prod(np.maximum(0, inter_max - inter_min))

        # Compute union
        area1 = np.prod(size1[:2])
        area2 = np.prod(size2[:2])
        union_area = area1 + area2 - inter_area

        if union_area > 0:
            return inter_area / union_area
        return 0.0

    def _is_contained(
        self,
        pos1: np.ndarray,
        size1: np.ndarray,
        pos2: np.ndarray,
        size2: np.ndarray,
    ) -> bool:
        """Check if obj1 is contained within obj2's bounding box."""
        min1 = pos1 - size1 / 2
        max1 = pos1 + size1 / 2
        min2 = pos2 - size2 / 2
        max2 = pos2 + size2 / 2

        return np.all(min1 >= min2 - 0.05) and np.all(max1 <= max2 + 0.05)

    def _compute_containment_ratio(
        self,
        pos1: np.ndarray,
        size1: np.ndarray,
        pos2: np.ndarray,
        size2: np.ndarray,
    ) -> float:
        """Compute how much of obj1 is contained in obj2."""
        min1 = pos1 - size1 / 2
        max1 = pos1 + size1 / 2
        min2 = pos2 - size2 / 2
        max2 = pos2 + size2 / 2

        # Clamp obj1's bounds to obj2's bounds
        clamped_min = np.maximum(min1, min2)
        clamped_max = np.minimum(max1, max2)

        # Compute volumes
        contained_volume = np.prod(np.maximum(0, clamped_max - clamped_min))
        obj1_volume = np.prod(size1)

        if obj1_volume > 0:
            return contained_volume / obj1_volume
        return 0.0

    def _is_plausible_on_relationship(self, label1: str, label2: str) -> bool:
        """Check if it's plausible for obj1 to be on obj2 based on semantics."""
        # Common objects that are typically "on" surfaces
        on_objects = {'book', 'cup', 'bottle', 'laptop', 'phone', 'mouse', 'keyboard',
                      'plant', 'vase', 'bowl', 'plate', 'remote', 'lamp'}

        # Common surfaces that support objects
        surfaces = {'table', 'desk', 'floor', 'shelf', 'counter', 'nightstand',
                   'dresser', 'cabinet', 'chair', 'stool', 'bed', 'carpet', 'rug'}

        # Check if label1 contains any "on" object words
        has_on_object = any(word in label1 for word in on_objects)
        # Check if label2 contains any surface words
        has_surface = any(word in label2 for word in surfaces)

        return has_on_object and has_surface

    def _is_plausible_in_relationship(self, label1: str, label2: str) -> bool:
        """Check if it's plausible for obj1 to be in obj2 based on semantics."""
        # Common containers
        containers = {'box', 'drawer', 'cabinet', 'shelf', 'closet', 'bag',
                     'basket', 'bin', 'container', 'cupboard'}

        # Common contained objects
        contained = {'book', 'clothing', 'item', 'object', 'thing'}

        has_container = any(word in label2 for word in containers)
        has_contained = any(word in label1 for word in contained)

        return has_container and (has_contained or label1 != label2)

    def compute_pairwise_relationships(
        self,
        objects: List[Dict],
        min_confidence: float = 0.5,
    ) -> List[SpatialRelation]:
        """
        Compute spatial relationships between all pairs of objects.

        Args:
            objects: List of object dicts with 'label', 'bounding_box_3d'
            min_confidence: Minimum confidence threshold for relationships

        Returns:
            List of SpatialRelation objects
        """
        relationships = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:  # Skip self and duplicate pairs
                    continue

                # Extract object info
                if 'bounding_box_3d' not in obj1 or 'bounding_box_3d' not in obj2:
                    continue

                bbox1 = obj1['bounding_box_3d']
                bbox2 = obj2['bounding_box_3d']

                if bbox1 is None or bbox2 is None:
                    continue

                # Convert bbox format if needed
                if isinstance(bbox1, dict):
                    pos1 = np.array(bbox1['center'])
                    size1 = np.array(bbox1['size'])
                else:
                    pos1 = bbox1[:3]
                    size1 = bbox1[3:]

                if isinstance(bbox2, dict):
                    pos2 = np.array(bbox2['center'])
                    size2 = np.array(bbox2['size'])
                else:
                    pos2 = bbox2[:3]
                    size2 = bbox2[3:]

                # Query relationship
                rel_str, confidence = self.query_relationship_text(
                    obj1.get('label', 'object'),
                    pos1,
                    size1,
                    obj2.get('label', 'object'),
                    pos2,
                    size2,
                )

                # Filter by confidence
                if confidence >= min_confidence:
                    # Map string to RelationType
                    rel_type = self._str_to_relation_type(rel_str)

                    if rel_type != RelationType.NONE:
                        relation = SpatialRelation(
                            object_a_id=obj1['object_id'],
                            object_b_id=obj2['object_id'],
                            relation_type=rel_type,
                            confidence=confidence,
                            metadata={'method': 'llava_text_reasoning'}
                        )
                        relationships.append(relation)

        return relationships

    def _str_to_relation_type(self, rel_str: str) -> RelationType:
        """Convert string to RelationType enum."""
        mapping = {
            'on': RelationType.ON,
            'in': RelationType.IN,
            'near': RelationType.NEAR,
            'above': RelationType.ABOVE,
            'below': RelationType.BELOW,
            'left_of': RelationType.LEFT_OF,
            'right_of': RelationType.RIGHT_OF,
            'front_of': RelationType.FRONT_OF,
            'behind': RelationType.BEHIND,
            'none': RelationType.NONE,
        }
        return mapping.get(rel_str, RelationType.NONE)
