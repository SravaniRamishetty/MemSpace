#!/usr/bin/env python3
"""
Vision-based LLaVA spatial relationship reasoning.

Uses actual visual input (RGB frames with object annotations) to reason
about spatial relationships, similar to ConceptGraphs' GPT-4V approach.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import re

from memspace.scenegraph.spatial_relations import RelationType, SpatialRelation


class LLaVAVisualReasoner:
    """
    Vision-based spatial relationship reasoning using LLaVA.

    Creates annotated composite images showing pairs of objects
    and queries LLaVA to determine their spatial relationship.

    Similar to ConceptGraphs' GPT-4V approach but using LLaVA-1.5.
    """

    def __init__(
        self,
        llava_wrapper,
        dataset,
        use_visual_reasoning=True,
        annotation_style='contour',  # 'contour' or 'bbox'
    ):
        """
        Initialize visual spatial reasoner.

        Args:
            llava_wrapper: LLaVAWrapper instance for VLM inference
            dataset: Dataset instance to load RGB frames
            use_visual_reasoning: If True, use visual input; else fall back
            annotation_style: How to annotate objects ('contour' or 'bbox')
        """
        self.llava_wrapper = llava_wrapper
        self.dataset = dataset
        self.use_visual_reasoning = use_visual_reasoning
        self.annotation_style = annotation_style

        # Prompt template inspired by ConceptGraphs
        self.prompt_template = """You are an agent specializing in identifying physical and spatial relationships between objects in annotated images.

In this image, two objects are marked with colored annotations:
- Object 1 (RED): {label1}
- Object 2 (BLUE): {label2}

Analyze their positions and output ONE spatial relationship type.

The relation types you must report are:
- "on" or "on top of": when object 1 is physically placed on top of object 2
- "in": when object 1 is inside or contained by object 2
- "near" or "next to": when objects are adjacent or nearby
- "above": when object 1 is above object 2 (but not touching)
- "below": when object 1 is below object 2 (physically underneath, but not necessarily touching)
- "none": when there is no clear relationship

What is the spatial relationship between Object 1 and Object 2?
Respond with ONLY the relationship type (e.g., "on", "in", "near", "none")."""

    def query_relationship_visual(
        self,
        obj1: Dict,
        obj2: Dict,
    ) -> Tuple[str, float]:
        """
        Query spatial relationship using visual reasoning.

        Args:
            obj1: Object 1 dict with 'object_id', 'label', 'first_seen_frame', 'current_bbox_2d'
            obj2: Object 2 dict with similar structure

        Returns:
            (relationship, confidence): e.g., ("on", 0.85)
        """
        if not self.use_visual_reasoning or self.llava_wrapper is None:
            # Fall back to geometric reasoning
            from memspace.scenegraph.llava_spatial_reasoning import LLaVASpatialReasoner
            reasoner = LLaVASpatialReasoner(llava_wrapper=None)

            bbox1 = obj1.get('bounding_box_3d')
            bbox2 = obj2.get('bounding_box_3d')

            if bbox1 is None or bbox2 is None:
                return "none", 0.0

            return reasoner.query_relationship_text(
                obj1.get('label', 'object'),
                bbox1[:3], bbox1[3:],
                obj2.get('label', 'object'),
                bbox2[:3], bbox2[3:]
            )

        try:
            # Find best frame where both objects are visible
            frame_id = self._find_best_common_frame(obj1, obj2)

            if frame_id is None:
                print(f"  No common frame found for objects {obj1['object_id']} and {obj2['object_id']}, using geometric reasoning")
                # Fall back to geometric
                from memspace.scenegraph.llava_spatial_reasoning import LLaVASpatialReasoner
                reasoner = LLaVASpatialReasoner(llava_wrapper=None)
                bbox1 = obj1.get('bounding_box_3d')
                bbox2 = obj2.get('bounding_box_3d')
                return reasoner.query_relationship_text(
                    obj1.get('label', 'object'),
                    bbox1[:3], bbox1[3:],
                    obj2.get('label', 'object'),
                    bbox2[:3], bbox2[3:]
                )

            # Create annotated composite image
            annotated_image = self._create_annotated_image(
                frame_id, obj1, obj2
            )

            # Query LLaVA with visual input
            relationship, confidence = self._query_llava_visual(
                annotated_image,
                obj1.get('label', 'object'),
                obj2.get('label', 'object')
            )

            return relationship, confidence

        except Exception as e:
            print(f"Visual reasoning failed: {e}, falling back to geometric")
            # Fall back to geometric reasoning
            from memspace.scenegraph.llava_spatial_reasoning import LLaVASpatialReasoner
            reasoner = LLaVASpatialReasoner(llava_wrapper=None)
            bbox1 = obj1.get('bounding_box_3d')
            bbox2 = obj2.get('bounding_box_3d')
            return reasoner.query_relationship_text(
                obj1.get('label', 'object'),
                bbox1[:3], bbox1[3:],
                obj2.get('label', 'object'),
                bbox2[:3], bbox2[3:]
            )

    def _find_best_common_frame(
        self,
        obj1: Dict,
        obj2: Dict
    ) -> Optional[int]:
        """
        Find the best frame where both objects are visible.

        Uses first_seen_frame and last_seen_frame to find overlap.

        Args:
            obj1: Object 1 dict
            obj2: Object 2 dict

        Returns:
            Frame ID or None if no overlap
        """
        # Get frame ranges
        start1 = obj1.get('first_seen_frame', 0)
        end1 = obj1.get('last_seen_frame', 0)
        start2 = obj2.get('first_seen_frame', 0)
        end2 = obj2.get('last_seen_frame', 0)

        # Find overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start > overlap_end:
            # No overlap
            return None

        # Use the middle frame of the overlap for best visibility
        best_frame = (overlap_start + overlap_end) // 2

        return best_frame

    def _create_annotated_image(
        self,
        frame_id: int,
        obj1: Dict,
        obj2: Dict
    ) -> np.ndarray:
        """
        Create annotated image showing both objects.

        Loads the RGB frame and draws annotations (bounding boxes or contours)
        for both objects with different colors.

        Args:
            frame_id: Frame ID to load
            obj1: Object 1 dict with 'current_bbox_2d'
            obj2: Object 2 dict with 'current_bbox_2d'

        Returns:
            Annotated RGB image (H x W x 3) uint8
        """
        # Load RGB frame using __getitem__
        color, _, _, _ = self.dataset[frame_id]

        # Convert to numpy array if tensor
        if hasattr(color, 'cpu'):
            # Move tensor to CPU first before converting
            rgb = color.cpu().numpy().astype(np.uint8)
        elif hasattr(color, 'numpy'):
            rgb = color.numpy().astype(np.uint8)
        else:
            rgb = np.array(color).astype(np.uint8)

        # Convert to BGR for OpenCV drawing (we'll convert back)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

        # Draw object 1 in RED
        bbox1 = obj1.get('current_bbox_2d')
        if bbox1 is not None and len(bbox1) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox1]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red (BGR)
            # Add label
            cv2.putText(
                image, "1", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )

        # Draw object 2 in BLUE
        bbox2 = obj2.get('current_bbox_2d')
        if bbox2 is not None and len(bbox2) == 4:
            x2_1, y2_1, x2_2, y2_2 = [int(v) for v in bbox2]
            cv2.rectangle(image, (x2_1, y2_1), (x2_2, y2_2), (255, 0, 0), 3)  # Blue (BGR)
            # Add label
            cv2.putText(
                image, "2", (x2_1, y2_1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2
            )

        # Convert back to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image_rgb

    def _query_llava_visual(
        self,
        image: np.ndarray,
        label1: str,
        label2: str
    ) -> Tuple[str, float]:
        """
        Query LLaVA with annotated image.

        Args:
            image: Annotated RGB image
            label1: Label for object 1
            label2: Label for object 2

        Returns:
            (relationship, confidence)
        """
        # Format prompt
        prompt = self.prompt_template.format(
            label1=label1,
            label2=label2
        )

        # Query LLaVA using caption_image method
        response = self.llava_wrapper.caption_image(image, query=prompt)

        # Parse response
        relationship = self._parse_relationship(response)

        # Higher confidence for visual reasoning
        confidence = 0.85 if relationship != "none" else 0.5

        return relationship, confidence

    def _parse_relationship(self, response: str) -> str:
        """
        Parse LLaVA response to extract relationship type.

        Args:
            response: LLaVA text response

        Returns:
            Relationship string (on/in/near/above/below/none)
        """
        response_lower = response.lower().strip()

        # Check for each relationship type
        if 'on top of' in response_lower or response_lower.startswith('on'):
            return 'on'
        elif 'under' in response_lower or 'underneath' in response_lower:
            return 'below'  # Map "under" to "below" since UNDER doesn't exist
        elif 'in' in response_lower and 'inside' in response_lower:
            return 'in'
        elif 'near' in response_lower or 'next to' in response_lower or 'beside' in response_lower:
            return 'near'
        elif 'above' in response_lower:
            return 'above'
        elif 'below' in response_lower:
            return 'below'
        elif 'left' in response_lower:
            return 'left_of'
        elif 'right' in response_lower:
            return 'right_of'
        elif 'front' in response_lower:
            return 'front_of'
        elif 'behind' in response_lower:
            return 'behind'
        else:
            return 'none'

    def compute_pairwise_relationships(
        self,
        objects: List[Dict],
        min_confidence: float = 0.5,
    ) -> List[SpatialRelation]:
        """
        Compute spatial relationships between all pairs of objects using visual reasoning.

        Args:
            objects: List of object dicts
            min_confidence: Minimum confidence threshold

        Returns:
            List of SpatialRelation objects
        """
        relationships = []

        total_pairs = len(objects) * (len(objects) - 1) // 2
        processed = 0

        print(f"\n🎨 Using vision-based LLaVA spatial reasoning...")
        print(f"   Total object pairs to process: {total_pairs}")

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue

                processed += 1
                if processed % 10 == 0:
                    print(f"   Progress: {processed}/{total_pairs} pairs...")

                # Query relationship
                rel_str, confidence = self.query_relationship_visual(obj1, obj2)

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
                            metadata={'method': 'llava_visual_reasoning'}
                        )
                        relationships.append(relation)

        print(f"✓ Processed all {total_pairs} pairs")
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
