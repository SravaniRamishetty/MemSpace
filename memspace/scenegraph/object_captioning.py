#!/usr/bin/env python3
"""
Object Captioning Module

Provides semantic labeling for tracked objects using VLMs (Vision Language Models).
Supports both Florence-2 and LLaVA.
Handles multi-view caption consolidation and object naming.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

from memspace.models.florence_wrapper import FlorenceWrapper
try:
    from memspace.models.llava_wrapper import LLaVAWrapper
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False


class ObjectCaptioner:
    """
    Manages semantic captioning for tracked objects

    Generates captions from multiple views and consolidates them into
    consistent object labels. Integrates with ObjectTracker to provide
    semantic understanding of tracked instances.

    Supports both Florence-2 and LLaVA VLMs.
    """

    def __init__(
        self,
        vlm_model: Union[FlorenceWrapper, 'LLaVAWrapper'],
        vlm_type: str = "florence",
        caption_task: Optional[str] = None,
        caption_query: Optional[str] = None,
        max_captions_per_object: int = 5,
        min_caption_length: int = 5,
    ):
        """
        Initialize Object Captioner

        Args:
            vlm_model: VLM wrapper instance (Florence or LLaVA)
            vlm_type: Type of VLM ("florence" or "llava")
            caption_task: Florence-2 task for captioning (only for Florence)
            caption_query: Query prompt (for LLaVA or custom Florence prompts)
            max_captions_per_object: Maximum captions to store per object
            min_caption_length: Minimum caption length to accept
        """
        self.vlm_model = vlm_model
        self.vlm_type = vlm_type.lower()

        # Set default caption task/query based on VLM type
        if self.vlm_type == "florence":
            self.caption_task = caption_task or "<DETAILED_CAPTION>"
            self.caption_query = None
        elif self.vlm_type == "llava":
            self.caption_task = None
            self.caption_query = caption_query or "What is the central object in this image?"
        else:
            raise ValueError(f"Unknown VLM type: {vlm_type}. Use 'florence' or 'llava'")

        self.max_captions_per_object = max_captions_per_object
        self.min_caption_length = min_caption_length

        # Store captions: object_id -> List[caption]
        self.object_captions: Dict[int, List[str]] = defaultdict(list)

        # Store final labels: object_id -> label
        self.object_labels: Dict[int, str] = {}

    def caption_frame_objects(
        self,
        image: np.ndarray,
        object_ids: List[int],
        bboxes: np.ndarray,
        padding: int = 20,
    ) -> Dict[int, str]:
        """
        Generate captions for objects in a frame

        Args:
            image: RGB image (H, W, 3), uint8
            object_ids: List of object IDs
            bboxes: Bounding boxes (N, 4) in format [x1, y1, x2, y2]
            padding: Padding around crops

        Returns:
            Dictionary mapping object_id -> caption
        """
        if len(object_ids) == 0:
            return {}

        # Extract crops and generate captions based on VLM type
        if self.vlm_type == "florence":
            crops, captions = self.vlm_model.caption_object_crops(
                image=image,
                bboxes=bboxes,
                padding=padding,
                task=self.caption_task,
            )
        elif self.vlm_type == "llava":
            crops, captions = self.vlm_model.caption_object_crops(
                image=image,
                bboxes=bboxes,
                padding=padding,
                query=self.caption_query,
            )
        else:
            raise ValueError(f"Unknown VLM type: {self.vlm_type}")

        # Store captions for each object
        frame_captions = {}
        for obj_id, caption in zip(object_ids, captions):
            # Filter out very short captions
            if len(caption) >= self.min_caption_length:
                # Add to collection
                if len(self.object_captions[obj_id]) < self.max_captions_per_object:
                    self.object_captions[obj_id].append(caption)

                frame_captions[obj_id] = caption

        return frame_captions

    def get_object_label(self, object_id: int, consolidate: bool = True) -> Optional[str]:
        """
        Get semantic label for an object

        Args:
            object_id: Object ID
            consolidate: Whether to consolidate multiple captions

        Returns:
            Object label or None
        """
        if object_id not in self.object_captions:
            return None

        # Check if we already have a consolidated label
        if object_id in self.object_labels:
            return self.object_labels[object_id]

        captions = self.object_captions[object_id]

        if len(captions) == 0:
            return None

        if consolidate and len(captions) > 1:
            # Consolidate multiple captions
            label = self._consolidate_captions(captions)
        else:
            # Use most recent caption
            label = captions[-1]

        # Cache the label
        self.object_labels[object_id] = label

        return label

    def _consolidate_captions(self, captions: List[str]) -> str:
        """
        Consolidate multiple captions into a single label

        Simple strategy: Find most common words/phrases and create a label.
        More sophisticated approach would use an LLM to merge captions.

        Args:
            captions: List of caption strings

        Returns:
            Consolidated label
        """
        # Simple strategy: Use the longest caption as it's likely most descriptive
        # In future, could use Florence-2 or simple word frequency analysis

        longest_caption = max(captions, key=len)

        # Extract first few words for a concise label (noun phrase extraction would be better)
        words = longest_caption.split()[:5]  # Take first 5 words
        label = " ".join(words)

        return label

    def get_all_labels(self) -> Dict[int, str]:
        """
        Get labels for all captioned objects

        Returns:
            Dictionary mapping object_id -> label
        """
        labels = {}
        for obj_id in self.object_captions.keys():
            label = self.get_object_label(obj_id)
            if label:
                labels[obj_id] = label

        return labels

    def get_statistics(self) -> dict:
        """
        Get captioning statistics

        Returns:
            Dictionary with statistics
        """
        num_captioned = len(self.object_captions)
        num_labeled = len(self.object_labels)

        total_captions = sum(len(caps) for caps in self.object_captions.values())
        avg_captions = total_captions / num_captioned if num_captioned > 0 else 0

        return {
            "num_captioned_objects": num_captioned,
            "num_labeled_objects": num_labeled,
            "total_captions": total_captions,
            "avg_captions_per_object": avg_captions,
        }

    def get_object_caption_history(self, object_id: int) -> List[str]:
        """
        Get all captions for an object

        Args:
            object_id: Object ID

        Returns:
            List of captions
        """
        return self.object_captions.get(object_id, [])

    def clear_object_captions(self, object_id: int):
        """
        Clear captions for an object

        Args:
            object_id: Object ID
        """
        if object_id in self.object_captions:
            del self.object_captions[object_id]
        if object_id in self.object_labels:
            del self.object_labels[object_id]
