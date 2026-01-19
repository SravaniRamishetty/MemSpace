"""
Object Tracker for multi-frame association
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from .object_instance import ObjectInstance, ObjectStatus


class ObjectTracker:
    """
    Tracks objects across multiple frames using spatial + semantic similarity

    Implements ConceptGraphs-style tracking:
    - Spatial similarity: Overlap-based (future: 3D IoU)
    - Visual similarity: CLIP embedding cosine similarity
    - Combined similarity: Weighted sum
    - Greedy matching: Each detection matched to best object
    """

    def __init__(
        self,
        sim_threshold: float = 1.0,
        spatial_weight: float = 0.5,
        clip_weight: float = 0.5,
        max_missing_frames: int = 10,
        min_observations: int = 2,
    ):
        """
        Args:
            sim_threshold: Minimum similarity for matching
            spatial_weight: Weight for spatial similarity (0-1)
            clip_weight: Weight for CLIP similarity (0-1)
            max_missing_frames: Max frames before object becomes inactive
            min_observations: Min observations before object is confirmed
        """
        self.sim_threshold = sim_threshold
        self.spatial_weight = spatial_weight
        self.clip_weight = clip_weight
        self.max_missing_frames = max_missing_frames
        self.min_observations = min_observations

        # Object tracking
        self.objects: List[ObjectInstance] = []
        self.next_object_id = 0
        self.current_frame = 0

        # Statistics
        self.total_detections = 0
        self.total_matches = 0
        self.total_new_objects = 0

    def compute_spatial_similarity(
        self,
        bbox1: np.ndarray,
        bbox2: np.ndarray,
    ) -> float:
        """
        Compute 2D bounding box IoU as spatial similarity

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU in [0, 1]
        """
        # Compute intersection
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Compute union
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def compute_clip_similarity(
        self,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> float:
        """
        Compute CLIP feature cosine similarity

        Args:
            feat1: CLIP embedding (D,), L2-normalized
            feat2: CLIP embedding (D,), L2-normalized

        Returns:
            Cosine similarity in [-1, 1]
        """
        # Cosine similarity (dot product of normalized vectors)
        return np.dot(feat1, feat2)

    def compute_similarity_matrix(
        self,
        detection_bboxes: np.ndarray,
        detection_features: np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarity matrix between detections and tracked objects

        Args:
            detection_bboxes: (M, 4) bounding boxes
            detection_features: (M, D) CLIP features

        Returns:
            similarity_matrix: (M, N) where M=detections, N=objects
        """
        M = len(detection_bboxes)
        N = len(self.objects)

        if N == 0:
            return np.zeros((M, 0))

        # Initialize similarity matrix
        spatial_sim = np.zeros((M, N))
        clip_sim = np.zeros((M, N))

        # Compute similarities
        for i in range(M):
            for j in range(N):
                obj = self.objects[j]

                # Only match with active/missing objects (not inactive)
                if obj.is_inactive():
                    spatial_sim[i, j] = -np.inf
                    clip_sim[i, j] = -np.inf
                    continue

                # Spatial similarity
                if obj.current_bbox is not None:
                    spatial_sim[i, j] = self.compute_spatial_similarity(
                        detection_bboxes[i],
                        obj.current_bbox
                    )

                # CLIP similarity
                if obj.clip_features is not None:
                    clip_sim[i, j] = self.compute_clip_similarity(
                        detection_features[i],
                        obj.clip_features
                    )

        # Aggregate similarities (ConceptGraphs style)
        # sim = (1 + phys_bias) * spatial + (1 - phys_bias) * visual
        # We use weights directly: sim = w_spatial * spatial + w_clip * clip
        combined_sim = self.spatial_weight * spatial_sim + self.clip_weight * clip_sim

        return combined_sim

    def match_detections(
        self,
        similarity_matrix: np.ndarray,
    ) -> List[Optional[int]]:
        """
        Match detections to objects using greedy matching

        Args:
            similarity_matrix: (M, N) similarity scores

        Returns:
            matches: List of length M, each element is object index or None
        """
        M = similarity_matrix.shape[0]
        matches = []

        for i in range(M):
            if similarity_matrix.shape[1] == 0:
                # No objects to match
                matches.append(None)
                continue

            max_sim = similarity_matrix[i].max()

            if max_sim < self.sim_threshold:
                # No good match
                matches.append(None)
            else:
                # Match to best object
                best_obj_idx = similarity_matrix[i].argmax()
                matches.append(best_obj_idx)

        return matches

    def update(
        self,
        frame_idx: int,
        masks: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        clip_features: np.ndarray,
        points_3d_list: Optional[List[np.ndarray]] = None,
        colors_3d_list: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Update tracker with new frame detections

        Args:
            frame_idx: Current frame index
            masks: (M, H, W) binary masks
            bboxes: (M, 4) bounding boxes
            scores: (M,) SAM scores
            clip_features: (M, D) CLIP embeddings
            points_3d_list: List of 3D points for each detection (optional)
            colors_3d_list: List of RGB colors for each detection (optional)

        Returns:
            object_ids: List of object IDs for each detection
            match_types: List indicating 0=new, 1=matched for each detection
        """
        self.current_frame = frame_idx
        self.total_detections += len(masks)

        M = len(masks)

        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(bboxes, clip_features)

        # Match detections to objects
        matches = self.match_detections(similarity_matrix)

        # Process matches
        object_ids = []
        match_types = []  # 0=new, 1=matched

        for i, obj_idx in enumerate(matches):
            if obj_idx is None:
                # Create new object
                new_obj = ObjectInstance(
                    object_id=self.next_object_id,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    num_observations=1,
                )

                # Initialize with first observation
                points_3d = points_3d_list[i] if points_3d_list is not None else None
                colors_3d = colors_3d_list[i] if colors_3d_list is not None else None

                new_obj.update(
                    frame_idx=frame_idx,
                    mask=masks[i],
                    bbox=bboxes[i],
                    score=scores[i],
                    clip_feature=clip_features[i],
                    points_3d=points_3d,
                    colors_3d=colors_3d,
                )

                self.objects.append(new_obj)
                object_ids.append(self.next_object_id)
                match_types.append(0)  # New object

                self.next_object_id += 1
                self.total_new_objects += 1
            else:
                # Update existing object
                obj = self.objects[obj_idx]

                points_3d = points_3d_list[i] if points_3d_list is not None else None
                colors_3d = colors_3d_list[i] if colors_3d_list is not None else None

                obj.update(
                    frame_idx=frame_idx,
                    mask=masks[i],
                    bbox=bboxes[i],
                    score=scores[i],
                    clip_feature=clip_features[i],
                    points_3d=points_3d,
                    colors_3d=colors_3d,
                )

                object_ids.append(obj.object_id)
                match_types.append(1)  # Matched object

                self.total_matches += 1

        # Mark objects not seen in this frame as missing
        matched_obj_indices = set(m for m in matches if m is not None)
        for i, obj in enumerate(self.objects):
            if i not in matched_obj_indices and not obj.is_inactive():
                obj.mark_missing(frame_idx)

        return object_ids, match_types

    def get_active_objects(self) -> List[ObjectInstance]:
        """Get all active objects"""
        return [obj for obj in self.objects if obj.is_active()]

    def get_all_objects(self) -> List[ObjectInstance]:
        """Get all objects (including missing/inactive)"""
        return self.objects

    def get_confirmed_objects(self) -> List[ObjectInstance]:
        """Get objects with enough observations (confirmed tracks)"""
        return [obj for obj in self.objects if obj.num_observations >= self.min_observations]

    def get_statistics(self) -> Dict[str, int]:
        """Get tracking statistics"""
        return {
            'total_objects': len(self.objects),
            'active_objects': len(self.get_active_objects()),
            'confirmed_objects': len(self.get_confirmed_objects()),
            'total_detections': self.total_detections,
            'total_matches': self.total_matches,
            'total_new_objects': self.total_new_objects,
            'current_frame': self.current_frame,
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"ObjectTracker(frame={stats['current_frame']}, "
                f"total={stats['total_objects']}, "
                f"active={stats['active_objects']}, "
                f"confirmed={stats['confirmed_objects']})")
