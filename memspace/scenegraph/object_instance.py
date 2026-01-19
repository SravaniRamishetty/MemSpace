"""
Object Instance class for tracking objects across frames
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ObjectStatus(Enum):
    """Object tracking status"""
    ACTIVE = "active"      # Currently being observed
    MISSING = "missing"    # Not seen for N frames
    INACTIVE = "inactive"  # Permanently lost (too many missing frames)


@dataclass
class ObjectInstance:
    """
    Represents a tracked object instance across multiple frames

    Inspired by ConceptGraphs' MapObject, but simplified for incremental development.
    Stores object state, observations, and features for multi-frame tracking.
    """

    # Unique object ID
    object_id: int

    # Status tracking
    status: ObjectStatus = ObjectStatus.ACTIVE
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    num_observations: int = 0
    num_missing_frames: int = 0

    # Visual features (CLIP embeddings)
    clip_features: Optional[np.ndarray] = None  # Aggregated CLIP feature
    clip_features_history: List[np.ndarray] = field(default_factory=list)

    # Masks (2D segmentation)
    current_mask: Optional[np.ndarray] = None  # Most recent mask (H, W)
    mask_history: List[np.ndarray] = field(default_factory=list)

    # Bounding boxes
    current_bbox: Optional[np.ndarray] = None  # Most recent bbox [x1, y1, x2, y2]
    bbox_history: List[np.ndarray] = field(default_factory=list)

    # Scores
    current_score: float = 0.0  # SAM confidence score
    score_history: List[float] = field(default_factory=list)

    # Spatial information (for future 3D reconstruction)
    points_3d: Optional[np.ndarray] = None  # 3D points (N, 3)
    colors_3d: Optional[np.ndarray] = None  # RGB colors (N, 3)

    # Metadata
    class_id: Optional[int] = None  # For future semantic labeling
    class_name: Optional[str] = None
    confidence: float = 1.0  # Overall confidence in this object

    # Tracking metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        frame_idx: int,
        mask: np.ndarray,
        bbox: np.ndarray,
        score: float,
        clip_feature: np.ndarray,
        points_3d: Optional[np.ndarray] = None,
        colors_3d: Optional[np.ndarray] = None,
    ):
        """
        Update object with new observation

        Args:
            frame_idx: Current frame index
            mask: Binary mask (H, W)
            bbox: Bounding box [x1, y1, x2, y2]
            score: SAM confidence score
            clip_feature: CLIP embedding (D,)
            points_3d: 3D points (N, 3), optional
            colors_3d: RGB colors (N, 3), optional
        """
        # Update status
        self.status = ObjectStatus.ACTIVE
        self.last_seen_frame = frame_idx
        self.num_observations += 1
        self.num_missing_frames = 0

        # Update current state
        self.current_mask = mask
        self.current_bbox = bbox
        self.current_score = score

        # Add to history
        self.mask_history.append(mask)
        self.bbox_history.append(bbox)
        self.score_history.append(score)
        self.clip_features_history.append(clip_feature)

        # Update aggregated CLIP features (running average)
        if self.clip_features is None:
            self.clip_features = clip_feature.copy()
        else:
            # Exponential moving average (give more weight to recent observations)
            alpha = 0.7  # Weight for new observation
            self.clip_features = alpha * clip_feature + (1 - alpha) * self.clip_features
            # Re-normalize
            self.clip_features = self.clip_features / np.linalg.norm(self.clip_features)

        # Update 3D points if provided
        if points_3d is not None:
            if self.points_3d is None:
                self.points_3d = points_3d
                self.colors_3d = colors_3d
            else:
                # Concatenate new points (simple accumulation for now)
                self.points_3d = np.vstack([self.points_3d, points_3d])
                if colors_3d is not None:
                    self.colors_3d = np.vstack([self.colors_3d, colors_3d])

    def mark_missing(self, frame_idx: int):
        """Mark object as missing in current frame"""
        self.num_missing_frames += 1

        # Update status based on missing count
        if self.num_missing_frames >= 10:  # Threshold for inactive
            self.status = ObjectStatus.INACTIVE
        else:
            self.status = ObjectStatus.MISSING

    def get_clip_feature(self) -> np.ndarray:
        """Get current CLIP feature (aggregated)"""
        return self.clip_features

    def get_bbox(self) -> np.ndarray:
        """Get current bounding box"""
        return self.current_bbox

    def get_mask(self) -> np.ndarray:
        """Get current mask"""
        return self.current_mask

    def get_age(self) -> int:
        """Get object age (number of frames since first seen)"""
        return self.last_seen_frame - self.first_seen_frame + 1

    def is_active(self) -> bool:
        """Check if object is currently active"""
        return self.status == ObjectStatus.ACTIVE

    def is_missing(self) -> bool:
        """Check if object is missing"""
        return self.status == ObjectStatus.MISSING

    def is_inactive(self) -> bool:
        """Check if object is inactive"""
        return self.status == ObjectStatus.INACTIVE

    def get_color(self) -> np.ndarray:
        """Get a consistent color for visualization (based on object ID)"""
        np.random.seed(self.object_id)
        return np.random.rand(3)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving"""
        return {
            'object_id': self.object_id,
            'status': self.status.value,
            'first_seen_frame': self.first_seen_frame,
            'last_seen_frame': self.last_seen_frame,
            'num_observations': self.num_observations,
            'num_missing_frames': self.num_missing_frames,
            'clip_features': self.clip_features,
            'current_bbox': self.current_bbox,
            'current_score': self.current_score,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'points_3d': self.points_3d,
            'colors_3d': self.colors_3d,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectInstance':
        """Deserialize from dictionary"""
        obj = cls(
            object_id=data['object_id'],
            status=ObjectStatus(data['status']),
            first_seen_frame=data['first_seen_frame'],
            last_seen_frame=data['last_seen_frame'],
            num_observations=data['num_observations'],
            num_missing_frames=data['num_missing_frames'],
            clip_features=data.get('clip_features'),
            current_bbox=data.get('current_bbox'),
            current_score=data.get('current_score', 0.0),
            class_id=data.get('class_id'),
            class_name=data.get('class_name'),
            confidence=data.get('confidence', 1.0),
            points_3d=data.get('points_3d'),
            colors_3d=data.get('colors_3d'),
            metadata=data.get('metadata', {}),
        )
        return obj

    def __repr__(self) -> str:
        return (f"ObjectInstance(id={self.object_id}, status={self.status.value}, "
                f"observations={self.num_observations}, age={self.get_age()}, "
                f"missing={self.num_missing_frames})")
