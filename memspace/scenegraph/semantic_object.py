"""
Semantic Object - Complete 3D object representation with semantic labels

This module extends ObjectInstance to include:
- 3D point cloud reconstruction
- Semantic labels from VLM
- 3D bounding boxes
- Confidence scoring
- Query interface
"""

import numpy as np
import open3d as o3d
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from .object_instance import ObjectInstance, ObjectStatus
from ..reconstruction.pointcloud_fusion import PointCloudAccumulator


class SemanticObject(ObjectInstance):
    """
    Complete semantic 3D object representation

    Combines:
    - Object tracking (from ObjectInstance)
    - 3D reconstruction (PointCloudAccumulator)
    - Semantic labeling (Florence-2 captions)
    - Query capabilities (CLIP features)

    This is the core data structure for scene graphs and spatial queries.
    """

    def __init__(
        self,
        object_id: int,
        voxel_size: float = 0.02,
        **kwargs
    ):
        """
        Initialize Semantic Object

        Args:
            object_id: Unique object ID
            voxel_size: Voxel size for point cloud downsampling
            **kwargs: Additional arguments passed to ObjectInstance
        """
        super().__init__(object_id=object_id, **kwargs)

        # 3D Reconstruction
        self.point_cloud_accumulator = PointCloudAccumulator(voxel_size=voxel_size)
        self.bounding_box_3d: Optional[np.ndarray] = None  # [center_x, center_y, center_z, w, h, d]
        self.bounding_box_3d_corners: Optional[np.ndarray] = None  # 8 corners (8, 3)

        # Semantic Information
        self.caption_history: List[str] = []
        self.label: Optional[str] = None
        self.semantic_confidence: float = 0.0

        # Overall confidence
        self.tracking_confidence: float = 0.0

    def integrate_frame(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ):
        """
        Integrate masked RGB-D frame into object's point cloud

        Args:
            color: RGB image (H, W, 3), uint8
            depth: Masked depth (H, W), float32 (non-object pixels should be 0)
            intrinsics: Camera intrinsics (3, 3) or (4, 4)
            extrinsics: Camera-to-world pose (4, 4)
        """
        self.point_cloud_accumulator.integrate(
            color=color,
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    def add_caption(self, caption: str):
        """Add a caption from VLM to this object"""
        if len(caption) >= 3:  # Minimum caption length
            self.caption_history.append(caption)

    def consolidate_label(self):
        """
        Consolidate multiple captions into single label

        Uses longest caption strategy (simple and effective)
        """
        if not self.caption_history:
            self.label = f"object_{self.object_id}"
            self.semantic_confidence = 0.0
            return

        # Use longest caption as label
        self.label = max(self.caption_history, key=len)

        # Compute semantic confidence based on caption consistency
        # Higher confidence if captions are similar/consistent
        if len(self.caption_history) == 1:
            self.semantic_confidence = 0.5
        else:
            # Simple heuristic: count how many captions contain key words from label
            label_words = set(self.label.lower().split())
            consistent_count = sum(
                1 for cap in self.caption_history
                if any(word in cap.lower() for word in label_words)
            )
            self.semantic_confidence = consistent_count / len(self.caption_history)

    def compute_3d_bbox(self):
        """
        Compute axis-aligned 3D bounding box from point cloud

        Sets:
            self.bounding_box_3d: [center_x, center_y, center_z, width, height, depth]
            self.bounding_box_3d_corners: 8 corner points (8, 3)
        """
        pcd = self.get_point_cloud()
        if pcd is None or not pcd.has_points():
            return

        # Get axis-aligned bounding box
        aabb = pcd.get_axis_aligned_bounding_box()

        # Extract center and extents
        center = aabb.get_center()
        extent = aabb.get_extent()

        self.bounding_box_3d = np.array([
            center[0], center[1], center[2],
            extent[0], extent[1], extent[2],
        ])

        # Compute 8 corners
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()

        self.bounding_box_3d_corners = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
        ])

    def compute_confidence(self):
        """
        Compute overall object confidence from tracking and semantics

        Tracking confidence: based on num_observations
        Semantic confidence: based on caption consistency
        Overall: average of both
        """
        # Tracking confidence: sigmoid-like function of observations
        # More observations = higher confidence
        self.tracking_confidence = min(1.0, self.num_observations / 10.0)

        # Overall confidence
        self.confidence = (self.tracking_confidence + self.semantic_confidence) / 2.0

    def get_point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """Get the accumulated point cloud"""
        pcd = self.point_cloud_accumulator.get_point_cloud()
        if pcd is None or not pcd.has_points():
            return None
        return pcd

    def get_num_points(self) -> int:
        """Get number of points in point cloud"""
        pcd = self.get_point_cloud()
        if pcd is None:
            return 0
        return len(pcd.points)

    def finalize(self):
        """
        Finalize object after all frames processed

        Consolidates label, computes 3D bbox, and updates confidence
        """
        self.consolidate_label()
        self.compute_3d_bbox()
        self.compute_confidence()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (extends ObjectInstance.to_dict())"""
        data = super().to_dict()

        # Add semantic 3D attributes
        data.update({
            'label': self.label,
            'caption_history': self.caption_history,
            'semantic_confidence': self.semantic_confidence,
            'tracking_confidence': self.tracking_confidence,
            'bounding_box_3d': self.bounding_box_3d,
            'bounding_box_3d_corners': self.bounding_box_3d_corners,
            'num_points': self.get_num_points(),
        })

        # Add point cloud if available
        pcd = self.get_point_cloud()
        if pcd is not None and pcd.has_points():
            data['point_cloud_points'] = np.asarray(pcd.points)
            if pcd.has_colors():
                data['point_cloud_colors'] = np.asarray(pcd.colors)

        return data

    def to_json_dict(self, output_dir: str = "outputs") -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary with links to external files

        Saves large arrays (point clouds, CLIP features) to separate .npy files
        and includes paths in the JSON.

        Args:
            output_dir: Directory to save .npy files

        Returns:
            Dictionary ready for JSON serialization
        """
        import os
        from pathlib import Path

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Basic attributes (JSON-serializable)
        data = {
            'object_id': int(self.object_id),
            'label': self.label,
            'caption_history': self.caption_history,
            'status': self.status.value,
            'first_seen_frame': int(self.first_seen_frame),
            'last_seen_frame': int(self.last_seen_frame),
            'num_observations': int(self.num_observations),
            'semantic_confidence': float(self.semantic_confidence),
            'tracking_confidence': float(self.tracking_confidence),
            'confidence': float(self.confidence),
            'num_points': int(self.get_num_points()),
        }

        # Bounding boxes (convert numpy to lists)
        if self.bounding_box_3d is not None:
            data['bounding_box_3d'] = {
                'center': self.bounding_box_3d[:3].tolist(),
                'size': self.bounding_box_3d[3:].tolist(),
            }
        else:
            data['bounding_box_3d'] = None

        if self.bounding_box_3d_corners is not None:
            data['bounding_box_3d_corners'] = self.bounding_box_3d_corners.tolist()
        else:
            data['bounding_box_3d_corners'] = None

        if self.current_bbox is not None:
            data['current_bbox_2d'] = self.current_bbox.tolist()
        else:
            data['current_bbox_2d'] = None

        # Save CLIP features to .npy and link
        if self.clip_features is not None:
            clip_path = f"{output_dir}/obj_{self.object_id:03d}_clip_features.npy"
            np.save(clip_path, self.clip_features)
            data['clip_features_path'] = clip_path
            data['clip_features_shape'] = list(self.clip_features.shape)
        else:
            data['clip_features_path'] = None
            data['clip_features_shape'] = None

        # Save point cloud to .npy and link
        pcd = self.get_point_cloud()
        if pcd is not None and pcd.has_points():
            points = np.asarray(pcd.points)
            points_path = f"{output_dir}/obj_{self.object_id:03d}_points.npy"
            np.save(points_path, points)
            data['point_cloud_points_path'] = points_path
            data['point_cloud_points_shape'] = list(points.shape)

            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                colors_path = f"{output_dir}/obj_{self.object_id:03d}_colors.npy"
                np.save(colors_path, colors)
                data['point_cloud_colors_path'] = colors_path
                data['point_cloud_colors_shape'] = list(colors.shape)
            else:
                data['point_cloud_colors_path'] = None
                data['point_cloud_colors_shape'] = None
        else:
            data['point_cloud_points_path'] = None
            data['point_cloud_points_shape'] = None
            data['point_cloud_colors_path'] = None
            data['point_cloud_colors_shape'] = None

        return data

    def __repr__(self) -> str:
        return (f"SemanticObject(id={self.object_id}, label='{self.label}', "
                f"points={self.get_num_points()}, observations={self.num_observations}, "
                f"confidence={self.confidence:.2f}, status={self.status.value})")


class SemanticObjectManager:
    """
    Manages collection of semantic objects for scene understanding

    Provides:
    - Object lookup by ID, label, or CLIP similarity
    - Filtering by confidence, points, observations
    - Export for scene graphs
    """

    def __init__(self, objects: Optional[List[SemanticObject]] = None):
        """
        Initialize manager

        Args:
            objects: List of SemanticObject instances
        """
        self.objects = objects if objects is not None else []

    def add_object(self, obj: SemanticObject):
        """Add an object to the collection"""
        self.objects.append(obj)

    def get_object_by_id(self, object_id: int) -> Optional[SemanticObject]:
        """Get object by ID"""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_objects_by_label(self, label: str, partial_match: bool = True) -> List[SemanticObject]:
        """
        Get objects matching a label

        Args:
            label: Label to match (e.g., "table", "chair")
            partial_match: If True, match substrings (e.g., "table" matches "wooden table")

        Returns:
            List of matching objects
        """
        if partial_match:
            return [obj for obj in self.objects if obj.label and label.lower() in obj.label.lower()]
        else:
            return [obj for obj in self.objects if obj.label and label.lower() == obj.label.lower()]

    def query_by_clip(
        self,
        query_features: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[SemanticObject, float]]:
        """
        Query objects by CLIP similarity

        Args:
            query_features: CLIP features for query (1024D, normalized)
            top_k: Return top K matches
            threshold: Minimum similarity threshold

        Returns:
            List of (object, similarity) tuples, sorted by similarity
        """
        results = []
        for obj in self.objects:
            if obj.clip_features is not None:
                sim = float(np.dot(query_features, obj.clip_features))
                if sim >= threshold:
                    results.append((obj, sim))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_confirmed_objects(
        self,
        min_observations: int = 2,
        min_points: int = 100,
        min_confidence: float = 0.0,
    ) -> List[SemanticObject]:
        """
        Get confirmed objects based on filtering criteria

        Args:
            min_observations: Minimum number of tracking observations
            min_points: Minimum number of 3D points
            min_confidence: Minimum overall confidence

        Returns:
            List of objects meeting all criteria
        """
        return [
            obj for obj in self.objects
            if (obj.num_observations >= min_observations and
                obj.get_num_points() >= min_points and
                obj.confidence >= min_confidence)
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the object collection"""
        confirmed = self.get_confirmed_objects()

        return {
            'total_objects': len(self.objects),
            'confirmed_objects': len(confirmed),
            'active_objects': len([o for o in self.objects if o.is_active()]),
            'objects_with_labels': len([o for o in self.objects if o.label]),
            'avg_points_per_object': np.mean([o.get_num_points() for o in self.objects]) if self.objects else 0,
            'avg_observations_per_object': np.mean([o.num_observations for o in self.objects]) if self.objects else 0,
            'avg_confidence': np.mean([o.confidence for o in self.objects]) if self.objects else 0,
        }

    def export_for_scene_graph(self) -> List[Dict[str, Any]]:
        """
        Export semantic objects in format suitable for scene graph construction

        Returns:
            List of object dictionaries with all relevant attributes
        """
        return [obj.to_dict() for obj in self.get_confirmed_objects()]

    def export_to_json(
        self,
        json_path: str = "outputs/semantic_objects.json",
        output_dir: str = "outputs/semantic_objects_data",
        min_observations: int = 2,
        min_points: int = 100,
    ):
        """
        Export semantic objects to JSON with separate .npy files for arrays

        Args:
            json_path: Path to save JSON file
            output_dir: Directory to save .npy files for point clouds and features
            min_observations: Minimum observations to export
            min_points: Minimum points to export

        File Structure:
            outputs/
            ├── semantic_objects.json              # Main JSON with metadata
            └── semantic_objects_data/
                ├── obj_000_points.npy             # Point cloud points
                ├── obj_000_colors.npy             # Point cloud colors
                ├── obj_000_clip_features.npy      # CLIP embeddings
                ├── obj_001_points.npy
                ├── obj_001_colors.npy
                ├── obj_001_clip_features.npy
                └── ...
        """
        import json
        from pathlib import Path

        # Get confirmed objects
        confirmed = self.get_confirmed_objects(
            min_observations=min_observations,
            min_points=min_points,
        )

        # Export each object
        objects_data = []
        for obj in confirmed:
            obj_dict = obj.to_json_dict(output_dir=output_dir)
            objects_data.append(obj_dict)

        # Create summary metadata
        export_data = {
            'metadata': {
                'total_objects': len(self.objects),
                'confirmed_objects': len(confirmed),
                'exported_objects': len(objects_data),
                'min_observations': min_observations,
                'min_points': min_points,
                'data_directory': output_dir,
            },
            'statistics': self.get_statistics(),
            'objects': objects_data,
        }

        # Save JSON
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return json_path, len(objects_data)

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    def __getitem__(self, idx: int) -> SemanticObject:
        return self.objects[idx]
