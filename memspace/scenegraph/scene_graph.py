"""
Scene Graph class for managing semantic objects and their spatial relationships.

Extends SemanticObjectManager with graph structure and relationship queries.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from memspace.scenegraph.semantic_object import SemanticObjectManager
from memspace.scenegraph.spatial_relations import (
    SpatialRelation,
    RelationType,
    compute_spatial_relationships,
    compute_bbox_overlap_faiss,
)


class SceneGraph(SemanticObjectManager):
    """
    Scene graph with semantic objects as nodes and spatial relationships as edges.

    Extends SemanticObjectManager with:
    - Spatial relationship edges
    - Graph-based queries
    - Connected components analysis
    - Minimum spanning tree construction
    """

    def __init__(self):
        super().__init__()
        self.relations: List[SpatialRelation] = []
        self.relation_index: Dict[Tuple[int, int], List[SpatialRelation]] = {}

    def add_relation(self, relation: SpatialRelation):
        """Add a spatial relationship edge to the graph."""
        self.relations.append(relation)

        # Index for fast lookup
        key = (relation.object_a_id, relation.object_b_id)
        if key not in self.relation_index:
            self.relation_index[key] = []
        self.relation_index[key].append(relation)

    def get_relations(
        self,
        object_id: Optional[int] = None,
        relation_type: Optional[RelationType] = None,
        min_confidence: float = 0.0,
    ) -> List[SpatialRelation]:
        """
        Query relationships by object ID, type, or confidence.

        Args:
            object_id: Filter to relations involving this object (either A or B)
            relation_type: Filter to this relation type
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching SpatialRelation objects
        """
        results = []

        for relation in self.relations:
            # Filter by object ID
            if object_id is not None:
                if relation.object_a_id != object_id and relation.object_b_id != object_id:
                    continue

            # Filter by relation type
            if relation_type is not None:
                if relation.relation_type != relation_type:
                    continue

            # Filter by confidence
            if relation.confidence < min_confidence:
                continue

            results.append(relation)

        return results

    def get_neighbors(
        self,
        object_id: int,
        relation_type: Optional[RelationType] = None,
    ) -> List[int]:
        """
        Get neighbor object IDs connected by relationships.

        Args:
            object_id: Source object ID
            relation_type: Optional filter by relation type

        Returns:
            List of neighbor object IDs
        """
        neighbors = set()

        for relation in self.get_relations(object_id=object_id, relation_type=relation_type):
            if relation.object_a_id == object_id:
                neighbors.add(relation.object_b_id)
            else:
                neighbors.add(relation.object_a_id)

        return list(neighbors)

    def compute_relationships(
        self,
        overlap_threshold: float = 0.01,
        config: Optional[Dict] = None,
    ):
        """
        Compute spatial relationships between all objects in the graph.

        Args:
            overlap_threshold: Only check relationships for objects with overlap > threshold
            config: Configuration for relationship detection parameters
        """
        print("🔗 Computing spatial relationships...")

        # Get confirmed objects
        objects = self.get_confirmed_objects(min_observations=2, min_points=100)

        if len(objects) == 0:
            print("⚠️  No objects to compute relationships for")
            return

        print(f"   Found {len(objects)} objects")

        # Prepare objects for overlap computation
        # Need to load point clouds from disk
        objects_with_pcd = []
        for obj in objects:
            # Load point cloud from .npy file if path exists
            if 'point_cloud_points_path' in obj:
                import open3d as o3d
                points_path = obj['point_cloud_points_path']
                points = np.load(points_path)

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                # Load colors if available
                if 'point_cloud_colors_path' in obj:
                    colors = np.load(obj['point_cloud_colors_path'])
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                obj_with_pcd = obj.copy()
                obj_with_pcd['pcd'] = pcd
                objects_with_pcd.append(obj_with_pcd)
            else:
                print(f"⚠️  Object {obj['object_id']} has no point cloud path, skipping")

        if len(objects_with_pcd) == 0:
            print("⚠️  No objects with point clouds to compute relationships for")
            return

        print(f"   Loaded {len(objects_with_pcd)} point clouds")

        # Compute overlap matrix using FAISS (like ConceptGraphs)
        print("   Computing bbox overlaps with FAISS...")
        overlap_matrix = compute_bbox_overlap_faiss(
            objects_with_pcd,
            distance_threshold=0.025,  # 2.5cm threshold
        )
        print(f"   Overlap matrix shape: {overlap_matrix.shape}")

        # Compute relationships
        print("   Detecting spatial relationships...")
        relations = compute_spatial_relationships(
            objects_with_pcd,
            overlap_matrix=overlap_matrix,
            overlap_threshold=overlap_threshold,
            config=config,
        )

        # Add relations to graph
        for relation in relations:
            self.add_relation(relation)

        print(f"✓ Found {len(relations)} spatial relationships")

    def build_mst_graph(self) -> Tuple[List[List[int]], np.ndarray]:
        """
        Build minimum spanning tree graph from relationships.

        Similar to ConceptGraphs' approach:
        1. Build weighted adjacency matrix from relationship confidences
        2. Compute minimum spanning tree
        3. Find connected components

        Returns:
            (components, mst): List of connected components and MST adjacency matrix
        """
        objects = self.get_confirmed_objects(min_observations=2, min_points=100)
        num_objects = len(objects)

        if num_objects == 0:
            return [], np.array([])

        # Create object ID to index mapping
        id_to_idx = {obj['object_id']: idx for idx, obj in enumerate(objects)}
        idx_to_id = {idx: obj['object_id'] for idx, obj in enumerate(objects)}

        # Build adjacency matrix from relationships
        weights = []
        rows = []
        cols = []

        for relation in self.relations:
            # Skip if objects not in confirmed set
            if relation.object_a_id not in id_to_idx or relation.object_b_id not in id_to_idx:
                continue

            i = id_to_idx[relation.object_a_id]
            j = id_to_idx[relation.object_b_id]

            if i == j:
                continue

            # Use confidence as edge weight
            weight = relation.confidence

            # Add edge (symmetric)
            weights.append(weight)
            rows.append(i)
            cols.append(j)
            weights.append(weight)
            rows.append(j)
            cols.append(i)

        # Create sparse adjacency matrix
        adjacency_matrix = csr_matrix(
            (weights, (rows, cols)),
            shape=(num_objects, num_objects),
        )

        # Find minimum spanning tree
        mst = minimum_spanning_tree(adjacency_matrix)

        # Find connected components
        n_components, labels = connected_components(mst)

        # Group objects by component
        components = []
        for component_id in range(n_components):
            component_indices = np.where(labels == component_id)[0]
            component_object_ids = [idx_to_id[idx] for idx in component_indices]
            components.append(component_object_ids)

        return components, mst.toarray()

    def query_objects_by_relation(
        self,
        query: str,
        relation_type: RelationType,
        min_confidence: float = 0.5,
    ) -> List[Dict]:
        """
        Query objects using natural language + relation type.

        Example: query="table", relation_type=RelationType.ON
        Returns objects that are "on" a table.

        Args:
            query: Natural language query (label or CLIP text)
            relation_type: Type of spatial relationship
            min_confidence: Minimum relation confidence

        Returns:
            List of matching objects
        """
        # First find objects matching the query
        target_objects = self.get_objects_by_label(query, min_score=0.5)

        if len(target_objects) == 0:
            # Try CLIP similarity
            target_objects = self.query_by_clip_similarity(query, top_k=5)

        if len(target_objects) == 0:
            return []

        # Find objects with the specified relationship to target objects
        result_ids = set()

        for target in target_objects:
            target_id = target['object_id']

            # Get all relations involving this object
            relations = self.get_relations(
                object_id=target_id,
                relation_type=relation_type,
                min_confidence=min_confidence,
            )

            # Collect related object IDs
            for relation in relations:
                if relation.object_b_id == target_id:
                    # Object A has relation to target
                    result_ids.add(relation.object_a_id)
                elif relation.object_a_id == target_id:
                    # Object B has relation to target
                    result_ids.add(relation.object_b_id)

        # Get full object data
        results = []
        for obj_id in result_ids:
            obj = self.get_object_by_id(obj_id)
            if obj is not None:
                results.append(obj)

        return results

    def export_to_json(
        self,
        json_path: str = "outputs/scene_graph.json",
        include_objects: bool = True,
    ) -> str:
        """
        Export scene graph to JSON.

        Args:
            json_path: Path to output JSON file
            include_objects: Whether to include full object data

        Returns:
            Path to saved JSON file
        """
        # Get confirmed objects
        objects = self.get_confirmed_objects(min_observations=2, min_points=100)

        # Build MST and components
        components, mst = self.build_mst_graph()

        # Prepare export data
        export_data = {
            'metadata': {
                'num_nodes': len(objects),
                'num_edges': len(self.relations),
                'num_components': len(components),
            },
            'statistics': self.get_statistics(),
            'components': [
                {
                    'component_id': i,
                    'object_ids': comp,
                    'size': len(comp),
                }
                for i, comp in enumerate(components)
            ],
            'relations': [
                {
                    'object_a_id': rel.object_a_id,
                    'object_b_id': rel.object_b_id,
                    'relation_type': rel.relation_type.value,
                    'confidence': float(rel.confidence),
                    'metadata': rel.metadata,
                }
                for rel in self.relations
            ],
        }

        # Optionally include object data
        if include_objects:
            # Convert numpy arrays to lists for JSON serialization
            objects_serializable = []
            for obj in objects:
                obj_copy = obj.copy()
                # Convert bounding box numpy array to list
                if 'bounding_box_3d' in obj_copy and isinstance(obj_copy['bounding_box_3d'], np.ndarray):
                    bbox = obj_copy['bounding_box_3d']
                    obj_copy['bounding_box_3d'] = {
                        'center': bbox[:3].tolist(),
                        'size': bbox[3:].tolist(),
                    }
                objects_serializable.append(obj_copy)
            export_data['objects'] = objects_serializable

        # Save to file
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return json_path

    def get_object_by_id(self, object_id: int) -> Optional[Dict]:
        """Get object data by ID."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return {
                    'object_id': obj.object_id,
                    'label': obj.label,
                    'bounding_box_3d': obj.bounding_box_3d,
                    'num_observations': obj.num_observations,
                    'confidence': obj.confidence,
                }
        return None

    def get_relation_summary(self) -> Dict[str, int]:
        """Get summary of relationships by type."""
        summary = defaultdict(int)

        for relation in self.relations:
            summary[relation.relation_type.value] += 1

        return dict(summary)
