"""
Demo 5.1: Scene Graph Construction

Load semantic objects from Phase 4.2 and build a spatial relationship graph.

Features:
- Load semantic objects from JSON (Phase 4.2 output)
- Compute spatial relationships (on, in, near, above, etc.)
- Build minimum spanning tree graph
- Find connected components
- Export scene graph to JSON
- Visualize in Rerun
"""

import json
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
import time

from memspace.scenegraph.scene_graph import SceneGraph
from memspace.scenegraph.spatial_relations import RelationType, SpatialRelation
from memspace.scenegraph.llava_spatial_reasoning import LLaVASpatialReasoner
from memspace.scenegraph.llava_visual_reasoning import LLaVAVisualReasoner


def load_semantic_objects_from_json(json_path: str) -> SceneGraph:
    """
    Load semantic objects from JSON file (Phase 4.2 output) into SceneGraph.

    Args:
        json_path: Path to semantic_objects.json

    Returns:
        SceneGraph populated with objects
    """
    print(f"📂 Loading semantic objects from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create scene graph
    scene_graph = SceneGraph()

    # Load object data
    num_loaded = 0
    for obj_data in data['objects']:
        # Reconstruct bounding box as numpy array
        if 'bounding_box_3d' in obj_data and obj_data['bounding_box_3d'] is not None:
            bbox_dict = obj_data['bounding_box_3d']
            bbox = np.array(bbox_dict['center'] + bbox_dict['size'])
            obj_data['bounding_box_3d'] = bbox

        num_loaded += 1

    # Store objects in scene graph (as simple dicts for now)
    # The SceneGraph inherits from SemanticObjectManager
    # We'll populate its objects list directly for query purposes
    scene_graph._loaded_objects = data['objects']

    print(f"✓ Loaded {num_loaded} semantic objects")
    print(f"   Total objects: {data['metadata']['total_objects']}")
    print(f"   Confirmed objects: {data['metadata']['confirmed_objects']}")

    return scene_graph


@hydra.main(config_path="../memspace/configs", config_name="demo_5_1", version_base=None)
def main(cfg: DictConfig):
    """Main demo function."""
    start_time = time.time()

    print("\n" + "=" * 60)
    print("Demo 5.1: Scene Graph Construction")
    print("=" * 60 + "\n")

    # Load semantic objects from Phase 4.2 output
    json_path = cfg.get('semantic_objects_json', 'outputs/semantic_objects.json')
    scene_graph = load_semantic_objects_from_json(json_path)

    # Override scene_graph's get_confirmed_objects to use loaded data
    def get_confirmed_objects_override(min_observations=2, min_points=100):
        """Return loaded objects that meet filtering criteria."""
        return [
            obj for obj in scene_graph._loaded_objects
            if obj.get('num_observations', 0) >= min_observations
            and obj.get('num_points', 0) >= min_points
        ]

    scene_graph.get_confirmed_objects = get_confirmed_objects_override

    # Compute spatial relationships
    spatial_method = cfg.get('spatial_reasoning_method', 'geometric')
    print(f"\n🔗 Computing spatial relationships (method: {spatial_method})...")

    if spatial_method == 'llava_visual':
        # Use vision-based LLaVA spatial reasoning with annotated images
        from memspace.dataset import get_dataset
        from memspace.models.llava_wrapper import LLaVAWrapper
        import os

        # Initialize dataset
        dataset = get_dataset(cfg.dataset, device=cfg.get('device', 'cuda'))

        # Initialize LLaVA wrapper
        llava_path = os.getenv('LLAVA_PYTHON_PATH')
        llava_ckpt = os.getenv('LLAVA_CKPT_PATH')

        if llava_path is None or llava_ckpt is None:
            print("❌ LLAVA_PYTHON_PATH or LLAVA_CKPT_PATH not set. Falling back to geometric reasoning.")
            spatial_method = 'geometric'
        else:
            # Get LLaVA config with safe defaults
            load_8bit = True
            load_4bit = False
            if 'model' in cfg and hasattr(cfg.model, 'load_8bit'):
                load_8bit = cfg.model.get('load_8bit', True)
                load_4bit = cfg.model.get('load_4bit', False)

            llava_wrapper = LLaVAWrapper(
                model_path=llava_ckpt,
                load_8bit=load_8bit,
                load_4bit=load_4bit,
            )

            reasoner = LLaVAVisualReasoner(
                llava_wrapper=llava_wrapper,
                dataset=dataset,
                use_visual_reasoning=True
            )

            # Get confirmed objects
            objects = scene_graph.get_confirmed_objects(
                min_observations=cfg.get('min_observations', 2),
                min_points=cfg.get('min_points', 100),
            )

            print(f"   Processing {len(objects)} objects with vision-based reasoning...")

            # Compute pairwise relationships
            relationships = reasoner.compute_pairwise_relationships(
                objects,
                min_confidence=cfg.relationships.get('min_confidence', 0.5),
            )

            # Store relationships in scene graph
            scene_graph.relations = relationships
            print(f"✓ Found {len(relationships)} relationships using vision-based LLaVA reasoning")

    if spatial_method == 'llava':
        # Use text-based LLaVA spatial reasoning
        reasoner = LLaVASpatialReasoner(llava_wrapper=None, use_text_reasoning=True)

        # Get confirmed objects
        objects = scene_graph.get_confirmed_objects(
            min_observations=cfg.get('min_observations', 2),
            min_points=cfg.get('min_points', 100),
        )

        print(f"   Processing {len(objects)} objects...")

        # Compute pairwise relationships
        relationships = reasoner.compute_pairwise_relationships(
            objects,
            min_confidence=cfg.relationships.get('min_confidence', 0.5),
        )

        # Store relationships in scene graph
        scene_graph.relations = relationships
        print(f"✓ Found {len(relationships)} relationships using LLaVA reasoning")

        # Print example queries (first 3)
        print("\n🔍 Example LLaVA Queries:")
        for i, obj1 in enumerate(objects[:3]):
            for j, obj2 in enumerate(objects[:3]):
                if i >= j:
                    continue

                label1 = obj1.get('label', 'unknown')
                label2 = obj2.get('label', 'unknown')

                if 'bounding_box_3d' in obj1 and 'bounding_box_3d' in obj2:
                    bbox1 = obj1['bounding_box_3d']
                    bbox2 = obj2['bounding_box_3d']
                    pos1 = bbox1[:3]
                    pos2 = bbox2[:3]
                    size1 = bbox1[3:]
                    size2 = bbox2[3:]

                    rel_str, conf = reasoner.query_relationship_text(
                        label1, pos1, size1,
                        label2, pos2, size2,
                    )

                    print(f"   Query: Object 1 is a \"{label1}\" at position ({pos1[0]:.2f}, {pos1[1]:.2f}, {pos1[2]:.2f})")
                    print(f"          Object 2 is a \"{label2}\" at position ({pos2[0]:.2f}, {pos2[1]:.2f}, {pos2[2]:.2f})")
                    print(f"          → Relationship: \"{rel_str}\" (confidence: {conf:.2f})\n")

    else:
        # Use geometric-based spatial reasoning (default)
        relationship_config = {
            'on_height_tolerance': cfg.relationships.get('on_height_tolerance', 0.05),
            'on_overlap_threshold': cfg.relationships.get('on_overlap_threshold', 0.3),
            'in_containment_threshold': cfg.relationships.get('in_containment_threshold', 0.8),
            'near_distance_threshold': cfg.relationships.get('near_distance_threshold', 0.5),
            'directional_threshold': cfg.relationships.get('directional_threshold', 0.3),
        }

        scene_graph.compute_relationships(
            overlap_threshold=cfg.relationships.get('overlap_threshold', 0.01),
            config=relationship_config,
        )

    # Build MST and connected components
    print("\n🌳 Building minimum spanning tree...")
    components, mst = scene_graph.build_mst_graph()
    print(f"✓ Found {len(components)} connected components")
    for i, comp in enumerate(components):
        print(f"   Component {i}: {len(comp)} objects")

    # Print relationship summary
    print("\n📊 Relationship Summary:")
    rel_summary = scene_graph.get_relation_summary()
    for rel_type, count in rel_summary.items():
        print(f"   {rel_type}: {count}")

    # Print some example relationships
    print("\n🔍 Example Relationships:")
    for i, relation in enumerate(scene_graph.relations[:10]):
        obj_a = scene_graph.get_object_by_id(relation.object_a_id)
        obj_b = scene_graph.get_object_by_id(relation.object_b_id)

        label_a = obj_a.get('label', 'unknown') if obj_a else 'unknown'
        label_b = obj_b.get('label', 'unknown') if obj_b else 'unknown'

        print(f"   {i+1}. \"{label_a}\" {relation.relation_type.value} \"{label_b}\" (conf: {relation.confidence:.2f})")

    # Export scene graph to JSON
    print("\n💾 Exporting scene graph...")
    output_path = cfg.get('output_json', 'outputs/scene_graph.json')
    scene_graph.export_to_json(
        json_path=output_path,
        include_objects=cfg.get('include_objects_in_export', True),
    )
    print(f"✓ Saved scene graph to {output_path}")

    # Visualize in Rerun (optional)
    if cfg.get('use_rerun', False):
        print("\n👁️  Visualizing in Rerun...")
        visualize_scene_graph_rerun(scene_graph, cfg)

    # Print summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✓ Scene graph construction complete in {elapsed_time:.2f}s")
    print(f"   Nodes (objects): {len(scene_graph._loaded_objects)}")
    print(f"   Edges (relationships): {len(scene_graph.relations)}")
    print(f"   Connected components: {len(components)}")
    print("=" * 60 + "\n")


def visualize_scene_graph_rerun(scene_graph: SceneGraph, cfg: DictConfig):
    """
    Visualize scene graph in Rerun.

    Shows:
    - 3D point clouds for each object
    - 3D bounding boxes
    - Relationship edges as lines
    - Labels and metadata
    """
    import rerun as rr

    # Initialize Rerun
    rr.init("memspace_scene_graph", spawn=True)

    # Log objects
    for obj in scene_graph._loaded_objects:
        obj_id = obj['object_id']
        label = obj.get('label', 'unknown')

        # Log point cloud
        if 'point_cloud_points_path' in obj:
            points = np.load(obj['point_cloud_points_path'])
            colors = None
            if 'point_cloud_colors_path' in obj:
                colors = np.load(obj['point_cloud_colors_path'])
                # Ensure colors are in [0, 255] range
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(np.uint8)

            rr.log(
                f"world/scene_graph/objects/obj_{obj_id:03d}/pointcloud",
                rr.Points3D(points, colors=colors, radii=0.01),
            )

        # Log 3D bounding box
        if 'bounding_box_3d' in obj and obj['bounding_box_3d'] is not None:
            bbox = obj['bounding_box_3d']
            center = bbox[:3]
            size = bbox[3:]

            # Create bbox corners
            half_size = size / 2
            corners = np.array([
                center + [-half_size[0], -half_size[1], -half_size[2]],
                center + [+half_size[0], -half_size[1], -half_size[2]],
                center + [+half_size[0], +half_size[1], -half_size[2]],
                center + [-half_size[0], +half_size[1], -half_size[2]],
                center + [-half_size[0], -half_size[1], +half_size[2]],
                center + [+half_size[0], -half_size[1], +half_size[2]],
                center + [+half_size[0], +half_size[1], +half_size[2]],
                center + [-half_size[0], +half_size[1], +half_size[2]],
            ])

            # Log as line strips (12 edges of cube)
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
            ]

            for edge in edges:
                p1, p2 = corners[edge]
                rr.log(
                    f"world/scene_graph/objects/obj_{obj_id:03d}/bbox",
                    rr.LineStrips3D([[p1, p2]], colors=[0, 255, 0]),
                )

        # Log label as text
        if 'bounding_box_3d' in obj:
            bbox = obj['bounding_box_3d']
            center = bbox[:3]
            rr.log(
                f"world/scene_graph/objects/obj_{obj_id:03d}/label",
                rr.TextDocument(
                    f"**{label}**\n"
                    f"ID: {obj_id}\n"
                    f"Observations: {obj.get('num_observations', 0)}\n"
                    f"Points: {obj.get('num_points', 0)}\n"
                    f"Confidence: {obj.get('confidence', 0):.2f}"
                ),
            )

    # Log relationships as lines between object centers
    for relation in scene_graph.relations:
        obj_a = scene_graph.get_object_by_id(relation.object_a_id)
        obj_b = scene_graph.get_object_by_id(relation.object_b_id)

        if obj_a is None or obj_b is None:
            continue

        # Get object centers from loaded data
        obj_a_full = next((o for o in scene_graph._loaded_objects if o['object_id'] == relation.object_a_id), None)
        obj_b_full = next((o for o in scene_graph._loaded_objects if o['object_id'] == relation.object_b_id), None)

        if obj_a_full is None or obj_b_full is None:
            continue

        if 'bounding_box_3d' not in obj_a_full or 'bounding_box_3d' not in obj_b_full:
            continue

        center_a = obj_a_full['bounding_box_3d'][:3]
        center_b = obj_b_full['bounding_box_3d'][:3]

        # Color by relation type
        color = {
            'on': [255, 0, 0],      # Red
            'in': [0, 0, 255],      # Blue
            'near': [255, 255, 0],  # Yellow
            'above': [255, 128, 0], # Orange
            'below': [128, 0, 255], # Purple
        }.get(relation.relation_type.value, [128, 128, 128])  # Gray default

        rr.log(
            f"world/scene_graph/relations/{relation.object_a_id}_{relation.relation_type.value}_{relation.object_b_id}",
            rr.LineStrips3D([[center_a, center_b]], colors=[color]),
        )

    print("✓ Rerun visualization complete")


if __name__ == "__main__":
    main()
