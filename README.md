# Hierarchical Scene Graphs for Persistent Robotic Memory

[![Version](https://img.shields.io/badge/version-1.4-blue.svg)](https://github.com/yourusername/yourrepo)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen.svg)](http://wiki.ros.org/noetic)

> **Autonomous Spatio-Temporal Semantic Mapping for Language-Grounded Robotic Reasoning**

An advanced mapping system that enables robots to explore unknown environments and build persistent, queryable 3D scene graphs. The system combines geometric segmentation, vision-language models, and hierarchical spatial representation to support complex natural language queries like *"Find the blue recycling bin"* — even for objects the robot has never been trained on.

---

## 🎯 Overview

This project implements a **label-agnostic 3D scene graph** that allows robots to:
- **Discover and map** objects they've never encountered before
- **Remember object locations** even when they move
- **Answer natural language queries** about their environment
- **Track temporal changes** in dynamic scenes
- **Navigate** using hierarchical spatial reasoning

The system is built on a **Hydra-DSG** (Dynamic Scene Graph) architecture with five hierarchical layers, from dense metric reconstruction to building-level topology.

---

## ✨ Key Features

- 🔍 **Zero-Shot Object Discovery**: Maps unknown objects using class-agnostic segmentation (SAM) + vision-language embeddings (CLIP)
- 🧠 **Persistent Memory**: Tracks objects across time with temporal edges and persistence scoring
- 💬 **Natural Language Queries**: Late-binding text-to-vision search for intuitive interaction
- 🏗️ **Hierarchical Representation**: 5-layer architecture from metric mesh to building topology
- 📍 **Dynamic Scene Understanding**: Handles moving objects with spatio-temporal nodes
- 🎯 **Multi-View Integration**: Aggregates observations across viewpoints for robust embeddings

---

## 🏗️ System Architecture

### Hierarchical Spatial Representation

| Layer | Type | Data Structure | Purpose |
|:-----:|:-----|:---------------|:--------|
| **L4** | Building | Topological Graph | Represents the global workspace, connecting multiple floors or wings |
| **L3** | Rooms | Semantic Regions | Partitioned using "Trigger B" (semantic distribution shifts) and GVG bottlenecks |
| **L2** | Places | GVG Nodes | Sparse representation of navigable free space and connectivity for path planning |
| **L1** | Objects | Spatio-Temporal Nodes | Label-agnostic nodes storing centroids, embeddings, and temporal history |
| **L0** | Metric | TSDF / Mesh | Dense 3D surface reconstruction for collision avoidance and visibility ray-casting |

### Two-Stage Discovery Pipeline

```
RGB-D Input → SAM Segmentation → CLIP Encoding → Multi-View Fusion → Scene Graph Update
```

1. **Class-Agnostic Segmentation**: SAM identifies object boundaries based on geometric saliency
2. **Feature Extraction**: CLIP encodes visual regions into language-aligned embeddings
3. **Multi-View Integration**: Aggregates observations across viewpoints using tracking IDs
4. **Graph Integration**: Updates spatio-temporal scene graph with new/updated nodes

---

## 🔬 Technical Deep Dive

### Spatio-Temporal Object Nodes (L1)

Each object is represented as a **stateful entity** with:

```python
{
  "instance_id": "OBJ_ID_402",                    # Persistent unique identifier
  "visual_embedding": [...],                       # CLIP feature vector (512-dim)
  "centroid": {"x": 2.34, "y": 1.12, "z": 0.85},  # 3D position
  "last_observed": "2024-01-13T10:23:45Z",         # Timestamp
  "persistence_score": 0.87,                       # Confidence [0,1]
  "dynamic_class": "Movable",                      # LLM-inferred affordance
  "temporal_edges": ["OBJ_ID_402_t1", "..."]       # Historical trajectory
}
```

### Data Association Algorithm

Objects are matched using a **hybrid metric-semantic cost function**:

```
C = w₁ · SpatialDistance + w₂ · (1 - CosineSimilarity(embeddings))
```

### Persistence Management

The system maintains object memory through three states:

- **Active**: Currently observed and associated with sensor data
- **Missing**: Expected but not visible (occlusion/temporary)
- **Past/Inactive**: Persistence score reached zero → archived with temporal links

<!-- **Visibility verification** uses ray-casting through the L0 mesh to distinguish occlusion from disappearance. -->

---

## 💡 Use Cases

### Natural Language Querying

```python
query = "Find the blue recycling bin"

# 1. Encode query to CLIP text embedding
text_embedding = clip.encode_text(query)

# 2. Vector search over scene graph
results = faiss_index.search(text_embedding, k=5)

# 3. Return location + temporal history
>> Object found at (x=3.2, y=-1.5, z=0.0)
>> Previously seen at: [(x=2.1, y=-1.3, z=0.0, t=-2h), ...]
```

### Dynamic Object Tracking

The system automatically detects when objects move and creates temporal edges:

```
OBJ_ID_042 (t=0, kitchen) → OBJ_ID_042 (t=10min, living room)
```

This enables queries like: *"Where was the red mug 30 minutes ago?"*

---


### Unified SemanticObject Data Structure

```python
SemanticObject:
  # Identity & Tracking
  - object_id: int                         # Unique persistent ID
  - status: ObjectStatus                   # PENDING, CONFIRMED, MISSING
  - num_observations: int                  # Number of frames tracked
  - first_seen_frame: int                  # First detection
  - last_seen_frame: int                   # Most recent detection

  # Semantic Information
  - label: str                             # Final consolidated label (e.g., "wooden table")
  - caption_history: List[str]             # Multi-view captions from Florence-2
  - semantic_confidence: float             # Labeling confidence [0,1]

  # Visual Features
  - clip_features: np.ndarray              # 1024D CLIP embedding (ViT-H-14)

  # 3D Geometry
  - point_cloud: PointCloudAccumulator     # Voxel-downsampled 3D reconstruction
  - bounding_box_3d: np.ndarray            # [center_x, center_y, center_z, width, height, depth]

  # Confidence Scoring
  - tracking_confidence: float             # Geometric consistency score
  - confidence: float                      # Combined tracking + semantic score
```

### Complete Pipeline Integration

```
RGB-D Stream → SAM Masks → CLIP Features → Object Tracking
                                              ↓
                                        Per-Object 3D Reconstruction
                                              ↓
                                        Florence-2 Captioning
                                              ↓
                                        SemanticObject with:
                                        - Persistent 3D geometry
                                        - Natural language labels
                                        - Visual embeddings
```

### JSON Export with External Arrays

Semantic objects are exported to JSON with large arrays stored separately for efficiency:

**File Structure:**
```
outputs/
├── semantic_objects.json              # Metadata (42KB)
└── semantic_objects_data/
    ├── obj_000_points.npy             # Point cloud XYZ (2.8MB)
    ├── obj_000_colors.npy             # Point cloud RGB (2.8MB)
    ├── obj_000_clip_features.npy      # CLIP embedding (4.2KB)
    └── ...
```

**JSON Schema:**
```json
{
  "metadata": {
    "total_objects": 22,
    "confirmed_objects": 22,
    "exported_objects": 22,
    "min_observations": 3,
    "min_points": 100
  },
  "statistics": {
    "active_objects": 22,
    "confirmed_objects": 22,
    "avg_observations_per_object": 8.5,
    "avg_points_per_object": 131432,
    "objects_with_labels": 22
  },
  "objects": [
    {
      "object_id": 0,
      "label": "wooden table with white surface",
      "caption_history": ["table", "wooden table with white surface"],
      "status": "confirmed",
      "first_seen_frame": 0,
      "last_seen_frame": 9,
      "num_observations": 10,
      "semantic_confidence": 0.92,
      "tracking_confidence": 0.88,
      "confidence": 0.90,
      "num_points": 145823,
      "bounding_box_3d": {
        "center": [1.23, 0.54, 0.75],
        "size": [0.80, 0.60, 0.72]
      },
      "clip_features_path": "outputs/semantic_objects_data/obj_000_clip_features.npy",
      "clip_features_shape": [1024],
      "point_cloud_points_path": "outputs/semantic_objects_data/obj_000_points.npy",
      "point_cloud_points_shape": [145823, 3],
      "point_cloud_colors_path": "outputs/semantic_objects_data/obj_000_colors.npy"
    }
  ]
}
```

### SemanticObjectManager Query Interface

```python
# Query by semantic label
tables = manager.get_objects_by_label("table", min_score=0.8)

# Query by CLIP similarity
similar = manager.query_by_clip_similarity(
    query_text="blue recycling bin",
    top_k=5
)

# Filter by spatial constraints
objects_in_roi = manager.filter_objects(
    min_observations=5,
    min_points=1000,
    bbox_filter=lambda bbox: bbox[2] > 0.5  # Center Z > 0.5m
)

# Export to JSON
manager.export_to_json(
    json_path="outputs/semantic_objects.json",
    output_dir="outputs/semantic_objects_data",
    min_observations=3,
    min_points=100
)
```


### Key Advantages

1. **Unified Representation**: Single data structure combining tracking, geometry, and semantics
2. **Efficient Storage**: Large arrays stored as .npy files, not in JSON
3. **Multi-View Robustness**: Caption history aggregates labels across viewpoints
4. **3D Queryable**: Spatial filtering with 3D bounding boxes
5. **Persistent Memory**: Object IDs maintained across frames
6. **Natural Language Grounded**: CLIP features + Florence-2 captions enable language queries

---

## 🚀 Quick Start

### Running Demos

All demos support Hydra config overrides for flexibility:

#### Phase 1: Data Pipeline
```bash
# Demo 1.2: RGB-D Data Pipeline & Visualization
python demos/demo_1_2_data_pipeline.py dataset.max_frames=10
```

#### Phase 2: Perception & Tracking
```bash
# Demo 2.1: SAM Segmentation
python demos/demo_2_1_sam_segmentation.py dataset.max_frames=10

# Demo 2.2: CLIP Embeddings
python demos/demo_2_2_clip_embeddings.py dataset.max_frames=10

# Demo 2.3: Object Tracking
python demos/demo_2_3_object_tracking.py dataset.max_frames=20
```

#### Phase 3: 3D Reconstruction
```bash
# Demo 3.1: TSDF Reconstruction
python demos/demo_3_1_tsdf_reconstruction.py dataset.max_frames=50

# Demo 3.2: Point Cloud Reconstruction
python demos/demo_3_2_pointcloud_reconstruction.py dataset.max_frames=50

# Demo 3.3: Object-level 3D Reconstruction
python demos/demo_3_3_object_reconstruction.py dataset.max_frames=30
```

#### Phase 4: Semantic Understanding
```bash
# Demo 4.1: Object Labeling with Florence-2
python demos/demo_4_1_object_labeling.py dataset.max_frames=20

# Demo 4.1: Object Labeling with LLaVA (requires setup)
export LLAVA_PYTHON_PATH=/path/to/LLaVA
export LLAVA_CKPT_PATH=/path/to/llava-v1.5-7b
python demos/demo_4_1_object_labeling.py vlm_type=llava dataset.max_frames=10

# Demo 4.2: Semantic Objects
python demos/demo_4_2_semantic_objects.py dataset.max_frames=20
```

#### Phase 5: Scene Graph Construction
```bash
# Demo 5.1: Scene Graph (Geometric reasoning - default)
python demos/demo_5_1_scene_graph.py

# Demo 5.1: Scene Graph (LLaVA text-based spatial reasoning)
export LLAVA_PYTHON_PATH=/path/to/LLaVA
export LLAVA_CKPT_PATH=/path/to/llava-v1.5-7b
python demos/demo_5_1_scene_graph.py spatial_reasoning_method=llava

# Demo 5.1: Scene Graph (LLaVA vision-based spatial reasoning)
export LLAVA_PYTHON_PATH=/path/to/LLaVA
export LLAVA_CKPT_PATH=/path/to/llava-v1.5-7b
python demos/demo_5_1_scene_graph.py spatial_reasoning_method=llava_visual
```

### Common Configuration Options

All demos support these Hydra config overrides:

```bash
# Dataset options
dataset.max_frames=N          # Limit number of frames
dataset.stride=N              # Process every Nth frame
dataset.dataset_path=/path    # Override dataset path

# Visualization options
rerun_spawn=false             # Don't spawn Rerun viewer
use_rerun=false               # Disable Rerun completely

# Device options
device=cuda                   # Use GPU (default)
device=cpu                    # Use CPU
```

### Example Usage

```bash
# Quick test with 5 frames, no viewer
python demos/demo_2_1_sam_segmentation.py dataset.max_frames=5 rerun_spawn=false

# Full scene reconstruction with custom stride
python demos/demo_3_1_tsdf_reconstruction.py dataset.stride=5 dataset.max_frames=100

# Scene graph with different dataset
python demos/demo_5_1_scene_graph.py dataset.dataset_path=/path/to/other/scene
```

For detailed demo documentation, see [`demos/README.md`](demos/README.md).

---

## 🛠️ Technical Stack

- **Segmentation**: SAM (Segment Anything Model)
- **Vision-Language**: CLIP (Contrastive Language-Image Pre-training)
- **Object Captioning**: Florence-2 Vision-Language Model
- **3D Reconstruction**: Open3D Point Cloud Accumulation with Voxel Downsampling
- **Graph Structure**: Hydra-DSG framework
- **Vector Search**: FAISS for efficient similarity search
- **Spatial Reasoning**: GVG (Generalized Voronoi Graph) for topology

---

## 📊 Performance Characteristics

- **Modularity**: Decoupled segmentation and embedding stages
- **Computational Cost**: Higher than unified models (multiple forward passes)
- **Robustness**: Multi-view aggregation filters noise and lighting variations
- **Scalability**: Hierarchical structure enables efficient large-scale mapping

---

