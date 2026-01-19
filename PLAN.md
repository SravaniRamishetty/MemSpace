# MemSpace Development Plan
## Building 3D Semantic Scene Graph - Incremental Development with Demos

**Last Updated:** 2026-01-14

---

## Project Structure (Following ConceptGraphs Modularity)

```
MemSpace/
├── venv/                         # Python virtual environment (not in git)
├── memspace/                     # Main package
│   ├── __init__.py
│   ├── configs/                  # Hydra configuration files
│   │   ├── base.yaml
│   │   ├── base_paths.yaml
│   │   ├── dataset/
│   │   │   ├── replica.yaml
│   │   │   └── record3d.yaml
│   │   ├── model/
│   │   │   ├── sam.yaml
│   │   │   └── clip.yaml
│   │   └── demo_*.yaml          # Config for each demo
│   ├── dataset/                  # Data loading utilities
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── replica_dataset.py
│   │   └── rgbd_utils.py
│   ├── models/                   # Model wrappers
│   │   ├── __init__.py
│   │   ├── sam_wrapper.py
│   │   └── clip_wrapper.py
│   ├── slam/                     # SLAM and mapping
│   │   ├── __init__.py
│   │   ├── object_tracker.py
│   │   ├── tsdf_fusion.py
│   │   └── persistence.py
│   ├── scenegraph/              # Scene graph construction
│   │   ├── __init__.py
│   │   ├── object_node.py
│   │   ├── spatial_relations.py
│   │   └── hierarchical_graph.py
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── rerun_utils.py
│   │   ├── geometry.py
│   │   └── io_utils.py
│   └── visualization/           # Visualization
│       ├── __init__.py
│       └── rerun_viz.py
├── demos/                       # Demo scripts for each phase
│   ├── demo_1_1_basic_setup.py
│   ├── demo_1_2_data_pipeline.py
│   ├── demo_2_1_sam_segmentation.py
│   └── ...
├── tests/                       # Unit tests
├── .gitignore
├── requirements.txt
├── setup.py
├── README.md
└── PLAN.md
```

---

## Development Phases

### Phase 0: Virtual Environment Setup ✓ (First step)

#### Setup 0: Create Python Virtual Environment
**Goal:** Set up isolated Python environment for the project

**Implementation:**
```bash
# Create virtual environment
cd /home/sravani/E-Lab/Spring2026/repos/MemSpace
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**Create `.gitignore`:**
```
# Virtual environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Hydra outputs
outputs/
multirun/

# Data and models
data/
*.pth
*.pt
*.ckpt
*.pkl
*.pkl.gz

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Rerun recordings
*.rrd

# OS
.DS_Store
Thumbs.db
```

**Deliverable:**
- Working virtual environment
- `.gitignore` file
- **Commit:** "Phase 0: Set up Python virtual environment"

---

### Phase 1: Foundation Setup

#### Demo 1.1: Basic Environment & Dependencies
**Goal:** Install dependencies and verify installation with Hydra + Rerun

**Implementation:**

**Create `requirements.txt`:**
```txt
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.7.0
Pillow>=9.5.0

# Deep learning models
open-clip-torch>=2.20.0
segment-anything>=1.0
mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git

# Configuration management
hydra-core>=1.3.0
omegaconf>=2.3.0

# Visualization
rerun-sdk>=0.17.0

# 3D processing
open3d>=0.17.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
scipy>=1.10.0
scikit-learn>=1.2.0
faiss-cpu>=1.7.4

# Optional: for record3d support
pyliblzfse
```

**Create `setup.py`:**
```python
from setuptools import setup, find_packages

setup(
    name="memspace",
    version="0.1.0",
    description="Hierarchical Scene Graphs for Persistent Robotic Memory",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # See requirements.txt
    ],
)
```

**Create base Hydra configs:**
- `memspace/configs/base.yaml`:
```yaml
defaults:
  - base_paths
  - _self_

# Logging
use_rerun: true
rerun_save_path: null  # null means don't save, or specify path

# General settings
device: cuda  # cuda or cpu
seed: 42
debug: false
```

- `memspace/configs/base_paths.yaml`:
```yaml
# Main paths
repo_root: /home/sravani/E-Lab/Spring2026/repos/MemSpace
data_root: /home/sravani/E-Lab/Spring2026/my_local_data

# Output paths
output_dir: ${repo_root}/outputs
```

**Create `demos/demo_1_1_basic_setup.py`:**
```python
import hydra
from omegaconf import DictConfig
import torch
import rerun as rr

@hydra.main(config_path="../memspace/configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print("="*50)
    print("MemSpace Demo 1.1: Basic Setup")
    print("="*50)

    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Initialize Rerun
    if cfg.use_rerun:
        rr.init("memspace_demo_1_1", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
        rr.log("world/status", rr.TextDocument("# MemSpace Initialized ✓\n\nEnvironment setup successful!"))
        print("✓ Rerun visualization initialized")

    print("\n✓ All systems operational!")
    print("="*50)

if __name__ == "__main__":
    main()
```

**Installation steps:**
```bash
# With venv activated
pip install -r requirements.txt
pip install -e .
```

**Deliverable:**
- Working Python environment with all dependencies
- Rerun window shows "MemSpace initialized"
- **Commit:** "Phase 1.1: Install dependencies and verify Hydra + Rerun"

---

#### Demo 1.2: Data Pipeline
**Goal:** Load and visualize RGB-D data using modular dataset classes

**Implementation:**
- Create `memspace/dataset/base_dataset.py`:
  - Abstract base class for RGB-D datasets
  - Methods: `__len__`, `__getitem__`, `get_intrinsics`, `get_pose`

- Create `memspace/dataset/replica_dataset.py`:
  - Implements ReplicaDataset
  - Loads RGB, depth, camera poses from Replica format

- Create `memspace/utils/rerun_utils.py`:
  - Wrapper functions for logging to Rerun
  - `log_rgb_image()`, `log_depth_image()`, `log_camera_pose()`
  - `log_point_cloud()`

- Create `memspace/configs/dataset/replica.yaml`:
```yaml
dataset_type: replica
scene_id: room0
dataset_path: ${base_paths.data_root}/Replica/${scene_id}

# Camera intrinsics (Replica default)
width: 1200
height: 680
fx: 600.0
fy: 600.0
cx: 599.5
cy: 339.5

# Frame settings
stride: 10  # Process every Nth frame
max_frames: 100  # null for all frames
```

- Create `demos/demo_1_2_data_pipeline.py`:
  - Load single frame from Replica dataset
  - Convert depth to point cloud
  - Visualize in Rerun:
    - RGB image
    - Depth image
    - 3D point cloud
    - Camera pose

**Deliverable:**
- Rerun visualization showing RGB, depth, and point cloud for single frame
- **Commit:** "Phase 1.2: Add RGB-D data loader and Rerun visualization"

---

### Phase 2: Object Detection & Segmentation (ConceptGraphs Core)

#### Demo 2.1: SAM Integration
**Goal:** Integrate SAM for class-agnostic segmentation on single frames

**Implementation:**
- Create `memspace/models/sam_wrapper.py`:
  - SAMWrapper class
  - Load MobileSAM or SAM model
  - Method: `segment_image(rgb, boxes=None)` → returns masks

- Create `memspace/configs/model/sam.yaml`:
```yaml
model_type: mobile_sam  # mobile_sam, vit_h, vit_l, vit_b
checkpoint_path: null  # auto-download if null
device: ${device}

# Segmentation parameters
points_per_side: 32
pred_iou_thresh: 0.88
stability_score_thresh: 0.95
crop_n_layers: 0
crop_n_points_downscale_factor: 1
min_mask_region_area: 100
```

- Create `demos/demo_2_1_sam_segmentation.py`:
  - Load frame from dataset
  - Run SAM segmentation
  - Visualize in Rerun:
    - Original RGB
    - Segmentation masks overlaid
    - Individual mask instances with colors

**Deliverable:**
- Rerun visualization showing SAM masks on single frame
- **Commit:** "Phase 2.1: Add SAM segmentation on single frames"

---

#### Demo 2.2: CLIP Embeddings & Text Queries
**Goal:** Extract CLIP embeddings for segments and enable text-based queries

**Implementation:**
- Create `memspace/models/clip_wrapper.py`:
  - CLIPWrapper class
  - Load CLIP model (e.g., ViT-B/32)
  - Methods:
    - `encode_image(crops)` → image embeddings
    - `encode_text(text)` → text embeddings
    - `compute_similarity(img_emb, text_emb)` → cosine similarity

- Create `memspace/configs/model/clip.yaml`:
```yaml
model_name: ViT-B-32
pretrained: openai
device: ${device}
batch_size: 32
```

- Create `memspace/scenegraph/object_node.py`:
  - ObjectNode dataclass:
    - `instance_id`: unique ID
    - `centroid`: 3D position
    - `bbox`: 3D bounding box
    - `embedding`: CLIP feature vector
    - `mask_2d`: 2D mask (if available)
    - `points_3d`: point cloud
    - `color_rgb`: mean color

- Create `demos/demo_2_2_clip_embeddings.py`:
  - Segment frame with SAM
  - Crop each segment
  - Extract CLIP embeddings
  - Create ObjectNode for each detection
  - Text query interface:
    - User inputs query (e.g., "chair")
    - Compute similarity with all objects
    - Highlight top-k matches in Rerun

**Deliverable:**
- Rerun visualization with text query highlighting matching objects
- **Commit:** "Phase 2.2: Add CLIP embeddings and text queries"

---

#### Demo 2.3: Multi-Frame Object Tracking
**Goal:** Track objects across multiple frames using spatial + embedding similarity

**Implementation:**
- Create `memspace/slam/object_tracker.py`:
  - ObjectTracker class
  - Data association algorithm:
    - Compute spatial distance between detections and existing objects
    - Compute embedding cosine similarity
    - Hybrid cost: `C = w1 * spatial_dist + w2 * (1 - cos_sim)`
    - Hungarian matching or greedy assignment
  - Methods:
    - `update(new_detections, camera_pose)` → associate or add new
    - `get_objects()` → return all tracked objects

- Create `memspace/utils/geometry.py`:
  - Transform point clouds with camera pose
  - Compute 3D centroids, bounding boxes
  - Point cloud merging/fusion

- Create `demos/demo_2_3_multi_frame_tracking.py`:
  - Load 10-20 frames from dataset
  - For each frame:
    - Segment with SAM
    - Extract CLIP embeddings
    - Create detections
    - Update ObjectTracker
  - Visualize in Rerun:
    - Show all tracked objects with consistent IDs
    - Different colors per instance
    - Trajectory paths for centroids

**Deliverable:**
- Rerun visualization showing consistent object tracking across frames
- **Commit:** "Phase 2.3: Add multi-frame object tracking"

---

### Phase 3: 3D Reconstruction & Scene Building

#### Demo 3.1: TSDF Fusion and 3D Reconstruction
**Goal:** Build dense 3D reconstruction using TSDF fusion

**Implementation:**
- Create `memspace/slam/tsdf_fusion.py`:
  - TSDFFusion class using Open3D's ScalableTSDFVolume
  - Voxel grid representation (8.0cm voxel size)
  - Progressive integration of depth frames with camera poses
  - Extract mesh and dense point cloud
  - Voxel downsampling for visualization (2cm)

- Create `demos/demo_3_1_tsdf_fusion.py`:
  - Load sequence of frames from Replica dataset
  - Integrate each frame into TSDF volume
  - Extract final mesh and point cloud
  - Visualize in Rerun:
    - Dense 3D reconstruction
    - Camera trajectory
    - Frame-by-frame integration progress

**Deliverable:**
- Rerun showing dense 3D reconstruction of entire scene
- **Commit:** "Phase 3.1: Add TSDF fusion for dense 3D reconstruction"

---

#### Demo 3.2: Point Cloud Accumulation (ConceptGraphs-style)
**Goal:** Accumulate multi-frame point clouds with voxel downsampling for memory efficiency

**Implementation:**
- Create `memspace/slam/point_cloud_accumulator.py`:
  - PointCloudAccumulator class
  - Voxel-based downsampling (2cm voxel size by default)
  - Progressive accumulation across frames
  - Color averaging within voxels
  - Methods:
    - `integrate(color, depth, intrinsics, pose)`: Add new frame
    - `get_point_cloud()`: Return accumulated Open3D point cloud
    - `get_points_and_colors()`: Return numpy arrays
  - Design inspired by ConceptGraphs' approach

- Create `demos/demo_3_2_point_cloud_accumulation.py`:
  - Load multiple frames from dataset
  - Accumulate point clouds progressively
  - Visualize growing point cloud in Rerun
  - Compare with/without downsampling
  - Statistics: point count, memory usage

**Key Features:**
- Memory efficient: ~300K points/frame → ~2-3M total (with downsampling)
- Real-time capable: ~0.05s per frame integration
- Color-averaged voxels for smooth appearance
- Configurable voxel size for quality vs performance trade-off

**Deliverable:**
- Rerun showing accumulated point cloud with statistics
- **Commit:** "Phase 3.2: Add point cloud accumulation with voxel downsampling"

---

#### Demo 3.3: Per-Object 3D Reconstruction
**Goal:** Build separate 3D models for each tracked object

**Implementation:**
- Integrate object tracking (Phase 2.3) with point cloud accumulation (Phase 3.2)
- Create per-object reconstruction pipeline:
  - Maintain separate PointCloudAccumulator for each tracked object
  - Masked depth integration (only object pixels)
  - Automatic object management (create/update accumulators)
  - Minimum points filtering (100 points threshold)

- Create `demos/demo_3_3_object_reconstruction.py`:
  - Run full pipeline: SAM → CLIP → Tracking → 3D Reconstruction
  - For each frame and each detected object:
    1. Get object_id from tracker
    2. Create masked depth (zero out non-object pixels)
    3. Integrate into object's accumulator
  - Visualize in Rerun:
    - Per-object point clouds with consistent colors
    - Object IDs and point counts
    - Only show objects with sufficient points (≥100)

**Key Design Choices:**
- Per-object accumulators for clean segmentation
- Masked depth prevents background contamination
- Consistent object colors (tracking → 3D visualization)
- Quality filtering (minimum points threshold)
- Visualization limits (max 20 objects by default)

**Performance:**
- Full pipeline: ~2.65s/frame (includes SAM, CLIP, tracking, 3D)
- Per-object integration: ~0.01s/object

**Deliverable:**
- Rerun showing individual 3D models for each tracked object
- **Commit:** "Phase 3.3: Add per-object 3D reconstruction"

---

### Phase 4: Semantic Understanding & Scene Graphs

#### Demo 4.1: Object Labeling with Florence-2
**Goal:** Add semantic understanding using Vision Language Model for object captioning

**Implementation:**
- Create `memspace/models/florence_wrapper.py`:
  - FlorenceWrapper class using Microsoft Florence-2 VLM
  - Two model sizes: base (232M params), large (771M params)
  - Caption generation tasks: `<CAPTION>`, `<DETAILED_CAPTION>`, etc.
  - Batch processing support
  - **CRITICAL**: Requires `transformers==4.49.0` (beam search bug in ≥4.50.0)

- Create `memspace/scenegraph/object_captioning.py`:
  - ObjectCaptioner class
  - Multi-view caption aggregation (up to 5 captions per object)
  - Caption consolidation (use longest caption as final label)
  - Caption interval optimization (every N frames)
  - Minimum observations filter (≥2 observations)

- Create `demos/demo_4_1_object_labeling.py`:
  - Run pipeline: SAM → CLIP → Tracking → Florence-2 Captioning
  - Caption objects from different viewpoints
  - Accumulate caption history per object
  - Consolidate to single label
  - Visualize in Rerun:
    - Object masks with captions
    - Caption history per object
    - Multi-view caption evolution

**Key Design Choices:**
- **Florence-2 vs GPT-4V/LLaVA**: Free, local, lightweight (~1s/object)
- **Crop-based captioning**: 20px padding (same as CLIP)
- **Multi-view aggregation**: Robust labels from multiple viewpoints
- **Caption interval**: Every 5 frames (speed vs coverage trade-off)
- **Simple consolidation**: Longest caption (no LLM needed)

**Performance:**
- Florence-2 inference: ~0.8-1.2s per object
- With interval=5: ~5x faster than per-frame captioning
- Memory: ~3GB VRAM for base model, ~6GB for large

**Installation Requirements:**
```bash
pip install transformers==4.49.0  # Critical version!
```

**Deliverable:**
- Rerun showing tracked objects with semantic labels
- **Commit:** "Phase 4.1: Add Florence-2 object labeling with multi-view aggregation"

---

#### Demo 4.2: Semantic 3D Objects
**Goal:** Unified semantic 3D object representations with scene graph export

**Implementation:**
- Create `memspace/scenegraph/semantic_object.py`:
  - SemanticObject class (extends ObjectInstance from Phase 2.3)
  - Unified representation combining:
    - Tracking: persistent IDs, observation counts, state
    - 3D Reconstruction: point cloud accumulator, 3D bounding box
    - Semantics: caption history, final label, confidence

- Create `memspace/scenegraph/scene_graph.py`:
  - SceneGraph class for managing semantic objects
  - JSON export with external arrays (.npy files)
  - Object filtering by confidence
  - Statistics and summary generation

- Create `demos/demo_4_2_semantic_objects.py`:
  - Run complete pipeline: SAM → CLIP → Tracking → 3D → Florence-2
  - Build unified SemanticObject representations
  - Compute 3D bounding boxes from point clouds
  - Export scene graph to JSON with:
    - Object metadata (ID, label, bbox, confidence)
    - External arrays (point clouds, CLIP features)
    - Scene statistics and summary
  - Visualize in Rerun:
    - Semantic 3D objects with labels
    - Bounding boxes
    - Confidence scores
    - Caption histories

**Key Features:**
- **Unified representation**: Tracking + 3D + Semantics in one class
- **Confidence scoring**: Combines tracking and semantic confidence
- **Efficient export**: JSON metadata + .npy arrays
- **Queryable**: Ready for spatial and semantic queries
- **Human-readable**: JSON files easy to inspect

**Scene Graph Export Format:**
```json
{
  "objects": [
    {
      "object_id": 0,
      "label": "wooden dining table",
      "bbox_3d": [x, y, z, w, h, d],
      "confidence": 0.85,
      "num_observations": 15,
      "caption_history": ["table", "wooden table", "wooden dining table"],
      "point_cloud_points_path": "obj_000_points.npy",
      "clip_features_path": "obj_000_clip_features.npy"
    }
  ],
  "statistics": {...}
}
```

**Deliverable:**
- Complete semantic scene graph with 3D objects
- JSON export with external arrays
- **Commit:** "Phase 4.2: Add semantic 3D objects with scene graph export"

---

### Phase 5: Spatial Relationships & Scene Graph Queries (Future Work)

#### Demo 5.1: Spatial Relationships
**Goal:** Compute spatial relationships between semantic objects

**Planned Implementation:**
- Create `memspace/scenegraph/spatial_relations.py`:
  - Compute relationships using 3D bounding boxes:
    - `supports(obj_a, obj_b)`: A is below B, close in XY
    - `contains(obj_a, obj_b)`: B is inside A's bbox
    - `near(obj_a, obj_b)`: distance < threshold
    - `above/below/left/right`: directional relations
  - Return edges with relationship type and confidence

- Update `SceneGraph` class:
  - Add spatial edges: `{(obj_id_a, obj_id_b): relation_type}`
  - Methods: `add_edge()`, `get_edges()`, `query_relations()`

- Visualize in Rerun:
  - Objects as nodes
  - Edges as lines between centroids
  - Edge labels showing relation type

**Status:** Not yet implemented (pending)

---

#### Demo 5.2: Natural Language Scene Queries
**Goal:** Query scene graph using natural language with FAISS indexing

**Planned Implementation:**
- Create `memspace/query/faiss_index.py`:
  - FAISSIndex class for fast similarity search
  - Index CLIP embeddings from all objects
  - Support queries like:
    - "Find all chairs" (semantic search)
    - "Objects on the table" (spatial + semantic)
    - "Red objects" (visual features)

- Create query engine with:
  - Text-to-embedding conversion (CLIP text encoder)
  - Top-k retrieval with confidence scores
  - Spatial filtering based on relationships
  - Query result visualization in Rerun

**Status:** Not yet implemented (pending)

---

### Phase 6: Future Work - Advanced Features

The following phases represent future directions for the project and have not yet been implemented:

#### Potential Future Directions:

**1. End-to-End Pipeline Integration**
- Unified pipeline class combining all components
- Configuration-driven execution via Hydra
- Save/load functionality for scene graphs
- Batch processing capabilities

**2. Real-World Data Support**
- Record3D dataset integration
- Support for other RGB-D formats (ScanNet, TUM RGB-D)
- Robustness to real-world noise and conditions
- Calibration handling for different sensors

**3. Advanced Querying**
- FAISS-based fast similarity search
- Complex spatial-semantic queries
- Temporal queries (object history tracking)
- Natural language query interface

**4. Interactive Visualization**
- Enhanced Rerun blueprints and layouts
- Layer toggling (dense reconstruction, objects, relationships)
- Timeline scrubbing and playback
- Object inspection and details on demand

**5. Dynamic Scene Understanding**
- Temporal tracking for moving objects
- Persistence management with occlusion handling
- Ray-casting for visibility checks
- Object state management (Active/Missing/Inactive)

**6. Multi-Room and Large-Scale Mapping**
- Room segmentation and hierarchical structure
- Global map optimization
- Loop closure detection
- Multi-session mapping

---

## Key Design Principles

1. **Modularity (like ConceptGraphs)**
   - Separate concerns: dataset, models, SLAM, scene graph
   - Each module independently testable
   - Clear interfaces between components

2. **Configuration-Driven (Hydra)**
   - All parameters in YAML configs
   - Easy experimentation with different settings
   - Config composition for different datasets/models

3. **Visualization-First (Rerun)**
   - Log everything to Rerun
   - Interactive debugging
   - Timeline-based inspection
   - Clear entity hierarchy

4. **Incremental & Demo-Driven**
   - Each demo is runnable standalone
   - Clear progression of features
   - Git history shows development path

5. **Virtual Environment Isolation**
   - All dependencies in isolated venv
   - Reproducible setup
   - No system-wide conflicts

6. **Borrow Intelligently**
   - ConceptGraphs: SAM + CLIP + object association + modularity
   - Hydra: Hierarchical DSG structure + persistence + layers
   - Don't reinvent: use proven architectures

---

## Environment Setup Commands

```bash
# Initial setup (Phase 0)
cd /home/sravani/E-Lab/Spring2026/repos/MemSpace
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Phase 1.1 - Install dependencies
pip install -r requirements.txt
pip install -e .

# Activate environment for development
source venv/bin/activate

# Deactivate when done
deactivate
```

---

## Testing Strategy

- Unit tests for each module
- Integration tests for pipelines
- Regression tests on Replica
- Visual tests via Rerun recordings

---

## Performance Goals

- Real-time capable (>5 fps) on GPU
- Scalable to 100+ objects per room
- Sub-second query response
- Memory efficient (streaming large scenes)

---

## Future Extensions (Post-Demo)

- LLM integration for complex reasoning
- Dynamic scene updates
- Multi-room exploration
- Robot navigation integration
- Collaborative mapping (multi-agent)
- Scene editing and manipulation

---

## Progress Tracking

| Phase | Demo | Status | Notes |
|-------|------|--------|-------|
| 0.0   | Virtual Environment Setup | ✅ Complete | Initial setup |
| 1.1   | Basic Setup | ✅ Complete | Hydra + Rerun + Dependencies |
| 1.2   | Data Pipeline | ✅ Complete | RGB-D loading + visualization |
| 2.1   | SAM Segmentation | ✅ Complete | With 2D mask merging |
| 2.2   | CLIP Embeddings | ✅ Complete | ViT-H-14 features |
| 2.3   | Multi-frame Tracking | ✅ Complete | Spatial + embedding matching |
| 3.1   | TSDF Fusion | ✅ Complete | Dense 3D reconstruction |
| 3.2   | Point Cloud Accumulation | ✅ Complete | ConceptGraphs-style voxel downsampling |
| 3.3   | Per-Object 3D | ✅ Complete | Separate models per object |
| 4.1   | Florence-2 Labeling | ✅ Complete | VLM-based object captioning |
| 4.2   | Semantic Objects | ✅ Complete | Unified 3D semantic scene graph |
| 5.1   | Spatial Relationships | ⏳ Future | Scene graph edges |
| 5.2   | Natural Language Queries | ⏳ Future | FAISS + query engine |
| 6.x   | Advanced Features | ⏳ Future | See Phase 6 for details |

---

## Evolution from Original Plan

**Last Updated:** 2026-01-19

The implementation diverged from the original plan starting at Phase 3, focusing on building a solid foundation of semantic 3D object representations rather than the full hierarchical scene graph structure. This was a deliberate choice to:

1. **Focus on Core Functionality First**: Get semantic 3D objects working end-to-end before adding spatial relationships
2. **Follow ConceptGraphs Approach**: Adopt proven point cloud accumulation methods from ConceptGraphs
3. **Prioritize Semantic Understanding**: Add Florence-2 VLM for object labeling earlier in the pipeline
4. **Enable Practical Applications**: Build export functionality for downstream tasks (robotics, AR/VR)

**Key Deviations:**

- **Phase 3**: Changed from hierarchical layers (L0/L1/L2/L3) to progressive 3D reconstruction (TSDF → Accumulated Point Clouds → Per-Object Models)
- **Phase 4**: Changed from temporal tracking/persistence to semantic labeling (Florence-2) and unified object representations
- **Phase 5-6**: Moved spatial relationships, queries, and advanced features to future work

**Rationale:**

The original plan envisioned a Hydra-inspired hierarchical Dynamic Scene Graph (DSG) with explicit layers for geometry, objects, spatial relationships, and rooms. However, during implementation, we recognized that:

1. Building robust semantic 3D object representations is foundational and more immediately useful
2. Spatial relationships can be computed on-demand from 3D bounding boxes when needed
3. Room segmentation is most valuable for large multi-room environments (future work)
4. The current implementation provides a solid foundation for future hierarchical graph extensions

**Result:**

The current implementation successfully delivers:
- Class-agnostic object detection and tracking
- Dense and per-object 3D reconstruction
- Semantic object labeling with VLM
- Exportable scene graphs with 3D geometry and semantics
- Foundation for spatial reasoning and natural language queries

---

## Notes & Decisions

- **Why virtual environment?** Isolation, reproducibility, no system conflicts
- **Why Rerun over Open3D?** Better timeline support, easier logging, modern UI
- **Why MobileSAM?** Faster inference, good enough for prototyping
- **Why Florence-2 over GPT-4V?** Free, local, lightweight, sufficient quality for object labeling
- **Why ConceptGraphs-style point cloud accumulation?** Proven approach, memory efficient, real-time capable
- **Why deviate from original hierarchical plan?** Focus on semantic 3D objects first, spatial relationships can be added later
- **Dataset priority:** Replica first (easier), then Record3D (real-world validation)

---

**Current Status (as of 2026-01-19):**

✅ **Completed**: Phases 0, 1, 2, 3, and 4 (through Demo 4.2)
- Full pipeline: RGB-D data → SAM → CLIP → Tracking → 3D Reconstruction → Florence-2 Labeling
- Semantic scene graphs with JSON export
- Working demos for all implemented phases

⏳ **Next Steps:**
1. **Demo 5.1**: Implement spatial relationships between objects
   - Use 3D bounding boxes to compute supports/contains/near relationships
   - Add spatial edges to scene graph
   - Visualize relationship graph in Rerun

2. **Demo 5.2**: Add natural language query interface
   - FAISS indexing for fast similarity search
   - Query engine for spatial-semantic queries
   - Interactive query visualization

3. **Future Work**: End-to-end pipeline integration, real-world data, advanced visualization
