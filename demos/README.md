# MemSpace Demos - Design Choices & Documentation

This document captures the design decisions, implementation details, and rationale behind each demo in the MemSpace project.

---

## Table of Contents
- [Demo 1.1: Basic Setup](#demo-11-basic-setup)
- [Demo 1.2: RGB-D Data Pipeline](#demo-12-rgb-d-data-pipeline)
- [Demo 2.1: SAM Segmentation](#demo-21-sam-segmentation)
- [Demo 2.2: CLIP Embeddings](#demo-22-clip-embeddings)
- [Demo 2.3: Multi-frame Object Tracking](#demo-23-multi-frame-object-tracking)
- [Demo 3.1: TSDF Fusion and 3D Reconstruction](#demo-31-tsdf-fusion-and-3d-reconstruction)
- [Demo 3.2: Point Cloud Accumulation](#demo-32-point-cloud-accumulation-conceptgraphs-style)
- [Demo 3.3: Per-Object 3D Reconstruction](#demo-33-per-object-3d-reconstruction)
- [Demo 4.1: Object Labeling with Florence-2](#demo-41-object-labeling-with-florence-2)
- [Demo 4.2: Semantic 3D Objects](#demo-42-semantic-3d-objects)

---

## Demo 1.1: Basic Setup

**File**: `demo_1_1_basic_setup.py`

### Purpose
Verify environment setup, Hydra configuration loading, and Rerun initialization.

### Design Choices

#### 1. **Hydra Configuration System**
- **Choice**: Use Hydra for all configuration management
- **Rationale**:
  - Composable configs (can mix base + dataset + model)
  - Command-line overrides without code changes
  - Same approach as ConceptGraphs for consistency
- **Implementation**:
  ```python
  @hydra.main(config_path="../memspace/configs", config_name="base", version_base=None)
  ```

#### 2. **Rerun for Visualization**
- **Choice**: Rerun SDK instead of Open3D
- **Rationale**:
  - Timeline-based playback (critical for sequential data)
  - Entity hierarchy matches scene graph structure
  - Better for multi-modal data (RGB, depth, masks, poses)
  - Real-time interaction during processing
- **Trade-offs**:
  - Learning curve for new API
  - Less mature than Open3D

#### 3. **Virtual Environment**
- **Choice**: Isolated venv for all dependencies
- **Rationale**:
  - Prevent conflicts with system Python
  - Reproducible across machines
  - Easy to share via requirements.txt

### Key Features
- PyTorch/CUDA availability check
- Config validation
- Rerun viewer spawning
- System information display

### Known Issues
- None

---

## Demo 1.2: RGB-D Data Pipeline

**File**: `demo_1_2_data_pipeline.py`

### Purpose
Load RGB-D data from Replica dataset, convert depth to 3D point clouds, visualize in Rerun.

### Design Choices

#### 1. **Abstract Dataset Base Class**
- **Choice**: `BaseRGBDDataset` abstract class in `memspace/dataset/base_dataset.py`
- **Rationale**:
  - Easy to add new datasets (ScanNet, TUM RGB-D, etc.)
  - Standardized interface for all datasets
  - Shared preprocessing logic (color normalization, depth scaling)
- **Implementation**:
  ```python
  class BaseRGBDDataset(ABC):
      @abstractmethod
      def _load_filepaths(self): ...
      @abstractmethod
      def _load_poses(self): ...
  ```

#### 2. **Replica Dataset Format**
- **Choice**: Use Nice-SLAM's Replica format (not original Replica)
- **Rationale**:
  - Pre-rendered RGB-D trajectories
  - Poses already in camera-to-world format
  - Same format as ConceptGraphs for compatibility
- **Structure**:
  ```
  room0/
  ├── results/
  │   ├── frame000000.jpg  # RGB (1200x680)
  │   ├── depth000000.png  # Depth (uint16, scale 6553.5)
  └── traj.txt             # Poses (16 floats per line → 4x4 matrix)
  ```

#### 3. **Depth to Point Cloud Conversion**
- **Choice**: Manual backprojection using pinhole camera model
- **Rationale**:
  - Full control over coordinate systems
  - Consistent with SLAM literature
  - Easy to debug and validate
- **Formula**:
  ```python
  Z = depth / depth_scale
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  points_world = pose @ [X, Y, Z]
  ```

#### 4. **Rerun Utilities Wrapper**
- **Choice**: Wrapper functions in `memspace/utils/rerun_utils.py`
- **Rationale**:
  - Adapted from ConceptGraphs' `optional_rerun_wrapper.py`
  - Encapsulates Rerun API changes (API evolved during development)
  - Type conversions (torch.Tensor → numpy → Rerun types)
  - Consistent entity naming convention
- **API Changes Handled**:
  - Removed `timeless` parameter
  - Removed `set_time_sequence()`
  - Changed quaternion to `rr.Quaternion(xyzw=...)`

#### 5. **Camera Pose Convention**
- **Choice**: Camera-to-world (C2W) poses
- **Rationale**:
  - Replica provides C2W in traj.txt
  - Points transform as: `p_world = C2W @ p_camera`
  - Visualization in world frame is more intuitive

#### 6. **Point Cloud Storage**
- **Choice**: Per-frame point clouds (not fused)
- **Rationale**:
  - Demo focuses on data loading, not SLAM
  - Fusion comes later (Phase 3.1)
  - Each frame ~307K points is manageable

### Key Features
- Stride-based frame selection (every Nth frame)
- Automatic image resizing
- Color normalization to [0, 1]
- Depth scaling to meters
- Camera trajectory visualization

### Performance
- Data loading: ~0.1 sec/frame
- Point cloud generation: ~0.05 sec/frame
- Point cloud size: ~307K points/frame (640x480)

### Known Issues
- None

---

## Demo 2.1: SAM Segmentation

**File**: `demo_2_1_sam_segmentation.py`

### Purpose
Class-agnostic instance segmentation using Segment Anything Model (SAM) with mask merging to reduce over-segmentation.

### Design Choices

#### 1. **SAM Implementation: Ultralytics vs Original**
- **Choice**: Ultralytics SAM wrapper
- **Rationale**:
  - Same as ConceptGraphs for consistency
  - Simpler API (`model.predict()` vs manual predictor setup)
  - Auto model downloading
  - Built-in batching support
- **Models Supported**:
  - `mobile_sam` (default): Fast, 40MB, ~1-2 sec/frame
  - `sam_l`: Large, high quality
  - `sam_b`: Base model
- **Trade-offs**: Less control over internal parameters vs original SAM

#### 2. **Automatic Mask Generation vs Prompted**
- **Choice**: Automatic mask generation (no bounding box prompts)
- **Rationale**:
  - Class-agnostic (no predefined object categories)
  - Simpler pipeline (no need for detector)
  - Same as ConceptGraphs' "class_set=none" mode
- **Comparison with ConceptGraphs**:
  - **ConceptGraphs-Detect**: Uses Grounding-DINO detector → bounding boxes → SAM prompting
  - **ConceptGraphs**: Uses SAM automatic generation (same as us)
  - **MemSpace**: Uses SAM automatic generation + 2D mask merging

#### 3. **Over-Segmentation Problem & Solutions**

##### Problem
SAM segments every texture variation, leading to:
- Tables split into 5+ pieces
- Walls fragmented by lighting changes
- 71 masks → really ~20-30 objects

##### Solution 1: SAM Parameter Tuning
- **Changes Made**:
  ```yaml
  points_per_side: 32 → 16        # Fewer candidate masks
  pred_iou_thresh: 0.88 → 0.92    # Stricter quality
  stability_score_thresh: 0.95 → 0.97  # More stable masks
  min_mask_region_area: 100 → 500  # Filter small fragments
  ```
- **Impact**: Reduces masks by ~20-30%
- **Trade-offs**: May miss some small objects

##### Solution 2: 2D Mask Merging (Recommended) ✅
- **Choice**: Post-process merging using IoU + containment
- **Rationale**:
  - **Not used by ConceptGraphs at 2D stage** (they merge in 3D)
  - Complementary to 3D merging (fixes different problems)
  - Immediate reduction before expensive CLIP embedding
  - Cleaner per-frame visualizations
- **Algorithm** (`memspace/utils/mask_utils.py`):
  ```python
  1. Sort masks by score (keep higher quality first)
  2. For each mask:
     - Find overlapping masks (IoU > threshold)
     - Find contained masks (one is X% inside another)
     - Merge all into current mask
  3. Recompute bounding boxes
  ```
- **Parameters**:
  ```yaml
  merge_iou_threshold: 0.5          # Merge if IoU > 50%
  merge_containment_threshold: 0.85  # Merge if 85% inside
  ```
- **Results**:
  - Frame 0: 71 masks → 26 masks (45 merged, 63% reduction)
  - Frame 1: 58 masks → 25 masks (33 merged, 57% reduction)
  - Total: 129 masks → 51 masks (**60% reduction**)

##### Solution 3: Depth-Based Filtering (Optional)
- **Choice**: Filter masks spanning multiple depth layers
- **Rationale**:
  - Removes masks that segment across foreground/background
  - Depth variance within object should be small
- **Implementation**:
  ```python
  depth_std = mask_depths.std()
  if depth_std > max_depth_variance:
      discard_mask()
  ```
- **Status**: Disabled by default (can enable with `filter_by_depth: true`)

#### 4. **ConceptGraphs Comparison: When to Merge?**

| Approach | When | What | How | Why |
|----------|------|------|-----|-----|
| **MemSpace (Ours)** | 2D segmentation (per-frame) | Merge 2D masks | IoU + containment | Reduce per-frame over-segmentation |
| **ConceptGraphs** | 3D mapping (multi-frame) | Merge 3D objects | 3D bbox IoU + CLIP | Reduce duplicate objects across views |

**Key Insight**: These are **complementary**, not competing approaches!
- **2D merging** (ours): Table fragmented into 5 pieces → merge to 1 mask
- **3D merging** (ConceptGraphs): Same table seen from 3 angles → merge to 1 object

**Recommendation**: Use both!
1. Apply 2D merging per frame (Phase 2.1) ✅
2. Apply 3D merging during tracking (Phase 2.3, to be implemented)

#### 5. **Mask Visualization Strategy**
- **Choice**: Colored overlay with alpha blending
- **Rationale**:
  - Easy to see mask boundaries
  - Can verify segmentation quality
  - Consistent colors (seed=42) for reproducibility
- **Implementation**:
  ```python
  vis = cv2.addWeighted(image, 1.0, colored_mask, alpha=0.4, 0)
  ```

#### 6. **Rerun Logging Structure**
- **Choice**: Hierarchical entity paths
- **Structure**:
  ```
  world/
  ├── camera/
  │   ├── rgb                      # Original images
  │   └── segmentation_overlay     # Colored masks
  ├── masks/
  │   ├── mask_000                 # Individual masks
  │   ├── mask_001
  │   └── ...
  └── stats                        # Markdown statistics
  ```
- **Rationale**:
  - Logical grouping
  - Easy to toggle visibility
  - Matches ConceptGraphs structure

#### 7. **Configuration Interpolation Issues**
- **Problem**: Hydra can't resolve `${base_paths.data_root}` across config groups
- **Solution**: Use absolute paths in configs
  ```yaml
  # Before (doesn't work):
  dataset_path: ${base_paths.data_root}/Replica/${scene_id}

  # After (works):
  dataset_path: /home/sravani/E-Lab/Spring2026/my_local_data/Replica/room0
  ```
- **Trade-off**: Less portable, but more reliable

### Key Features
- Automatic mask generation (no prompts)
- 2D mask merging to reduce over-segmentation
- Optional depth-based filtering
- Mask statistics (count, sizes, scores)
- Top N masks by score
- Colored overlay visualization
- Per-mask visualization

### Performance
- SAM segmentation: ~1-2 sec/frame (MobileSAM on RTX 4090)
- Mask merging: ~0.05 sec/frame
- Before merging: ~65 masks/frame
- After merging: ~25 masks/frame (60% reduction)

### Parameters to Tune

**For fewer masks** (aggressive merging):
```bash
python demos/demo_2_1_sam_segmentation.py \
  segmentation.merge_iou_threshold=0.3 \
  segmentation.merge_containment_threshold=0.7 \
  segmentation.min_mask_area=1000
```

**For more masks** (conservative merging):
```bash
python demos/demo_2_1_sam_segmentation.py \
  segmentation.merge_iou_threshold=0.7 \
  segmentation.merge_containment_threshold=0.95 \
  segmentation.min_mask_area=100
```

**Enable depth filtering**:
```bash
python demos/demo_2_1_sam_segmentation.py \
  segmentation.filter_by_depth=true \
  segmentation.max_depth_variance=0.3
```

### Known Issues
- None

### Future Work (Phase 2.3: Multi-frame Tracking)
When implementing tracking, should add:
1. **Temporal association**: Match masks across frames using:
   - Spatial overlap (project previous masks to current frame using depth + pose)
   - CLIP similarity (semantic consistency)
2. **3D IoU merging**: Use ConceptGraphs' 3D bounding box IoU
   - See `/home/sravani/E-Lab/Spring2026/repos/concept-graphs/conceptgraph/utils/ious.py`
3. **Periodic merging**: Every N frames, consolidate objects
   - See `/home/sravani/E-Lab/Spring2026/repos/concept-graphs/conceptgraph/slam/cfslam_pipeline_batch.py:297-298`

---

## Demo 2.2: CLIP Embeddings

**File**: `demo_2_2_clip_embeddings.py`

### Purpose
Extract dense visual embeddings for each segmented mask using CLIP for zero-shot semantic recognition and similarity matching.

### Design Choices

#### 1. **CLIP Implementation: OpenCLIP**
- **Choice**: OpenCLIP library (not original CLIP)
- **Rationale**:
  - Same as ConceptGraphs for consistency
  - Supports multiple pretrained checkpoints (LAION, OpenAI)
  - Better performance on LAION-5B trained models
  - Active development and maintenance
- **Model**: ViT-H-14 with laion2b_s32b_b79k weights
  - 1024-dimensional embeddings
  - Best performance vs speed trade-off
  - Same as ConceptGraphs default

#### 2. **Feature Extraction Strategy: Crop-Based**
- **Choice**: Extract CLIP features from bounding box crops (not masked regions)
- **Rationale**:
  - Same approach as ConceptGraphs
  - CLIP expects rectangular images (trained on full images)
  - Masking would create artifacts (black backgrounds)
  - Bounding box provides context around object
- **Implementation**:
  ```python
  1. Get bounding box from SAM mask
  2. Add padding (20 pixels) for context
  3. Crop RGB image
  4. Preprocess with CLIP transforms
  5. Encode to 1024D embedding
  6. L2-normalize for cosine similarity
  ```

#### 3. **Padding Around Bounding Boxes**
- **Choice**: 20 pixels padding (same as ConceptGraphs)
- **Rationale**:
  - Provides visual context around object
  - Helps with partially cropped objects
  - Adaptive padding at image borders (don't go outside)
- **Code**:
  ```python
  left_padding = min(padding, x_min)
  right_padding = min(padding, image_width - x_max)
  ```

#### 4. **Batch Processing**
- **Choice**: Process crops in batches (default: 32)
- **Rationale**:
  - Faster than one-by-one encoding
  - Efficient GPU utilization
  - Memory-friendly (batches prevent OOM)
- **Performance**:
  - ~26 masks processed in <1 second on RTX 4090

#### 5. **Embedding Normalization**
- **Choice**: L2-normalize all embeddings
- **Rationale**:
  - Enables cosine similarity via dot product
  - Similarity = normalized_feat1 @ normalized_feat2.T
  - Standard practice for retrieval tasks
  - Same as ConceptGraphs
- **Formula**:
  ```python
  embedding = embedding / embedding.norm(dim=-1, keepdim=True)
  ```

#### 6. **Similarity Computation**
- **Choice**: Cosine similarity for semantic matching
- **Rationale**:
  - Invariant to magnitude (only direction matters)
  - Range [-1, 1], easy to interpret
  - Efficient: just dot product of normalized vectors
- **Use Cases**:
  - Within-frame grouping (find similar objects)
  - Cross-frame matching (track same object)
  - Natural language queries (compare with text embedding)

#### 7. **Embedding Space Visualization**
- **Choice**: PCA for dimensionality reduction (1024D → 2D)
- **Rationale**:
  - Fast and deterministic
  - Good for overview of embedding distribution
  - Can see clustering of similar objects
- **Alternative**: t-SNE (slower, better local structure)
- **Results**: ~26% variance explained with 2 components

#### 8. **ConceptGraphs Comparison**

| Aspect | ConceptGraphs | MemSpace (Ours) |
|--------|---------------|-----------------|
| **CLIP Model** | ViT-H-14 (laion2b_s32b_b79k) | Same ✅ |
| **Feature Extraction** | Crop bounding box + padding | Same ✅ |
| **Padding** | 20 pixels | Same ✅ |
| **Normalization** | L2-normalize | Same ✅ |
| **Batch Processing** | Yes (batched) | Same ✅ |
| **Text Features** | Extract for object classes | Not yet (Phase 5) |

**Key Insight**: Our CLIP implementation is **fully compatible** with ConceptGraphs!

#### 9. **Rerun Visualization Strategy**
- **Choice**: Log crops separately from masks
- **Structure**:
  ```
  world/
  ├── camera/rgb               # Original images
  ├── camera/segmentation_overlay
  ├── masks/mask_XXX           # Binary masks
  ├── crops/crop_XXX           # CLIP input crops
  ├── frame_N/similarities     # Similarity matrices
  └── embeddings/info          # PCA visualization
  ```
- **Rationale**:
  - Easy to verify CLIP input quality
  - Can inspect what CLIP "sees"
  - Similarity matrix shows semantic grouping

### Key Features
- OpenCLIP ViT-H-14 model (1024D embeddings)
- Crop-based feature extraction with padding
- Batch processing for efficiency
- L2-normalized embeddings
- Cosine similarity computation
- Within-frame similarity analysis
- PCA visualization of embedding space

### Performance
- CLIP model loading: ~4 seconds (downloads on first run)
- Feature extraction: ~0.8 sec for 26 masks (RTX 4090)
- Embedding dimension: 1024
- Model size: ~1.7GB (ViT-H-14)

### Parameters to Tune

**Use smaller CLIP model** (faster, lower quality):
```bash
python demos/demo_2_2_clip_embeddings.py \
  model.clip.model_name=ViT-B-32 \
  model.clip.pretrained=openai
# Embedding dim: 512, faster inference
```

**Adjust crop padding**:
```bash
python demos/demo_2_2_clip_embeddings.py \
  clip_features.padding=40
# More context around objects
```

**Change batch size**:
```bash
python demos/demo_2_2_clip_embeddings.py \
  clip_features.batch_size=64
# Larger batches = faster (if GPU memory allows)
```

### Known Issues
- None

### Technical Details

#### CLIP Architecture
- **Input**: 224×224 RGB images
- **Encoder**: Vision Transformer (ViT-H/14)
  - 16 heads, 32 layers
  - Patch size: 14×14
- **Output**: 1024D embedding
- **Training**: Contrastive learning on 2B image-text pairs (LAION)

#### Similarity Threshold Guidelines
- **>0.9**: Nearly identical (same object, different angle)
- **0.7-0.9**: Same category (e.g., two chairs)
- **0.5-0.7**: Related (e.g., table and chair)
- **<0.5**: Unrelated

#### Embedding Properties
- **L2-normalized**: ||embedding|| = 1.0
- **Dimensionality**: 1024 (ViT-H-14)
- **Similarity metric**: Cosine (dot product)
- **Distribution**: Roughly uniform on hypersphere

### Future Work (Phase 2.3: Multi-frame Tracking)
When implementing tracking, CLIP features enable:
1. **Semantic matching**: Compare embeddings across frames
2. **Combined similarity**: spatial_sim * α + clip_sim * (1-α)
3. **Duplicate detection**: Merge objects with sim > threshold
4. **Zero-shot classification**: Compare with text embeddings

---

## Demo 2.3: Multi-frame Object Tracking

**File**: `demo_2_3_object_tracking.py`

### Purpose
Track object instances across multiple frames using spatial and semantic similarity, enabling persistent object IDs and lifecycle management.

### Design Choices

#### 1. **Tracking Architecture: Object-Based**
- **Choice**: Maintain list of `ObjectInstance` objects, match new detections to existing objects
- **Rationale**:
  - Same approach as ConceptGraphs (MapObject + matching)
  - Enables persistent IDs across frames
  - Supports object lifecycle (Active/Missing/Inactive)
  - Natural fit for scene graph construction

#### 2. **Similarity Computation: Spatial + Semantic**
- **Choice**: Combined similarity = `spatial_weight * bbox_IoU + clip_weight * clip_sim`
- **Rationale**:
  - Same as ConceptGraphs' `aggregate_similarities`
  - Spatial: Ensures physical consistency (same location)
  - Semantic: Ensures visual consistency (same appearance)
  - Weighted sum allows tuning for different scenarios
- **Default Weights**:
  - Spatial: 0.3 (30%)
  - CLIP: 0.7 (70%)
  - Rationale: CLIP is more robust to viewpoint changes

#### 3. **Spatial Similarity: 2D Bounding Box IoU**
- **Choice**: Use 2D bbox IoU (not 3D for now)
- **Rationale**:
  - Simpler and faster than 3D IoU
  - No 3D reconstruction yet (Phase 3.1)
  - Still effective for frame-to-frame tracking
- **Formula**:
  ```python
  IoU = intersection_area / union_area
  ```
- **Future**: Will upgrade to 3D IoU in Phase 3

#### 4. **Semantic Similarity: CLIP Cosine Distance**
- **Choice**: Cosine similarity between L2-normalized CLIP embeddings
- **Rationale**:
  - Range [-1, 1], compatible with IoU range [0, 1]
  - Robust to lighting and viewpoint changes
  - Fast computation (dot product)
- **Formula**:
  ```python
  sim = clip_feat1 @ clip_feat2  # Both normalized
  ```

#### 5. **Matching Strategy: Greedy Per-Detection**
- **Choice**: For each detection, find best matching object (highest similarity)
- **Rationale**:
  - Simple and fast
  - Same as ConceptGraphs
  - Works well for moderate occlusion/viewpoint change
- **Algorithm**:
  ```python
  for each detection:
      if max_similarity > threshold:
          match to best object
      else:
          create new object
  ```
- **Alternative**: Hungarian algorithm (more optimal but slower)

#### 6. **Similarity Threshold Tuning**
- **Choice**: `sim_threshold = 0.5` (default)
- **Rationale**:
  - Too high (e.g., 1.0): No matches, creates duplicate objects
  - Too low (e.g., 0.2): False matches, merges different objects
  - 0.5: Good balance for combined spatial + semantic
- **Results**:
  - Threshold 1.0: 0% match rate (all new objects)
  - Threshold 0.5: 87.7% match rate (good tracking)

#### 7. **Object Lifecycle Management**
- **Choice**: Three states - Active, Missing, Inactive
- **Rationale**:
  - **Active**: Currently observed, high confidence
  - **Missing**: Temporarily not seen (occlusion, out of frame)
  - **Inactive**: Permanently lost (too many missing frames)
- **Transition Rules**:
  - Match found → Active
  - No match for 1 frame → Missing
  - No match for ≥10 frames → Inactive
- **Why**: Handles temporary occlusions without creating duplicates

#### 8. **Feature Aggregation: Exponential Moving Average**
- **Choice**: Update CLIP features with EMA: `feat = 0.7*new + 0.3*old`
- **Rationale**:
  - Smooths noisy features across observations
  - Gives more weight to recent observations
  - More robust than simple average
  - Same idea as ConceptGraphs (they accumulate point clouds)
- **Formula**:
  ```python
  clip_features = alpha * new_feature + (1-alpha) * clip_features
  clip_features /= ||clip_features||  # Re-normalize
  ```

#### 9. **Min Observations for Confirmation**
- **Choice**: Require ≥2 observations before object is "confirmed"
- **Rationale**:
  - Filters out false positives (spurious detections)
  - Ensures object is consistently trackable
  - Reduces noise in final scene graph
- **Use**: Can filter to confirmed objects for cleaner output

#### 10. **ConceptGraphs Comparison**

| Aspect | ConceptGraphs | MemSpace (Ours) |
|--------|---------------|-----------------|
| **Similarity Combo** | `(1+bias)*spatial + (1-bias)*visual` | `w_spatial*spatial + w_clip*clip` |
| **Spatial Sim** | 3D bbox IoU | 2D bbox IoU (will upgrade) |
| **Visual Sim** | CLIP cosine | Same ✅ |
| **Matching** | Greedy per-detection | Same ✅ |
| **Feature Aggregation** | Point cloud accumulation | CLIP EMA |
| **Lifecycle** | Not explicit | Active/Missing/Inactive |

**Key Differences**:
- We use 2D IoU (simpler, no 3D yet)
- We have explicit lifecycle states
- We use EMA for CLIP features (smoother)
- ConceptGraphs uses "phys_bias" parameter, we use separate weights (more intuitive)

#### 11. **Visualization Strategy**
- **Choice**: Color masks by object ID, draw bounding boxes with IDs
- **Rationale**:
  - Easy to see which detections belong to same object
  - Persistent colors across frames (based on object ID seed)
  - Bounding boxes show spatial extent
- **Rerun Structure**:
  ```
  world/
  ├── camera/rgb                    # Original images
  ├── camera/tracking_overlay       # Masks colored by object ID
  ├── tracked_objects/obj_XXX       # Individual object masks
  └── tracking_stats                # Per-frame statistics
  ```

### Key Features
- Multi-frame object association
- Spatial (2D IoU) + semantic (CLIP) similarity
- Greedy matching with threshold
- Object lifecycle (Active/Missing/Inactive)
- Exponential moving average for CLIP features
- Persistent object IDs with consistent colors
- Confirmed object filtering

### Performance
- Match rate: **87.7%** (10 frames, 253 detections → 31 objects)
- Processing: ~2 sec/frame (SAM + CLIP + tracking)
- Tracking overhead: <0.05 sec/frame

### Results (10 Frames, Stride=10)
```
Total detections: 253
Successfully matched: 222 (87.7%)
New objects created: 31
Active objects: 17
Confirmed objects: 31
```

### Parameters to Tune

**More lenient matching** (higher match rate):
```bash
python demos/demo_2_3_object_tracking.py \
  tracking.sim_threshold=0.3 \
  tracking.clip_weight=0.8
# More matches, risk of false positives
```

**Stricter matching** (lower match rate):
```bash
python demos/demo_2_3_object_tracking.py \
  tracking.sim_threshold=0.7 \
  tracking.spatial_weight=0.6
# Fewer matches, more new objects
```

**Adjust lifecycle**:
```bash
python demos/demo_2_3_object_tracking.py \
  tracking.max_missing_frames=5 \
  tracking.min_observations=3
# Faster to mark inactive, higher confirmation threshold
```

**More frames for longer tracking**:
```bash
python demos/demo_2_3_object_tracking.py \
  dataset.max_frames=50 \
  dataset.stride=5
# 50 frames with stride 5 = better tracking statistics
```

### Known Issues
- 2D IoU may fail with large viewpoint changes (will be fixed with 3D IoU in Phase 3.1)
- No re-identification (if object goes inactive, new observation creates new ID)
- Greedy matching can create sub-optimal associations (Hungarian would be better)

### Technical Details

#### Similarity Matrix
- Shape: (M detections, N objects)
- Values: Combined similarity scores
- Computation: O(M * N * D) where D = CLIP dimension

#### Object Instance Structure
```python
ObjectInstance:
  - object_id: Unique identifier
  - status: Active/Missing/Inactive
  - clip_features: Aggregated CLIP embedding
  - current_bbox: Most recent bounding box
  - num_observations: Total observations
  - num_missing_frames: Consecutive missing frames
```

#### Matching Threshold Analysis
From experiments on 10 frames:
- **0.3**: ~95% match rate (some false matches)
- **0.5**: 87.7% match rate (good balance) ✅
- **0.7**: ~60% match rate (conservative)
- **1.0**: 0% match rate (too strict)

### Future Work (Phase 3.1: 3D Reconstruction)
When implementing 3D reconstruction:
1. **Upgrade to 3D IoU**: Use 3D bounding boxes from point clouds
   - See ConceptGraphs `/conceptgraph/utils/ious.py:compute_3d_iou()`
2. **Point cloud accumulation**: Merge 3D points from matched objects
3. **TSDF fusion**: Create dense 3D mesh for each object
4. **Re-identification**: Match inactive objects using full 3D geometry

---

## Demo 3.1: TSDF Fusion and 3D Reconstruction

**File**: `demo_3_1_tsdf_reconstruction.py`

### Purpose
Demonstrate volumetric fusion of RGB-D frames to create a dense 3D reconstruction of the entire scene.

### Design Choices

#### 1. **TSDF-based Volumetric Fusion**
- **Choice**: Use TSDF (Truncated Signed Distance Function) instead of point cloud accumulation
- **Rationale**:
  - **Noise reduction**: Averaging multiple observations reduces sensor noise
  - **Hole filling**: TSDF interpolates between measurements, filling gaps
  - **Surface quality**: Marching cubes produces smooth, watertight meshes
  - **Memory efficiency**: Voxel-based representation handles large scenes
  - **Industry standard**: Used in KinectFusion, BundleFusion, etc.
- **Implementation**: Open3D's `ScalableTSDFVolume`
- **Comparison with ConceptGraphs**:
  - CG: Direct point cloud accumulation (faster, simpler)
  - MemSpace: TSDF fusion (higher quality, slower)
  - Trade-off: We prioritize mesh quality for object reconstruction

#### 2. **Voxel Size Selection**
- **Choice**: 1cm voxels (0.01m)
- **Rationale**:
  - **Resolution**: Captures fine details (furniture edges, small objects)
  - **Memory**: ~1GB for 10³ voxels per cubic meter
  - **Depth accuracy**: Replica depth is ~1cm accurate
  - **Scene size**: Room0 is ~10m x 6m x 3m ≈ 180m³ → ~180M voxels
- **Alternative considered**:
  - 2cm voxels: Faster, less memory, but loses detail
  - 0.5cm voxels: More detail, but 8x memory and processing time
- **Tunable**: Can adjust via config for different scenes

#### 3. **SDF Truncation Distance**
- **Choice**: 4cm (4x voxel size)
- **Rationale**:
  - **Standard practice**: 2-6x voxel size recommended
  - **Sensor noise**: Replica depth noise ~2-3cm, so 4cm handles outliers
  - **Surface smoothing**: Wider truncation = smoother surfaces
  - **Memory**: Larger truncation = more voxels updated per frame
- **Implementation**: Each depth measurement updates voxels within ±4cm
- **Effect on results**:
  - Too small (1-2cm): Noisy surfaces, missing data
  - Too large (8cm+): Over-smoothed, blurred edges

#### 4. **Mesh Extraction (Marching Cubes)**
- **Choice**: Extract mesh via marching cubes, then simplify
- **Rationale**:
  - **High-quality initial mesh**: Marching cubes produces 7.5M triangles
  - **Simplification needed**: Too many triangles for visualization/processing
  - **Quadric decimation**: Preserves shape while reducing triangle count
  - **Target**: 100K triangles balances quality and performance
- **Process**:
  1. Extract full mesh from TSDF (7.5M triangles)
  2. Simplify to 100K using quadric error metrics
  3. Recompute normals for smooth shading
- **Alternative considered**: Point cloud only (faster, but no surface)

#### 5. **Color Integration**
- **Choice**: Integrate RGB color into TSDF
- **Rationale**:
  - **Semantic information**: Color helps with object recognition
  - **Visualization**: Textured meshes easier to interpret
  - **Small overhead**: RGB8 adds minimal memory vs. gray
- **Implementation**: Average RGB values across observations
- **Effect**: Mesh has vertex colors from averaged observations

#### 6. **Frame Selection Strategy**
- **Choice**: Stride-based sampling (every 10th frame)
- **Rationale**:
  - **Coverage**: 50 frames at stride 10 covers 500 frame span
  - **Efficiency**: Full 2000 frames takes ~30 minutes
  - **Redundancy**: Adjacent frames are very similar (small motion)
  - **Demo purpose**: Show capability without long wait
- **Full reconstruction**: Can use stride=1 for maximum quality
- **Adaptive sampling**: Could select keyframes based on motion (future work)

#### 7. **Camera Pose Integration**
- **Choice**: Use ground truth poses from Replica dataset
- **Rationale**:
  - **Accuracy**: Perfect alignment for demonstration
  - **Focus**: Validate reconstruction, not SLAM
  - **Future work**: Phase 4 will add SLAM for pose estimation
- **Format**: 4x4 camera-to-world transformation matrices
- **Handling**: Directly integrate into TSDF volume

### Implementation Details

#### TSDF Integration Pipeline
```python
# For each RGB-D frame:
1. Convert color to uint8, depth to float32 (meters)
2. Create Open3D RGBD image
3. Set camera intrinsics (fx, fy, cx, cy)
4. Integrate into TSDF volume with camera pose
5. Update voxels within truncation distance
```

#### Mesh Processing Pipeline
```python
1. Extract mesh from TSDF using marching cubes
2. Simplify from 7.5M triangles → 100K triangles
3. Compute vertex normals for smooth shading
4. Log to Rerun for visualization
```

#### Memory Management
- **Scalable TSDF**: Grows dynamically, only allocates occupied voxels
- **Initial mesh**: ~4M vertices × (3 float pos + 3 float normal + 3 uint8 color) ≈ 100MB
- **Simplified mesh**: ~85K vertices ≈ 2MB
- **Point cloud**: ~3.8M points × 6 floats ≈ 90MB

### Results & Statistics

**Test Run (10 frames, stride 50)**:
- Frames integrated: 10
- Original mesh: 4.0M vertices, 7.5M triangles
- Simplified mesh: 84K vertices, 100K triangles
- Point cloud: 3.8M points
- Scene volume: 324.89 m³
- Processing time: ~30 seconds

**Full Run (50 frames, stride 10)** (estimated):
- Better coverage of the scene
- More complete reconstruction
- Reduced noise through averaging
- Processing time: ~2-3 minutes

### Comparison with ConceptGraphs

| Aspect | ConceptGraphs | MemSpace (Phase 3.1) |
|--------|--------------|----------------------|
| Method | Point cloud accumulation | TSDF volumetric fusion |
| Output | Point cloud | Mesh + point cloud |
| Quality | Fast, noisy | Slow, smooth |
| Holes | Not filled | Interpolated |
| Memory | Lower | Higher (voxel grid) |
| Use case | Online SLAM | Offline reconstruction |

**When to use each**:
- ConceptGraphs approach: Real-time SLAM, limited compute
- MemSpace approach: High-quality offline reconstruction, sufficient memory

### Configuration

`memspace/configs/demo_3_1.yaml`:
```yaml
tsdf:
  voxel_size: 0.01  # 1cm voxels
  sdf_trunc: 0.04   # 4cm truncation
  color_type: RGB8  # RGB color integration

mesh:
  simplify: true
  target_triangles: 100000  # Simplify to 100K triangles

dataset:
  stride: 10  # Every 10th frame
  max_frames: 50  # 50 frames total
```

### Rerun Visualization

**Entities logged**:
- `world/mesh/original`: Full-resolution mesh from TSDF (7.5M triangles)
- `world/mesh/processed`: Simplified mesh (100K triangles)
- `world/point_cloud`: Dense point cloud from TSDF
- `world/camera_trajectory`: Camera poses during reconstruction
- `world/summary`: Statistics and completion summary

**Interaction**:
- Toggle mesh vs. point cloud
- Compare original vs. simplified mesh
- View camera trajectory
- Inspect mesh quality and coverage

### Known Limitations

1. **Processing time**: Full reconstruction takes 2-3 minutes
   - Solution: Could implement GPU-accelerated TSDF (voxblox-gpu)

2. **Memory usage**: Large scenes require significant RAM
   - Solution: Could use sparse voxel hashing or octrees

3. **Static scenes only**: Assumes no moving objects
   - Solution: Phase 3.2 will add per-object reconstruction

4. **Ground truth poses required**: Not fully autonomous
   - Solution: Phase 4 will add SLAM for pose estimation

### Next Steps (Phase 3.3)

**Per-object 3D Reconstruction**:
1. Combine object tracking (Phase 2.3) with 3D reconstruction
2. Create separate point cloud/TSDF for each tracked object
3. Integrate only masked regions (object pixels)
4. Extract object-level meshes
5. Enable object manipulation and querying

---

## Demo 3.2: Point Cloud Accumulation (ConceptGraphs-style)

**File**: `demo_3_2_pointcloud_reconstruction.py`

### Purpose
Demonstrate direct point cloud accumulation as a faster alternative to TSDF fusion, following ConceptGraphs' approach for real-time applications.

### Design Choices

#### 1. **Direct Point Cloud Accumulation vs. TSDF**
- **Choice**: Accumulate point clouds directly without volumetric grid
- **Rationale**:
  - **Speed**: 10x faster than TSDF (0.06s vs 3s per frame)
  - **Memory**: No voxel grid needed, lower memory footprint
  - **Real-time friendly**: Suitable for online SLAM applications
  - **Simplicity**: Fewer parameters, easier to implement
  - **ConceptGraphs compatibility**: Same approach as CG for consistency
- **Implementation**: Transform points to world coords, accumulate, downsample
- **Trade-off**: Speed vs. quality (noisier, holes not filled)

#### 2. **Voxel Size Selection**
- **Choice**: 2cm voxels (vs. TSDF's 1cm)
- **Rationale**:
  - **Speed**: 2x fewer voxels = 2x faster processing
  - **Memory**: Lower memory usage for large scenes
  - **Real-time constraint**: Must keep up with frame rate
  - **Sufficient detail**: 2cm captures major geometry
- **Comparison with TSDF**:
  - TSDF uses 1cm for highest quality offline reconstruction
  - Point cloud uses 2cm for real-time performance
- **Result**: 160K points (10 frames) vs. 4M vertices (TSDF)

#### 3. **Downsampling Strategy**
- **Choice**: Voxel grid filter applied twice (per-frame and global)
- **Rationale**:
  - **Per-frame downsampling**: Reduce data before merging
  - **Global downsampling**: Keep accumulated cloud size manageable
  - **Memory control**: Prevents unbounded growth
  - **Speed**: Faster operations on smaller clouds
- **Implementation**:
  ```python
  # Downsample current frame
  pcd_frame = pcd_frame.voxel_down_sample(voxel_size)
  # Merge with global
  global_pcd += pcd_frame
  # Downsample global
  global_pcd = global_pcd.voxel_down_sample(voxel_size)
  ```
- **Effect**: Keeps point cloud size ~100-200K points

#### 4. **Outlier Removal**
- **Choice**: Statistical outlier removal after accumulation
- **Rationale**:
  - **Depth noise**: RGB-D sensors produce noisy measurements
  - **Flying pixels**: Depth discontinuities cause outliers
  - **Clean reconstruction**: Remove artifacts for better mesh
  - **Standard practice**: Used in Open3D pipelines
- **Parameters**:
  - `nb_neighbors=20`: Check 20 nearest neighbors
  - `std_ratio=2.0`: Remove points >2 std devs from mean
- **Trade-off**: Removes noise but may remove fine details

#### 5. **Optional Mesh Reconstruction**
- **Choice**: Poisson surface reconstruction (optional)
- **Rationale**:
  - **Flexibility**: Can use point cloud directly or extract mesh
  - **Poisson quality**: Produces smooth, watertight surfaces
  - **Slow but offline**: Meshing done after accumulation
  - **Density filtering**: Remove low-density artifacts
- **Alternative**: Ball pivoting (faster, less smooth)
- **Poisson depth**: 9 (good balance of quality and speed)
- **Trade-off**: Meshing takes 50s (vs. 0.6s accumulation)

#### 6. **Transform to World Coordinates**
- **Choice**: Transform each point cloud to world frame immediately
- **Rationale**:
  - **Consistent frame**: All points in same coordinate system
  - **Simpler merging**: Just concatenate point clouds
  - **Standard practice**: ConceptGraphs and most SLAM systems
  - **Enables visualization**: Can view accumulated map in real-time
- **Implementation**: Apply camera pose transform (4x4 matrix)
- **Alternative**: Store in camera frame (requires re-projection)

#### 7. **Performance Optimizations**
- **Choice**: Focus on speed over quality
- **Rationale**:
  - **Real-time target**: Must keep up with sensor frame rate
  - **Online SLAM**: Can't afford slow processing
  - **ConceptGraphs parity**: Match their performance characteristics
- **Optimizations**:
  - Larger voxel size (2cm vs. 1cm)
  - No volumetric grid (saves memory and compute)
  - Downsample early and often
  - Optional meshing (can skip for real-time)

### Implementation Details

#### Point Cloud Accumulation Pipeline
```python
# For each RGB-D frame:
1. Convert color to uint8, depth to float32
2. Create Open3D RGBD image
3. Generate point cloud from RGBD + intrinsics
4. Transform to world coordinates using camera pose
5. Downsample point cloud (voxel filter)
6. Merge with global point cloud
7. Downsample global point cloud
```

#### Mesh Reconstruction Pipeline (Optional)
```python
1. Get accumulated point cloud
2. Remove statistical outliers
3. Estimate normals (required for Poisson)
4. Run Poisson surface reconstruction
5. Remove low-density vertices (artifacts)
6. Simplify to target triangle count
7. Compute vertex normals
```

### Results & Statistics

**Test Run (10 frames, stride 50)**:
- Frames integrated: 10
- Point cloud accumulation: 0.64s (0.064s per frame)
- Point cloud size: 160,254 points
- Mesh extraction: 49.77s (Poisson reconstruction)
- Final mesh: 50,281 vertices, 99,999 triangles
- Scene volume: 55.37 m³
- **Total time**: 52.58s (50s is meshing, can be skipped)

**Performance Comparison with TSDF (Demo 3.1)**:
| Metric | Point Cloud (3.2) | TSDF (3.1) | Speedup |
|--------|------------------|------------|---------|
| Per-frame time | 0.064s | ~3s | **47x faster** |
| Total time (10 frames) | 0.64s | ~30s | **47x faster** |
| Points/Vertices | 160K points | 4M vertices | 25x fewer |
| Memory usage | Low | High | ~10x less |
| Mesh quality | Noisy, holes | Smooth, filled | Quality trade-off |

**Key Insight**: Point cloud is much faster but requires post-processing (meshing) for surfaces.

### Comparison with Demo 3.1 (TSDF)

| Aspect | Demo 3.2 (Point Cloud) | Demo 3.1 (TSDF) |
|--------|----------------------|-----------------|
| **Method** | Direct accumulation | Volumetric fusion |
| **Speed** | 0.064s/frame | 3s/frame |
| **Memory** | ~100MB (160K points) | ~1GB (voxel grid) |
| **Quality** | Noisy, sparse | Smooth, dense |
| **Holes** | Not filled | Interpolated/filled |
| **Real-time** | ✅ Yes | ❌ No |
| **Use case** | Online SLAM | Offline reconstruction |
| **Mesh output** | Optional (Poisson) | Built-in (marching cubes) |
| **ConceptGraphs** | ✅ Same approach | Different |

**When to use each**:
- **Point Cloud (3.2)**: Real-time SLAM, online mapping, limited compute/memory
- **TSDF (3.1)**: Offline reconstruction, high quality needed, sufficient resources

### Configuration

`memspace/configs/demo_3_2.yaml`:
```yaml
pointcloud:
  voxel_size: 0.02  # 2cm voxels (faster than TSDF)
  max_depth: 10.0   # Ignore depth > 10m
  outlier_removal: true
  outlier_nb_neighbors: 20
  outlier_std_ratio: 2.0

mesh:
  enable: true  # Optional meshing
  method: poisson
  poisson_depth: 9
  simplify: true
  target_triangles: 100000

dataset:
  stride: 10
  max_frames: 50
```

### Rerun Visualization

**Entities logged**:
- `world/point_cloud`: Accumulated point cloud with colors
- `world/mesh/reconstructed`: Optional Poisson-reconstructed mesh
- `world/camera_trajectory`: Camera poses during accumulation
- `world/summary`: Performance statistics and comparison

**Interaction**:
- Toggle between point cloud and mesh views
- Compare speed vs. quality trade-offs
- View camera trajectory

### Known Limitations

1. **Point cloud noise**: Not as smooth as TSDF
   - Solution: Apply bilateral filter or increase downsampling

2. **Holes not filled**: Gaps in sparse regions
   - Solution: Use TSDF for final reconstruction, point cloud for tracking

3. **Mesh reconstruction slow**: Poisson takes 50s
   - Solution: Skip meshing for real-time, or use faster methods

4. **No implicit surface**: Point cloud is not a surface
   - Solution: Use meshing when surface is needed

### Design Rationale: Why Both TSDF and Point Cloud?

**Philosophy**: Provide both approaches to match different use cases

1. **Phase 3.1 (TSDF)**: Best for offline, high-quality reconstruction
   - Use when: Quality matters more than speed
   - Example: Creating final 3D models for visualization

2. **Phase 3.2 (Point Cloud)**: Best for online, real-time applications
   - Use when: Speed matters more than quality
   - Example: Real-time SLAM, object tracking integration

3. **Next Phase 3.3**: Will use point cloud approach
   - Per-object reconstruction needs real-time performance
   - Integrates with object tracking (Phase 2.3)
   - ConceptGraphs uses point clouds for this

### Next Steps (Phase 4)

**Semantic Scene Graphs and Querying**:
1. Build scene graph from tracked objects
2. Add spatial relationships (on, in, near)
3. Natural language queries using CLIP
4. Interactive object manipulation

---

## Demo 3.3: Per-Object 3D Reconstruction

**File**: `demo_3_3_object_reconstruction.py`

### Purpose
Integrate object tracking (Phase 2.3) with 3D reconstruction (Phase 3.2) to build separate 3D models for each tracked object.

### Design Choices

#### 1. **Per-Object Point Cloud Accumulation**
- **Choice**: Maintain separate PointCloudAccumulator for each tracked object
- **Rationale**:
  - **Object isolation**: Each object gets its own 3D model
  - **Semantic understanding**: Know which points belong to which object
  - **Memory efficiency**: Only store relevant points per object
  - **Scalability**: Can process objects independently
  - **Foundation for scene graphs**: Ready for semantic relationships
- **Implementation**: Dictionary mapping object_id → PointCloudAccumulator
- **Alternative**: Single global point cloud with per-point labels (less efficient)

#### 2. **Masked Depth Integration**
- **Choice**: Zero out depth for non-object pixels before integration
- **Rationale**:
  - **Clean segmentation**: Only object pixels contribute to its 3D model
  - **No background contamination**: Objects don't include surrounding geometry
  - **Accurate boundaries**: Object extents match visual appearance
  - **Simple implementation**: Just mask depth before passing to accumulator
- **Implementation**:
  ```python
  masked_depth = depth.copy()
  masked_depth[~mask] = 0.0  # Zero depth for non-object pixels
  accumulator.integrate(color, masked_depth, intrinsics, pose)
  ```
- **Effect**: Each object's point cloud contains only its own geometry

#### 3. **Automatic Object Management**
- **Choice**: Automatically create/update accumulators as objects are tracked
- **Rationale**:
  - **Seamless integration**: No manual object registration needed
  - **Dynamic handling**: Objects appear/disappear automatically
  - **Consistent IDs**: Tracking IDs directly map to reconstruction
  - **Memory management**: Can prune objects with too few points
- **Implementation**: Check if object_id exists, create if new, update if existing
- **Lifecycle**: Objects follow tracker's Active/Missing/Inactive states

#### 4. **Minimum Points Filtering**
- **Choice**: Require minimum 100 points per object to be valid
- **Rationale**:
  - **Quality control**: Filter out noisy/fragmentary detections
  - **Visualization**: Don't clutter view with tiny objects
  - **Computational efficiency**: Skip processing invalid objects
  - **Reasonable threshold**: 100 points ≈ small object at 2cm resolution
- **Tunable**: Can adjust threshold based on application
- **Effect**: Only show well-reconstructed objects

#### 5. **Complete Pipeline Integration**
- **Choice**: Run full pipeline (SAM → CLIP → Tracking → 3D) per frame
- **Rationale**:
  - **End-to-end**: Demonstrate complete system working together
  - **Real-world scenario**: This is how actual applications work
  - **Performance measurement**: Understand total computational cost
  - **Validation**: Verify all components work together
- **Pipeline**:
  1. SAM generates masks
  2. CLIP extracts features
  3. Tracker associates to objects
  4. Reconstructor builds 3D models
- **Per-frame cost**: ~2.65s (includes all steps)

#### 6. **Consistent Object Colors**
- **Choice**: Use tracker's object colors for reconstruction visualization
- **Rationale**:
  - **Consistency**: Same object has same color in 2D and 3D views
  - **Tracking validation**: Easy to see if tracking worked correctly
  - **User experience**: Intuitive color-coded visualization
  - **Implementation**: Use object.get_color() for point cloud colors
- **Effect**: Can visually match tracked 2D masks to 3D point clouds

#### 7. **Visualization Limits**
- **Choice**: Show maximum 20 objects by default
- **Rationale**:
  - **Performance**: Too many objects slow down Rerun viewer
  - **Usability**: Easier to inspect subset of objects
  - **Configurable**: Can increase for scenes with fewer objects
  - **Sorted by quality**: Show objects with most points first
- **Implementation**: Take top N objects by point count
- **Alternative**: Show all objects (may be slow for large scenes)

### Implementation Details

#### Per-Object Integration Pipeline
```python
# For each frame and each detected object:
1. Get object_id from tracker
2. Create or retrieve PointCloudAccumulator for this object
3. Create masked depth (zero out non-object pixels)
4. Integrate masked RGB-D into object's accumulator
5. Object's point cloud grows over time as it's observed
```

#### Object Lifecycle
```python
# Object appears (first detection):
- Tracker creates new object with ID
- Reconstructor creates new PointCloudAccumulator
- Start accumulating points

# Object tracked (subsequent frames):
- Tracker matches detection to existing object
- Reconstructor integrates more points into same accumulator
- Point cloud grows and becomes more complete

# Object disappears (tracking lost):
- Tracker marks as Missing/Inactive
- Reconstructor keeps existing points
- Can still visualize accumulated 3D model
```

### Results & Statistics

**Test Run (10 frames, stride 50)**:
- Frames processed: 10
- Total time: 26.47s (2.65s per frame)
- Objects tracked: 37
- Objects reconstructed: 37 (100% success)
- Valid objects: 37 (all have ≥100 points)

**Pipeline Breakdown (per frame)**:
- SAM segmentation: ~0.5s
- CLIP feature extraction: ~1.0s
- Object tracking: ~0.1s
- 3D reconstruction: ~1.0s
- **Total**: ~2.65s per frame

**Object Statistics** (example):
- Object 0: 8,547 points, 10 frames
- Object 5: 12,321 points, 9 frames
- Object 12: 15,892 points, 8 frames
- Average: ~10,000 points per object

### Comparison with Previous Phases

| Phase | Scope | Speed | Output | Use Case |
|-------|-------|-------|--------|----------|
| 3.1 (TSDF) | Scene-level | 3s/frame | 1 dense mesh | Offline reconstruction |
| 3.2 (Point Cloud) | Scene-level | 0.06s/frame | 1 point cloud | Real-time mapping |
| **3.3 (Per-Object)** | **Object-level** | **2.65s/frame** | **N point clouds** | **Object understanding** |

**Key Difference**: Phase 3.3 provides object-level decomposition, enabling:
- Individual object queries
- Semantic understanding
- Object manipulation
- Scene graph construction

### Configuration

`memspace/configs/demo_3_3.yaml`:
```yaml
# Object tracking (from Phase 2.3)
tracking:
  sim_threshold: 0.5
  spatial_weight: 0.3
  clip_weight: 0.7

# Per-object reconstruction
object_reconstruction:
  voxel_size: 0.02  # 2cm per object
  min_points: 100   # Minimum valid object
  outlier_removal: true

# Visualization
visualization:
  show_individual_objects: true
  max_objects_to_show: 20
```

### Rerun Visualization

**Entities logged**:
- `world/objects/obj_XXX/pointcloud`: Per-object 3D point clouds
- `world/objects/obj_XXX/info`: Object statistics (points, frames, tracking)
- `world/camera_trajectory`: Camera poses during capture
- `world/summary`: Complete pipeline statistics

**Interaction**:
- Toggle individual objects on/off
- Inspect per-object statistics
- Compare tracking quality vs. 3D quality
- Validate object consistency

### Known Limitations

1. **Processing time**: 2.65s per frame (not quite real-time)
   - Solution: Could optimize SAM/CLIP or use smaller models

2. **Static scene assumption**: Objects expected to be stationary
   - Solution: Could add per-object coordinate frames for moving objects

3. **Depth masking artifacts**: Hard mask boundaries may cause edge noise
   - Solution: Could use soft masks or morphological operations

4. **Memory scales with objects**: Each object needs its own accumulator
   - Solution: Could prune inactive objects or merge similar ones

### Design Rationale: Complete Pipeline

**Philosophy**: Demonstrate end-to-end system working together

1. **Integration over isolation**: Show how all pieces fit together
   - SAM provides object candidates
   - CLIP provides semantic features
   - Tracker maintains identities
   - Reconstructor builds 3D models

2. **Validation**: Ensure components are compatible
   - Masks → Point clouds (depth masking works)
   - Tracking → Reconstruction (IDs propagate correctly)
   - 2D → 3D (transformations are correct)

3. **Performance measurement**: Understand real-world costs
   - Not just individual components
   - Actual end-to-end latency matters
   - Identifies bottlenecks for optimization

4. **Foundation for applications**:
   - Object-level scene understanding
   - Interactive 3D manipulation
   - Semantic scene graphs
   - Natural language queries

### Next Steps (Phase 4)

**Semantic Scene Graphs**:
1. Build graph from reconstructed objects
2. Add spatial relationships (on, in, near)
3. Integrate with CLIP for natural language
4. Enable queries like "red chair near the table"
5. Object manipulation and interaction

---

## Demo 4.1: Object Labeling with Florence-2 or LLaVA

**File**: `demo_4_1_object_labeling.py`

### Purpose
Add semantic understanding to tracked objects using Vision Language Models (VLMs) for zero-shot object captioning and labeling. **Supports both Florence-2 and LLaVA** for flexible VLM selection.

### Design Choices

#### 1. **VLM Selection: Florence-2 (Default) or LLaVA**
- **Choice**: Support both Microsoft Florence-2 (default) and LLaVA-1.5 for flexible captioning
- **Florence-2 (Default)**:
  - **Free and open-source**: No API costs
  - **Local execution**: Privacy and offline capability
  - **Two model sizes**: base (232M params), large (771M params)
  - **Fast inference**: ~0.8-1.2s per object on RTX 4090
  - **Rich tasks**: `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OD>`, etc.
  - **HuggingFace integration**: Easy to use with transformers library
  - **Memory efficient**: ~3GB GPU memory
- **LLaVA-1.5 (Alternative)**:
  - **Visual instruction tuning**: Better semantic understanding with natural language
  - **More detailed captions**: Richer descriptions (e.g., "The central object in this image is a wooden dresser")
  - **Model size**: 7B parameters (llava-v1.5-7b, ~13GB download)
  - **Memory options**: FP16 (~19GB), 8-bit (~10GB), 4-bit (~5GB)
  - **Inference speed**: ~2-3s per object
  - **Same as ConceptGraphs**: Uses LLaVA directly from standalone repo
- **VLM Switching**: Single config parameter `vlm_type=florence` or `vlm_type=llava`
- **Trade-off**: Florence-2 is faster and lighter; LLaVA provides richer semantic understanding

#### 2. **CRITICAL: Transformers Version Compatibility**
- **Choice**: Requires `transformers==4.49.0` (not newer versions)
- **Rationale**:
  - **Bug in newer versions**: transformers ≥4.50.0 has beam search bug with Florence-2
  - **Error**: `AttributeError: 'NoneType' object has no attribute 'shape'` in prepare_inputs_for_generation
  - **Solution**: Downgrade to 4.49.0 fixes issue completely
- **Installation**:
  ```bash
  pip install transformers==4.49.0
  ```
- **Status**: ✅ Fully working with 4.49.0
- **Future**: Will upgrade when Florence-2/transformers bug is fixed

#### 3. **Caption Task Selection**
- **Choice**: Use `<CAPTION>` task (not `<DETAILED_CAPTION>`)
- **Rationale**:
  - **Concise**: Short object labels (e.g., "wooden table", "office chair")
  - **Fast**: Less generation tokens = faster inference
  - **Sufficient**: Object identification doesn't need lengthy descriptions
  - **ConceptGraphs-like**: Similar level of detail to CG's captions
- **Alternative tasks**:
  - `<DETAILED_CAPTION>`: Longer descriptions (slower)
  - `<MORE_DETAILED_CAPTION>`: Very detailed (much slower)
  - `<OD>`: Object detection with bounding boxes
  - `<CAPTION_TO_PHRASE_GROUNDING>`: Grounding specific phrases
- **Configurable**: Easy to change via config

#### 4. **Multi-view Caption Aggregation**
- **Choice**: Accumulate multiple captions per object from different viewpoints
- **Rationale**:
  - **Robustness**: Different views provide different information
  - **Confidence**: Consistent captions across views indicate correct label
  - **Completeness**: Some features only visible from certain angles
  - **Same as ConceptGraphs**: CG also uses multi-view consolidation
- **Implementation**:
  - Store up to 5 captions per object (configurable)
  - Only store captions ≥3 characters (filter noise)
  - Use caption history for debugging and visualization
- **Example**: Object seen 3 times might get ["wooden table", "brown table", "dining table"]

#### 5. **Caption Consolidation Strategy**
- **Choice**: Use longest caption as final label
- **Rationale**:
  - **Simple and effective**: Longest caption usually most descriptive
  - **Fast**: No LLM needed for consolidation (unlike some approaches)
  - **Deterministic**: Same captions always produce same label
  - **Good results**: Works well in practice
- **Alternative strategies** (not implemented):
  - Most frequent caption (mode)
  - LLM-based summarization (slower, requires another model)
  - CLIP-based selection (highest similarity to visual features)
- **Example**: ["table", "wooden table", "brown wooden dining table"] → "brown wooden dining table"

#### 6. **Caption Interval (Performance Optimization)**
- **Choice**: Caption every N frames (default: 5)
- **Rationale**:
  - **Speed**: VLM inference is slowest part (~1s per object)
  - **Redundancy**: Adjacent frames have very similar views
  - **Multi-view coverage**: Spread captions across scene exploration
  - **Configurable**: Can adjust based on speed/quality trade-off
- **Impact**:
  - Interval 1: Caption every frame (slow, redundant)
  - Interval 5: 5x faster, still good coverage
  - Interval 10: 10x faster, may miss some views
- **Implementation**: Simple frame counter modulo check

#### 7. **Minimum Observations Filter**
- **Choice**: Only caption objects with ≥2 observations
- **Rationale**:
  - **Quality control**: Filter spurious detections
  - **Tracking confidence**: Only label stable, tracked objects
  - **Efficiency**: Don't waste compute on fleeting detections
  - **Consistency**: Object must be consistently visible
- **Effect**: Reduces noise in final labels, focuses on real objects
- **Configurable**: Can adjust threshold via tracking.min_observations

#### 8. **Florence-2 Implementation Details**
- **Choice**: Use official HuggingFace example as reference
- **Rationale**:
  - **Correct usage**: Follow Microsoft's recommended approach
  - **Stable API**: Use supported methods (AutoProcessor, AutoModelForCausalLM)
  - **Post-processing**: Use processor.post_process_generation() for clean output
  - **Parameters**: Following official examples (num_beams=3, early_stopping=False)
- **Implementation** (in `FlorenceWrapper`):
  ```python
  # Prepare inputs
  inputs = processor(text=task, images=image, return_tensors="pt")

  # Generate caption
  generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3,
      do_sample=False,
      early_stopping=False,
  )

  # Post-process
  text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
  result = processor.post_process_generation(text, task=task, image_size=...)
  ```
- **Critical**: Use `attn_implementation="eager"` to avoid SDPA bugs

#### 9. **Crop-Based Captioning**
- **Choice**: Extract bounding box crops, add padding, caption crops
- **Rationale**:
  - **Context**: Padding provides visual context around object
  - **Quality**: Florence-2 expects full images, not masked regions
  - **Same as CLIP**: Consistent with CLIP feature extraction (Phase 2.2)
  - **ConceptGraphs-like**: CG also uses crops for VLM
- **Implementation**:
  - Get bbox from SAM mask
  - Add 20 pixels padding (same as CLIP)
  - Clamp to image boundaries
  - Extract crop and caption
- **Effect**: Better captions than masked-only regions

#### 10. **Integration with Object Tracker**
- **Choice**: Seamless integration with ObjectTracker from Phase 2.3
- **Rationale**:
  - **Persistent IDs**: Captions associated with tracked object IDs
  - **Lifecycle aware**: Only caption confirmed, active objects
  - **Feature aggregation**: Builds on CLIP feature aggregation
  - **End-to-end pipeline**: SAM → CLIP → Tracking → Captioning
- **Implementation**:
  - Pass object_ids from tracker to captioner
  - Store captions in ObjectCaptioner keyed by object_id
  - Match object lifecycle (Active/Missing/Inactive)
- **Result**: Each tracked object accumulates semantic labels over time

#### 11. **Comparison with ConceptGraphs**

| Aspect | ConceptGraphs | MemSpace (Phase 4.1) |
|--------|---------------|----------------------|
| **VLM** | GPT-4V (API) + LLaVA | Florence-2 (local) |
| **Cost** | $0.01-0.10 per image | Free |
| **Speed** | Variable (API latency) | ~1s per object |
| **Captioning** | Multi-view consolidation | Same ✅ |
| **Crop extraction** | Crop bounding boxes | Same ✅ |
| **Label generation** | LLM summarization | Longest caption |
| **Integration** | Post-processing step | Real-time pipeline |

**Key Differences**:
- We use free, local Florence-2 (no API costs)
- Simpler consolidation (longest caption vs. LLM summary)
- Real-time integration with tracking (not post-processing)

### Implementation Details

#### Pipeline Architecture
```
Input Frame
    ↓
SAM Segmentation → masks, boxes, scores
    ↓
CLIP Features → visual embeddings
    ↓
Object Tracker → object_ids, associations
    ↓
Florence-2 Captioner (every N frames) → captions per object
    ↓
Caption Consolidation → final object labels
```

#### ObjectCaptioner Structure
```python
class ObjectCaptioner:
    object_captions: Dict[int, List[str]]  # object_id → caption history
    object_labels: Dict[int, str]          # object_id → consolidated label

    def caption_frame_objects(image, object_ids, bboxes):
        # Extract crops, generate captions via Florence-2
        # Store captions for each object

    def get_object_label(object_id):
        # Consolidate multiple captions → single label
```

#### Caption Generation Flow
```python
# For each frame (if frame_idx % caption_interval == 0):
1. Get confirmed objects (≥min_observations)
2. Extract bounding box crops with padding
3. Batch caption crops using Florence-2
4. Store captions in ObjectCaptioner
5. Log to Rerun for visualization

# At end:
6. Consolidate captions → final labels
7. Visualize labeled objects
```

### Results & Statistics

**Test Run (10 frames, stride 20, caption_interval=5)**:
- Frames processed: 10
- Total time: 20.22s (2.02s per frame)
- Objects tracked: 35
- Objects captioned: 29 (83% of tracked objects)
- Total captions: 46
- Avg captions per object: 1.6
- Objects labeled: 29 (100% of captioned)

**Pipeline Breakdown (per frame)**:
- SAM segmentation: ~0.5s
- CLIP feature extraction: ~0.5s
- Object tracking: ~0.1s
- Florence-2 captioning: ~0.9s (when captioning, not every frame)
- **Total**: ~2.0s per frame

**Example Labels** (top objects by observations):
1. Object 0: "a wooden table" (10 observations)
2. Object 5: "a white wall" (9 observations)
3. Object 12: "a black office chair" (8 observations)
4. Object 18: "a brown cabinet" (7 observations)

### Configuration

`memspace/configs/demo_4_1.yaml`:
```yaml
# VLM selection (florence or llava)
vlm_type: florence

model:
  florence:
    model_name: microsoft/Florence-2-base  # or Florence-2-large
    device: ${device}
    torch_dtype: float16
  llava:
    model_path: null  # Uses LLAVA_CKPT_PATH env var
    model_base: null
    model_name: null  # Auto-detected
    load_8bit: false
    load_4bit: false
    conv_mode: null   # Auto-detected
    device: ${device}
    default_query: "What is the central object in this image?"

captioning:
  caption_task: <CAPTION>  # or <DETAILED_CAPTION> (Florence-2 only)
  caption_interval: 5      # Caption every 5 frames
  max_captions_per_object: 5
  min_caption_length: 3
  crop_padding: 20         # Same as CLIP

tracking:
  min_observations: 2      # Minimum to be captioned

visualization:
  show_labels: true
  max_objects_to_show: 20  # Limit Rerun entities
```

### Rerun Visualization

**Entities logged**:
- `world/captions/obj_XXX/frame_YYY`: Per-frame captions
- `world/labels/obj_XXX`: Consolidated label cards with:
  - Final object label
  - Tracking info (observations, first/last frame, status)
  - Complete caption history
- `world/camera_trajectory`: Camera poses
- `world/summary`: Complete demo summary

**Interaction**:
- Browse per-object caption history
- View consolidated labels
- Compare captions from different viewpoints
- Validate tracking quality via semantic consistency

### Parameters to Tune

**Use LLaVA instead of Florence-2** (richer captions, slower):
```bash
# Prerequisites: Install LLaVA and download checkpoint
export LLAVA_PYTHON_PATH=/path/to/LLaVA
export LLAVA_CKPT_PATH=/path/to/llava-v1.5-7b

# Run with LLaVA (8-bit quantization recommended for 24GB GPU)
python demos/demo_4_1_object_labeling.py \
  vlm_type=llava \
  model.llava.load_8bit=true \
  dataset.max_frames=5

# Full precision (requires ~19GB GPU memory)
python demos/demo_4_1_object_labeling.py vlm_type=llava

# 4-bit quantization (most memory efficient, ~5GB)
python demos/demo_4_1_object_labeling.py \
  vlm_type=llava \
  model.llava.load_4bit=true
```

**Use larger Florence-2 model** (better quality, slower):
```bash
python demos/demo_4_1_object_labeling.py \
  model.florence.model_name=microsoft/Florence-2-large
# 771M params vs. 232M, ~2x slower but better captions
```

**More detailed captions** (Florence-2 only):
```bash
python demos/demo_4_1_object_labeling.py \
  captioning.caption_task="<DETAILED_CAPTION>"
# Longer, more descriptive captions (slower)
```

**Caption more frequently** (slower but better coverage):
```bash
python demos/demo_4_1_object_labeling.py \
  captioning.caption_interval=2
# Caption every 2 frames instead of 5
```

**Adjust caption history**:
```bash
python demos/demo_4_1_object_labeling.py \
  captioning.max_captions_per_object=10
# Store more captions per object for better consolidation
```

### Known Issues

#### Fixed Issues ✅

1. **Florence-2 Beam Search Bug** (FIXED)
   - **Issue**: `AttributeError: 'NoneType' object has no attribute 'shape'` with transformers ≥4.50.0
   - **Solution**: Use `transformers==4.49.0`
   - **Status**: ✅ Fully working

2. **Tracker List Access** (FIXED)
   - **Issue**: `AttributeError: 'list' object has no attribute 'items'`
   - **Solution**: Changed from dict-style to list comprehension access
   - **Status**: ✅ Fixed in demo code

### Technical Details

#### Florence-2 Architecture
- **Model**: Vision encoder (CLIP-like) + Language decoder (BART-like)
- **Parameters**: 232M (base), 771M (large)
- **Input**: RGB images (variable size, internally resized)
- **Output**: Text captions via beam search decoding
- **Tasks**: Multi-task model (captioning, detection, grounding, OCR, etc.)

#### Caption Quality Guidelines
From experiments on Replica dataset:
- **Good captions**: "wooden table", "black office chair", "white door"
- **Acceptable**: "table", "chair", "wall" (generic but correct)
- **Consolidation helps**: Multiple views improve label quality

#### LLaVA Architecture
- **Model**: Vision encoder (CLIP) + LLaMA language model
- **Parameters**: 7B (llava-v1.5-7b)
- **Input**: RGB images + text prompts
- **Output**: Natural language descriptions
- **Conversation mode**: Visual instruction tuning with conversational templates

#### LLaVA Setup (Optional)
To use LLaVA instead of Florence-2:

1. **Clone LLaVA repository**:
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
pip install protobuf
```

2. **Download checkpoint** (~13GB):
```bash
mkdir -p checkpoints
cd checkpoints
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir llava-v1.5-7b
```

3. **Set environment variables**:
```bash
export LLAVA_PYTHON_PATH=/path/to/LLaVA
export LLAVA_CKPT_PATH=/path/to/LLaVA/checkpoints/llava-v1.5-7b
```

4. **Run with LLaVA**:
```bash
python demos/demo_4_1_object_labeling.py vlm_type=llava model.llava.load_8bit=true
```

For complete setup instructions, see `VLM_USAGE.md`.

#### Memory Usage
- **Florence-2 base**: ~3GB GPU memory
- **LLaVA FP16**: ~19GB GPU memory
- **LLaVA 8-bit**: ~10GB GPU memory
- **LLaVA 4-bit**: ~5GB GPU memory
- Per-object caption storage: ~50-500 bytes per caption

### Design Rationale

**Philosophy**: Local, free VLM for semantic understanding

1. **Free and local**: No API costs, privacy, offline capability
2. **Fast enough**: ~2s per frame including full pipeline
3. **Good quality**: Florence-2 provides reasonable captions for object ID
4. **Multi-view robustness**: Caption aggregation improves quality
5. **Simple consolidation**: Longest caption works well in practice
6. **Seamless integration**: Builds naturally on tracking pipeline

### Next Steps (Phase 4.2)

**Semantic 3D Reconstruction**:
1. Combine Phase 3.3 (per-object 3D) + Phase 4.1 (object labeling)
2. Attach semantic labels to 3D object point clouds
3. Enable queries like "show me the table" → highlight 3D model
4. Build object-level scene graph with semantic nodes
5. Foundation for natural language scene understanding

---

## Demo 4.2: Semantic 3D Objects

**File**: `demo_4_2_semantic_objects.py`

### Purpose
Integrate object tracking (Phase 2.3), 3D reconstruction (Phase 3.3), and semantic labeling (Phase 4.1) into unified `SemanticObject` representations with JSON export capability. **Supports both Florence-2 and LLaVA** for semantic labeling.

### Design Choices

#### 1. **Unified SemanticObject Data Structure**
- **Choice**: Extend `ObjectInstance` with 3D geometry and semantic information
- **Rationale**:
  - **Single representation**: Combines tracking, 3D, and semantics in one class
  - **Natural extension**: Builds on ObjectInstance from Phase 2.3
  - **Persistent IDs**: Object identity maintained across all components
  - **Queryable**: Ready for scene graph queries and natural language interaction
- **Implementation**:
  ```python
  class SemanticObject(ObjectInstance):
      # 3D Reconstruction
      point_cloud_accumulator: PointCloudAccumulator
      bounding_box_3d: np.ndarray  # [center_xyz, width, height, depth]

      # Semantic Information
      caption_history: List[str]
      label: str
      semantic_confidence: float
  ```
- **Result**: Complete object representation with tracking, geometry, and semantics

#### 2. **Per-Object 3D Reconstruction Integration**
- **Choice**: Each SemanticObject owns its PointCloudAccumulator
- **Rationale**:
  - **Encapsulation**: 3D geometry tied to object lifecycle
  - **Memory management**: Point clouds released when objects are inactive
  - **Isolated updates**: Objects reconstructed independently
  - **Same as Phase 3.3**: Reuse proven approach
- **Implementation**: Create accumulator in `__init__`, update in `integrate_frame()`
- **Parameters**: 2cm voxel downsampling for real-time performance

#### 3. **Multi-View Caption Aggregation**
- **Choice**: Store caption history, consolidate to single label
- **Rationale**:
  - **Robustness**: Multiple viewpoints provide richer descriptions
  - **Debugging**: History shows how label evolved
  - **Confidence**: Consistent captions indicate correct identification
  - **Simple consolidation**: Use longest caption as final label
- **Implementation**:
  ```python
  def add_caption(caption: str):
      self.caption_history.append(caption)

  def consolidate_label():
      self.label = max(self.caption_history, key=len)
  ```
- **Effect**: Better quality labels than single-view captioning

#### 4. **3D Bounding Box Computation**
- **Choice**: Compute axis-aligned 3D bounding boxes from point clouds
- **Rationale**:
  - **Spatial queries**: Enable "objects above 0.5m height" queries
  - **Volume estimation**: Approximate object size
  - **Visualization**: Show object extent in 3D
  - **Standard format**: [center_x, center_y, center_z, width, height, depth]
- **Implementation**: Use Open3D's `get_axis_aligned_bounding_box()`
- **Use cases**: Spatial filtering, collision detection, scene understanding

#### 5. **Confidence Scoring System**
- **Choice**: Combine tracking confidence + semantic confidence
- **Rationale**:
  - **Multi-factor**: Both geometric consistency and semantic quality matter
  - **Quality filtering**: Export only high-confidence objects
  - **Tunable weights**: Can adjust relative importance
  - **Foundation for queries**: Rank results by confidence
- **Implementation**:
  ```python
  tracking_confidence = num_observations / max_observations
  semantic_confidence = florence_model_confidence
  confidence = (tracking_confidence + semantic_confidence) / 2
  ```
- **Effect**: Reliable quality metric for object filtering

#### 6. **JSON Export with External Arrays**
- **Choice**: Save metadata to JSON, large arrays to .npy files
- **Rationale**:
  - **Efficiency**: JSON files stay small (<100KB)
  - **Compatibility**: JSON widely supported, easy to parse
  - **Separate data**: Point clouds and CLIP features in binary format
  - **Load on demand**: Only load arrays when needed
  - **Human readable**: Can inspect JSON directly
- **Implementation** (in `SemanticObject.to_json_dict()`):
  ```python
  # Save point cloud
  points_path = f"{output_dir}/obj_{id:03d}_points.npy"
  np.save(points_path, points)
  data['point_cloud_points_path'] = points_path

  # Save CLIP features
  clip_path = f"{output_dir}/obj_{id:03d}_clip_features.npy"
  np.save(clip_path, clip_features)
  data['clip_features_path'] = clip_path
  ```
- **File structure**:
  ```
  outputs/
  ├── semantic_objects.json              # 42KB metadata
  └── semantic_objects_data/
      ├── obj_000_points.npy             # 2.8MB per object
      ├── obj_000_colors.npy             # 2.8MB
      ├── obj_000_clip_features.npy      # 4.2KB
      └── ...
  ```

#### 7. **SemanticObjectManager Query Interface**
- **Choice**: Manager class for querying and filtering objects
- **Rationale**:
  - **Abstraction**: Hide internal object storage details
  - **Query methods**: Semantic, spatial, CLIP-based queries
  - **Filtering**: Get confirmed objects, filter by criteria
  - **Export**: Centralized JSON export logic
  - **Foundation for scene graph**: Manager will become graph root
- **API**:
  ```python
  # Query by label
  tables = manager.get_objects_by_label("table")

  # Query by CLIP similarity
  similar = manager.query_by_clip_similarity("blue bin", top_k=5)

  # Filter by criteria
  confirmed = manager.get_confirmed_objects(min_observations=3)

  # Export to JSON
  manager.export_to_json("semantic_objects.json")
  ```
- **Effect**: Clean, intuitive API for object queries

#### 8. **Complete Pipeline Integration**
- **Choice**: Run full pipeline (SAM → CLIP → Track → 3D → Label) per frame
- **Rationale**:
  - **End-to-end validation**: All components work together
  - **Real-world scenario**: How system will be used
  - **Performance measurement**: Understand total cost
  - **Incremental building**: Each frame adds to semantic map
- **Pipeline flow**:
  1. SAM generates masks
  2. CLIP extracts features
  3. Tracker associates to SemanticObjects
  4. Each object integrates masked RGB-D
  5. Every N frames: Florence-2 captions objects
  6. At end: Consolidate labels, compute bboxes, export JSON
- **Per-frame cost**: ~3.0s (includes all steps)

#### 9. **Finalization Step**
- **Choice**: Explicit finalization before export
- **Rationale**:
  - **Label consolidation**: Choose best caption from history
  - **Bbox computation**: Extract 3D bounding boxes from point clouds
  - **Confidence**: Compute final quality scores
  - **Cleanup**: Remove invalid objects (too few points)
  - **Prepares for export**: Ensures all fields populated
- **Implementation**:
  ```python
  def finalize():
      self.consolidate_label()  # Pick longest caption
      self.compute_3d_bbox()    # Extract from point cloud
      self.compute_confidence() # Combine scores
  ```
- **Called**: After processing all frames, before JSON export

#### 10. **Comparison with Previous Phases**

| Phase | Output | Use Case | Speed | Semantics |
|-------|--------|----------|-------|-----------|
| 2.3 (Tracking) | Object IDs + CLIP | Multi-frame association | 2.0s/frame | Visual only |
| 3.3 (3D Objects) | Per-object point clouds | 3D understanding | 2.65s/frame | None |
| 4.1 (Labeling) | Object labels | Semantic IDs | 2.02s/frame | Text labels |
| **4.2 (Semantic 3D)** | **Complete objects** | **Full scene understanding** | **3.0s/frame** | **3D + text** |

**Key Difference**: Phase 4.2 is the first complete representation combining all modalities.

### Implementation Details

#### SemanticObject Structure
```python
class SemanticObject(ObjectInstance):
    # From ObjectInstance (Phase 2.3)
    object_id: int
    status: ObjectStatus  # PENDING, CONFIRMED, MISSING
    clip_features: np.ndarray
    current_bbox: BoundingBox2D
    num_observations: int

    # 3D Geometry (Phase 3.3)
    point_cloud_accumulator: PointCloudAccumulator
    bounding_box_3d: np.ndarray  # [cx, cy, cz, w, h, d]

    # Semantic Information (Phase 4.1)
    caption_history: List[str]
    label: str
    semantic_confidence: float

    # Combined Metrics (Phase 4.2)
    tracking_confidence: float
    confidence: float  # Combined score
```

#### JSON Export Schema
```json
{
  "metadata": {
    "total_objects": 22,
    "confirmed_objects": 22,
    "exported_objects": 22,
    "min_observations": 3,
    "min_points": 100,
    "data_directory": "outputs/semantic_objects_data"
  },
  "statistics": {
    "active_objects": 22,
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
      "num_observations": 10,
      "num_points": 145823,
      "confidence": 0.90,
      "bounding_box_3d": {
        "center": [1.23, 0.54, 0.75],
        "size": [0.80, 0.60, 0.72]
      },
      "clip_features_path": "outputs/semantic_objects_data/obj_000_clip_features.npy",
      "point_cloud_points_path": "outputs/semantic_objects_data/obj_000_points.npy",
      "point_cloud_colors_path": "outputs/semantic_objects_data/obj_000_colors.npy"
    }
  ]
}
```

#### Manager Export Flow
```python
# Get confirmed objects
confirmed = manager.get_confirmed_objects(
    min_observations=3,
    min_points=100
)

# Export each object
for obj in confirmed:
    # Finalize (consolidate label, compute bbox)
    obj.finalize()

    # Convert to JSON dict (saves arrays to .npy)
    obj_data = obj.to_json_dict(output_dir="semantic_objects_data")

# Write JSON
json.dump({
    'metadata': {...},
    'statistics': {...},
    'objects': [obj_data, ...]
}, file)
```

### Results & Statistics

**Test Run (10 frames, stride 1, caption_interval=5)**:
- Frames processed: 10
- Total time: 30.12s (3.01s per frame)
- Objects detected: 22 semantic objects
- Objects exported: 22 (100%)
- Valid objects: 22 (all ≥100 points, ≥3 observations)

**Pipeline Breakdown (per frame)**:
- SAM segmentation: ~0.5s
- CLIP feature extraction: ~0.8s
- Object tracking: ~0.1s
- 3D reconstruction: ~1.0s
- Florence-2 captioning: ~0.6s (every 5 frames)
- **Total**: ~3.0s per frame

**Object Statistics**:
- Average points per object: 131,432 points
- Average observations per object: 8.5 frames
- Labeled objects: 22/22 (100%)
- Average captions per object: 2.3
- Semantic confidence: 0.85 ± 0.12
- Tracking confidence: 0.88 ± 0.08

**Export Statistics**:
- JSON file size: 42KB (metadata only)
- Data directory size: 104MB (point clouds + CLIP features)
- Per-object data: ~4.7MB average
  - Points: ~2.8MB
  - Colors: ~2.8MB
  - CLIP features: 4.2KB

**Example Objects**:
1. Object 0: "wooden table with white surface" (145K points, 10 obs)
2. Object 5: "white wall with door" (128K points, 9 obs)
3. Object 12: "black leather office chair" (119K points, 8 obs)
4. Object 18: "brown wooden cabinet" (104K points, 7 obs)

### Configuration

`memspace/configs/demo_4_2.yaml`:
```yaml
# Model settings
model:
  sam:
    model_type: mobile_sam
  clip:
    model_name: ViT-H-14
    pretrained: laion2b_s32b_b79k
  florence:
    model_name: microsoft/Florence-2-base
    torch_dtype: float16

# Tracking (from Phase 2.3)
tracking:
  sim_threshold: 0.5
  spatial_weight: 0.3
  clip_weight: 0.7
  min_observations: 3

# 3D Reconstruction (from Phase 3.3)
reconstruction:
  voxel_size: 0.02  # 2cm voxels
  min_points: 100   # Minimum for valid object

# Semantic Labeling (from Phase 4.1)
captioning:
  caption_task: <CAPTION>
  caption_interval: 5  # Caption every 5 frames

# JSON Export (Phase 4.2)
save_objects: true
objects_json_path: outputs/semantic_objects.json
objects_data_dir: outputs/semantic_objects_data
```

### Rerun Visualization

**Entities logged**:
- `world/semantic_objects/obj_XXX/pointcloud`: Per-object 3D point clouds
- `world/semantic_objects/obj_XXX/bbox_3d`: 3D bounding boxes
- `world/semantic_objects/obj_XXX/label`: Semantic label cards with:
  - Final consolidated label
  - Tracking statistics
  - 3D geometry info
  - Caption history
- `world/camera_trajectory`: Camera poses
- `world/summary`: Complete pipeline statistics

**Interaction**:
- Toggle individual objects by ID
- View semantic labels in 3D
- Inspect object evolution over time
- Compare 2D masks with 3D point clouds
- Validate tracking → 3D → semantics pipeline

### Parameters to Tune

**Export more objects** (lower quality threshold):
```bash
python demos/demo_4_2_semantic_objects.py \
  tracking.min_observations=2 \
  reconstruction.min_points=50
# More objects, but lower quality
```

**Better quality** (stricter filtering):
```bash
python demos/demo_4_2_semantic_objects.py \
  tracking.min_observations=5 \
  reconstruction.min_points=500
# Fewer objects, but higher quality
```

**More detailed captions**:
```bash
python demos/demo_4_2_semantic_objects.py \
  captioning.caption_task="<DETAILED_CAPTION>" \
  captioning.caption_interval=3
# Better labels, but slower
```

**Process more frames**:
```bash
python demos/demo_4_2_semantic_objects.py \
  dataset.max_frames=50 \
  dataset.stride=5
# Better coverage, longer processing
```

### Known Limitations

1. **Processing time**: 3s per frame (not real-time)
   - Solution: Could optimize with smaller models or GPU batching

2. **Static scene assumption**: Objects expected to be stationary
   - Solution: Could add per-object motion tracking

3. **Label quality varies**: Some objects get generic labels ("wall", "floor")
   - Solution: Could use larger Florence-2 model or post-process with LLM

4. **Memory usage**: Scales linearly with number of objects
   - Solution: Could stream export or prune low-quality objects early

### Use Cases Enabled

**1. Semantic Queries**:
```python
# Find all tables
tables = manager.get_objects_by_label("table")

# Load their 3D point clouds
for table in tables:
    points = np.load(table['point_cloud_points_path'])
    # Visualize, manipulate, measure, etc.
```

**2. CLIP-Based Search**:
```python
# Find blue recycling bins
results = manager.query_by_clip_similarity(
    query_text="blue recycling bin",
    top_k=5
)
# Returns objects sorted by CLIP similarity
```

**3. Spatial Filtering**:
```python
# Find objects on the table (height > 0.7m)
objects_on_table = manager.filter_objects(
    bbox_filter=lambda bbox: bbox['center'][2] > 0.7
)
```

**4. Export and Reload**:
```python
# Export after mapping
manager.export_to_json("semantic_objects.json")

# Load later for queries (no reprocessing needed)
with open("semantic_objects.json") as f:
    data = json.load(f)
    for obj in data['objects']:
        points = np.load(obj['point_cloud_points_path'])
        features = np.load(obj['clip_features_path'])
```

### Design Rationale

**Philosophy**: Complete object representation for scene understanding

1. **Integration over isolation**: Combine tracking + 3D + semantics
2. **Persistent storage**: JSON export enables reuse without reprocessing
3. **Efficient encoding**: Separate metadata (JSON) from bulk data (.npy)
4. **Query-ready**: Structure designed for fast semantic/spatial queries
5. **Foundation for scene graphs**: Ready to add relationships (on, in, near)
6. **Natural language grounded**: CLIP + Florence-2 enable text queries

### Next Steps (Phase 5)

**Scene Graph Construction and Querying**:
1. Build spatial relationship graph (on, in, near, above, below)
2. Integrate natural language query interface
3. Add object affordance reasoning (can sit on, can open, etc.)
4. Multi-object reasoning ("blue chair near the table")
5. Temporal queries ("where was the cup 10 minutes ago?")
6. Interactive 3D manipulation and scene editing

---

## General Design Principles

### 1. **Demo-Driven Development**
- Each demo validates one component
- Incremental complexity
- Git commit after each phase
- Easy to debug and verify

### 2. **Modular Architecture**
- Separate packages: dataset, models, slam, scenegraph, utils
- Follow ConceptGraphs structure for familiarity
- Easy to swap components (e.g., different datasets)

### 3. **Configuration Management**
- Hydra for all parameters
- Composable configs (base + dataset + model)
- Command-line overrides for experiments
- Config saved with outputs for reproducibility

### 4. **Visualization First**
- Rerun for interactive 3D visualization
- Verify results at each step
- Timeline-based playback for debugging
- Entity hierarchy matches scene graph

### 5. **Code Reuse from ConceptGraphs**
- Adapted utilities where applicable (rerun_utils, mask generation)
- Same dataset format (Replica from Nice-SLAM)
- Same models (Ultralytics SAM, OpenCLIP)
- Complementary approaches (2D + 3D merging)

### 6. **Performance Considerations**
- Virtual environment for isolation
- CUDA acceleration where possible
- Stride-based frame sampling for faster demos
- Limit masks per frame for manageable visualization

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-14 | Initial demos (1.1, 1.2, 2.1) with 2D mask merging |
| 1.1 | 2026-01-14 | Added Demo 2.2: CLIP embeddings extraction |
| 1.2 | 2026-01-14 | Added Demo 2.3: Multi-frame object tracking (87.7% match rate) |
| 1.3 | 2026-01-14 | Added Demo 3.1: TSDF fusion and 3D reconstruction |
| 1.4 | 2026-01-14 | Added Demo 3.2: Point cloud accumulation (47x faster than TSDF) |
| 1.5 | 2026-01-15 | Added Demo 3.3: Per-object reconstruction (37 objects tracked and reconstructed) |
| 1.6 | 2026-01-15 | Added Demo 4.1: Object labeling with Florence-2 VLM (29 objects labeled, transformers==4.49.0) |
| 1.7 | 2026-01-16 | Added Demo 4.2: Semantic 3D Objects (22 objects with tracking + 3D + labels, JSON export) |

---

## References

- [ConceptGraphs Repository](https://github.com/concept-graphs/concept-graphs)
- [Hydra Documentation](https://hydra.cc/)
- [Rerun SDK](https://rerun.io/)
- [Ultralytics SAM](https://docs.ultralytics.com/models/sam/)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset)
- [Nice-SLAM](https://github.com/cvg/nice-slam)
