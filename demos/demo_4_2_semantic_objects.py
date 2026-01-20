#!/usr/bin/env python3
"""
Demo 4.2: Semantic 3D Objects

This demo demonstrates the complete semantic 3D object understanding pipeline:
- Object tracking (Phase 2.3)
- Per-object 3D reconstruction (Phase 3.3)
- Semantic labeling with Florence-2 (Phase 4.1)
- Unified SemanticObject representation

Complete Pipeline: SAM → CLIP → Tracking → 3D Reconstruction → Labeling
"""

import sys
from pathlib import Path
import time
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import rerun as rr
import open3d as o3d
from typing import List

from memspace.dataset import get_dataset
from memspace.models.sam_wrapper import SAMWrapper
from memspace.models.clip_wrapper import CLIPWrapper
from memspace.models.florence_wrapper import FlorenceWrapper
from memspace.models.llava_wrapper import LLaVAWrapper
from memspace.scenegraph.semantic_object import SemanticObject, SemanticObjectManager
from memspace.utils import rerun_utils
from memspace.utils.mask_utils import merge_overlapping_masks


class SemanticObjectTracker:
    """
    Tracks semantic objects with 3D reconstruction and labeling

    Extends basic tracking to create SemanticObject instances
    """

    def __init__(
        self,
        sim_threshold: float = 0.5,
        spatial_weight: float = 0.3,
        clip_weight: float = 0.7,
        max_missing_frames: int = 10,
        min_observations: int = 2,
        voxel_size: float = 0.02,
    ):
        self.sim_threshold = sim_threshold
        self.spatial_weight = spatial_weight
        self.clip_weight = clip_weight
        self.max_missing_frames = max_missing_frames
        self.min_observations = min_observations
        self.voxel_size = voxel_size

        # Semantic objects
        self.objects: List[SemanticObject] = []
        self.next_object_id = 0

        # Statistics
        self.total_detections = 0
        self.total_matches = 0

    def compute_similarity(
        self,
        bbox1: np.ndarray,
        bbox2: np.ndarray,
        feat1: np.ndarray,
        feat2: np.ndarray,
    ) -> float:
        """Compute combined spatial + semantic similarity"""
        # Spatial similarity (IoU)
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            spatial_sim = 0.0
        else:
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union_area = bbox1_area + bbox2_area - inter_area
            spatial_sim = inter_area / union_area if union_area > 0 else 0.0

        # CLIP similarity
        clip_sim = float(np.dot(feat1, feat2))

        # Combined
        return self.spatial_weight * spatial_sim + self.clip_weight * clip_sim

    def update(
        self,
        frame_idx: int,
        masks: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        clip_features: np.ndarray,
    ):
        """Update tracker with new detections"""
        self.total_detections += len(masks)

        # Mark all objects as potentially missing
        for obj in self.objects:
            if obj.is_active():
                obj.num_missing_frames += 1

        # Match detections to existing objects
        object_ids = []
        match_types = []

        for i in range(len(masks)):
            mask = masks[i]
            bbox = bboxes[i]
            score = scores[i]
            feat = clip_features[i]

            # Find best matching object
            best_obj = None
            best_sim = -1.0

            for obj in self.objects:
                if obj.status.value in ['active', 'missing']:
                    sim = self.compute_similarity(
                        bbox, obj.current_bbox, feat, obj.clip_features
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_obj = obj

            # Match or create new
            if best_sim >= self.sim_threshold and best_obj is not None:
                # Match to existing object
                best_obj.update(frame_idx, mask, bbox, score, feat)
                object_ids.append(best_obj.object_id)
                match_types.append('matched')
                self.total_matches += 1
            else:
                # Create new object
                new_obj = SemanticObject(
                    object_id=self.next_object_id,
                    voxel_size=self.voxel_size,
                    first_seen_frame=frame_idx,
                )
                new_obj.update(frame_idx, mask, bbox, score, feat)
                self.objects.append(new_obj)
                object_ids.append(new_obj.object_id)
                match_types.append('new')
                self.next_object_id += 1

        # Update status for missing objects
        for obj in self.objects:
            if obj.num_missing_frames >= self.max_missing_frames:
                obj.status = obj.status.__class__.INACTIVE
            elif obj.num_missing_frames > 0:
                obj.status = obj.status.__class__.MISSING

        return object_ids, match_types


@hydra.main(config_path="../memspace/configs", config_name="demo_4_2", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 70)
    print("MemSpace Demo 4.2: Semantic 3D Objects")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        spawn = cfg.get('rerun_spawn', True)
        save_path = cfg.get('rerun_save_path', None)

        if save_path:
            rr.init("memspace/demo_4_2", spawn=spawn)
            rr.save(save_path)
            print(f"✓ Rerun recording to: {save_path}")
        else:
            rr.init("memspace/demo_4_2", spawn=spawn)
            if spawn:
                print("✓ Rerun viewer spawned")

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    else:
        print("⚠  Rerun visualization disabled")
        return

    print()

    # Initialize models
    print(f"🤖 Initializing SAM model: {cfg.model.sam.model_type}")
    sam_model = SAMWrapper(
        model_type=cfg.model.sam.model_type,
        device=cfg.device,
    )
    print()

    print(f"🤖 Initializing CLIP model: {cfg.model.clip.model_name}")
    clip_model = CLIPWrapper(
        model_name=cfg.model.clip.model_name,
        pretrained=cfg.model.clip.pretrained,
        device=cfg.device,
    )
    print()

    # Initialize VLM based on vlm_type
    vlm_type = cfg.get('vlm_type', 'florence').lower()

    if vlm_type == "florence":
        print(f"🤖 Initializing Florence-2 model: {cfg.model.florence.model_name}")
        vlm_model = FlorenceWrapper(
            model_name=cfg.model.florence.model_name,
            device=cfg.model.florence.device,
            torch_dtype=cfg.model.florence.get('torch_dtype', 'float16'),
        )
        caption_task = cfg.captioning.caption_task
        caption_query = None
        print()
    elif vlm_type == "llava":
        print(f"🤖 Initializing LLaVA model")
        llava_cfg = cfg.model.llava
        vlm_model = LLaVAWrapper(
            model_path=llava_cfg.get('model_path'),
            model_base=llava_cfg.get('model_base'),
            model_name=llava_cfg.get('model_name'),
            load_8bit=llava_cfg.get('load_8bit', False),
            load_4bit=llava_cfg.get('load_4bit', False),
            device=llava_cfg.get('device', cfg.device),
            conv_mode=llava_cfg.get('conv_mode'),
        )
        caption_task = None
        caption_query = llava_cfg.get('default_query', 'What is the central object in this image?')
        print()
    else:
        raise ValueError(f"Unknown vlm_type: {vlm_type}. Must be 'florence' or 'llava'.")

    # Initialize Semantic Object Tracker
    track_cfg = cfg.tracking
    print(f"🎯 Initializing Semantic Object Tracker")
    tracker = SemanticObjectTracker(
        sim_threshold=track_cfg.sim_threshold,
        spatial_weight=track_cfg.spatial_weight,
        clip_weight=track_cfg.clip_weight,
        max_missing_frames=track_cfg.max_missing_frames,
        min_observations=track_cfg.min_observations,
        voxel_size=cfg.reconstruction.voxel_size,
    )
    print()

    # Load dataset
    print(f"📂 Loading  dataset from: {cfg.dataset.dataset_path}")

    try:
        dataset = get_dataset(cfg.dataset, device=cfg.device)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"✓ Dataset loaded: {len(dataset)} frames")
    print()

    # Get settings
    seg_cfg = cfg.segmentation
    clip_cfg = cfg.clip_features
    caption_cfg = cfg.captioning
    recon_cfg = cfg.reconstruction

    print("🎬 Processing frames with complete pipeline...")
    print(f"   Pipeline: SAM → CLIP → Tracking → 3D Reconstruction → Labeling")
    print(f"   Frames: {len(dataset)}")
    print()

    start_time = time.time()

    for frame_idx in range(len(dataset)):
        if frame_idx % 5 == 0 or frame_idx == len(dataset) - 1:
            print(f"Frame {frame_idx+1}/{len(dataset)}")

        # Load frame
        color, depth, intrinsics, pose = dataset[frame_idx]
        color_np = color.cpu().numpy().astype(np.uint8)
        depth_np = depth.cpu().numpy().astype(np.float32)

        # 1. SAM Segmentation
        masks, boxes, scores = sam_model.generate_masks(
            color_np,
            min_mask_region_area=seg_cfg.min_mask_area,
        )

        if len(masks) == 0:
            continue

        # Merge overlapping masks
        if seg_cfg.get('merge_masks', True):
            masks, boxes, scores = merge_overlapping_masks(
                masks, boxes, scores,
                iou_threshold=seg_cfg.get('merge_iou_threshold', 0.5),
                containment_threshold=seg_cfg.get('merge_containment_threshold', 0.85),
            )

        # Limit masks
        if len(masks) > seg_cfg.max_masks_per_frame:
            sorted_idx = np.argsort(scores)[::-1][:seg_cfg.max_masks_per_frame]
            masks = masks[sorted_idx]
            boxes = boxes[sorted_idx]
            scores = scores[sorted_idx]

        # 2. CLIP Features
        crops, features = clip_model.extract_mask_features(
            color_np,
            boxes,
            padding=clip_cfg.padding,
            batch_size=clip_cfg.batch_size,
        )

        # 3. Object Tracking
        object_ids, match_types = tracker.update(
            frame_idx=frame_idx,
            masks=masks,
            bboxes=boxes,
            scores=scores,
            clip_features=features,
        )

        # 4. Per-Object 3D Reconstruction
        for i, obj_id in enumerate(object_ids):
            # Find object
            obj = next((o for o in tracker.objects if o.object_id == obj_id), None)
            if obj is None:
                continue

            # Create masked depth (zero out non-object pixels)
            masked_depth = depth_np.copy()
            masked_depth[~masks[i]] = 0.0

            # Integrate into object's point cloud
            obj.integrate_frame(
                color=color_np,
                depth=masked_depth,
                intrinsics=intrinsics.cpu().numpy() if isinstance(intrinsics, torch.Tensor) else intrinsics,
                extrinsics=pose.cpu().numpy() if isinstance(pose, torch.Tensor) else pose,
            )

        # 5. Semantic Labeling (every N frames)
        if frame_idx % caption_cfg.caption_interval == 0:
            # Get confirmed objects
            confirmed_indices = []
            confirmed_ids = []
            confirmed_boxes = []

            for i, obj_id in enumerate(object_ids):
                obj = next((o for o in tracker.objects if o.object_id == obj_id), None)
                if obj and obj.num_observations >= track_cfg.min_observations:
                    confirmed_indices.append(i)
                    confirmed_ids.append(obj_id)
                    confirmed_boxes.append(boxes[i])

            if len(confirmed_ids) > 0:
                confirmed_boxes_np = np.array(confirmed_boxes)

                # Extract crops and caption
                H, W = color_np.shape[:2]
                captions = []

                for bbox in confirmed_boxes_np:
                    x1, y1, x2, y2 = bbox.astype(int)

                    # Add padding
                    padding = caption_cfg.crop_padding
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(W, x2 + padding)
                    y2 = min(H, y2 + padding)

                    # Extract crop
                    crop = color_np[y1:y2, x1:x2]

                    # Caption with appropriate VLM
                    if vlm_type == "florence":
                        caption = vlm_model.caption_image(
                            crop,
                            task=caption_task,
                        )
                    else:  # llava
                        caption = vlm_model.caption_image(
                            crop,
                            query=caption_query,
                        )
                    captions.append(caption)

                # Add captions to objects
                for obj_id, caption in zip(confirmed_ids, captions):
                    obj = next((o for o in tracker.objects if o.object_id == obj_id), None)
                    if obj:
                        obj.add_caption(caption)

    total_time = time.time() - start_time

    print()
    print(f"✓ Processed {len(dataset)} frames in {total_time:.2f}s ({total_time/len(dataset):.3f}s per frame)")
    print()

    # Finalize all objects
    print("🏗️  Finalizing semantic objects...")
    for obj in tracker.objects:
        obj.finalize()

    # Create manager
    manager = SemanticObjectManager(tracker.objects)

    # Get confirmed objects
    confirmed = manager.get_confirmed_objects(
        min_observations=track_cfg.min_observations,
        min_points=recon_cfg.min_points,
    )

    print(f"✓ Finalized {len(confirmed)} confirmed semantic objects")
    print()

    # Get statistics
    stats = manager.get_statistics()

    print(f"📊 Semantic Object Statistics:")
    print(f"   Total objects tracked: {stats['total_objects']}")
    print(f"   Confirmed objects: {stats['confirmed_objects']}")
    print(f"   Objects with labels: {stats['objects_with_labels']}")
    print(f"   Avg points per object: {stats['avg_points_per_object']:.0f}")
    print(f"   Avg observations per object: {stats['avg_observations_per_object']:.1f}")
    print(f"   Avg confidence: {stats['avg_confidence']:.2f}")
    print()

    # Visualize in Rerun
    if cfg.use_rerun:
        print("📊 Logging semantic 3D objects to Rerun...")

        # Sort by num points
        confirmed_sorted = sorted(confirmed, key=lambda x: x.get_num_points(), reverse=True)
        objects_to_show = confirmed_sorted[:cfg.visualization.max_objects_to_show]

        for obj in objects_to_show:
            # Get point cloud
            pcd = obj.get_point_cloud()
            if pcd is None or not pcd.has_points():
                continue

            # Log point cloud
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            rr.log(
                f"world/objects/obj_{obj.object_id:03d}/pointcloud",
                rr.Points3D(
                    positions=points,
                    colors=colors if colors is not None else obj.get_color(),
                )
            )

            # Log 3D bounding box
            if obj.bounding_box_3d is not None:
                center = obj.bounding_box_3d[:3]
                size = obj.bounding_box_3d[3:]

                rr.log(
                    f"world/objects/obj_{obj.object_id:03d}/bbox",
                    rr.Boxes3D(
                        half_sizes=[size/2],
                        centers=[center],
                        colors=[obj.get_color()],
                    )
                )

            # Log object info card
            info_text = f"""
**Object {obj.object_id}**

**Semantic Label**: {obj.label}

**3D Geometry**:
- Points: {obj.get_num_points():,}
- Bounding Box: {obj.bounding_box_3d[:3] if obj.bounding_box_3d is not None else 'N/A'}
- Size: {obj.bounding_box_3d[3:] if obj.bounding_box_3d is not None else 'N/A'}

**Tracking**:
- Observations: {obj.num_observations}
- First seen: Frame {obj.first_seen_frame}
- Last seen: Frame {obj.last_seen_frame}
- Status: {obj.status.value}

**Confidence**:
- Overall: {obj.confidence:.2f}
- Tracking: {obj.tracking_confidence:.2f}
- Semantic: {obj.semantic_confidence:.2f}

**Caption History** ({len(obj.caption_history)} captions):
"""
            for i, caption in enumerate(obj.caption_history, 1):
                info_text += f"{i}. {caption}\n"

            rr.log(
                f"world/objects/obj_{obj.object_id:03d}/info",
                rr.TextDocument(info_text, media_type=rr.MediaType.MARKDOWN)
            )

        print(f"✓ Logged {len(objects_to_show)} semantic 3D objects")
        print()

    # Save semantic objects to JSON
    if cfg.get('save_objects', True):  # Default to True
        json_path = cfg.get('objects_json_path', 'outputs/semantic_objects.json')
        data_dir = cfg.get('objects_data_dir', 'outputs/semantic_objects_data')

        print(f"💾 Exporting semantic objects to JSON...")
        json_file, num_exported = manager.export_to_json(
            json_path=json_path,
            output_dir=data_dir,
            min_observations=track_cfg.min_observations,
            min_points=recon_cfg.min_points,
        )

        print(f"✓ Saved {num_exported} objects to:")
        print(f"   JSON: {json_file}")
        print(f"   Data: {data_dir}/")
        print()

    print("=" * 70)
    print("✅ Demo 4.2 completed successfully!")
    print("=" * 70)
    print()
    print(f"📊 Final Results:")
    print(f"   Frames processed: {len(dataset)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Per-frame time: {total_time/len(dataset):.3f}s")
    print(f"   Semantic objects: {len(confirmed)}")
    print()
    print("🎯 Complete Pipeline:")
    print("   SAM → CLIP → Tracking → 3D Reconstruction → Labeling")
    print()


if __name__ == "__main__":
    main()
