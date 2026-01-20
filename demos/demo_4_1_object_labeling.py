#!/usr/bin/env python3
"""
Demo 4.1: Object Labeling with VLM (Vision Language Model)

This demo demonstrates:
- Semantic labeling of tracked objects using VLM (Florence-2 or LLaVA)
- Multi-view caption generation
- Integration of tracking + captioning
- Labeled object visualization in Rerun
- Complete pipeline: SAM → CLIP → Tracking → Captioning

Supports both Florence-2 and LLaVA VLMs.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import rerun as rr

from memspace.dataset import get_dataset
from memspace.models.sam_wrapper import SAMWrapper
from memspace.models.clip_wrapper import CLIPWrapper
from memspace.models.florence_wrapper import FlorenceWrapper
try:
    from memspace.models.llava_wrapper import LLaVAWrapper
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

from memspace.scenegraph.object_tracker import ObjectTracker
from memspace.scenegraph.object_captioning import ObjectCaptioner
from memspace.utils import rerun_utils
from memspace.utils.mask_utils import merge_overlapping_masks


@hydra.main(config_path="../memspace/configs", config_name="demo_4_1", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    # Get VLM type from config
    vlm_type = cfg.get('vlm_type', 'florence').lower()

    print("=" * 70)
    print(f"MemSpace Demo 4.1: Object Labeling with {vlm_type.upper()} VLM")
    print("=" * 70)
    print()

    # Initialize Rerun
    if cfg.use_rerun:
        spawn = cfg.get('rerun_spawn', True)
        save_path = cfg.get('rerun_save_path', None)

        if save_path:
            rr.init("memspace/demo_4_1", spawn=spawn)
            rr.save(save_path)
            print(f"✓ Rerun recording to: {save_path}")
            if spawn:
                print("✓ Rerun viewer spawned")
        else:
            rr.init("memspace/demo_4_1", spawn=spawn)
            if spawn:
                print("✓ Rerun viewer spawned")
            else:
                print("✓ Rerun initialized (no viewer)")

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    else:
        print("⚠  Rerun visualization disabled in config")
        return

    print()

    # Initialize SAM model
    print(f"🤖 Initializing SAM model: {cfg.model.sam.model_type}")
    sam_model = SAMWrapper(
        model_type=cfg.model.sam.model_type,
        device=cfg.device,
    )
    print()

    # Initialize CLIP model
    print(f"🤖 Initializing CLIP model: {cfg.model.clip.model_name}")
    clip_model = CLIPWrapper(
        model_name=cfg.model.clip.model_name,
        pretrained=cfg.model.clip.pretrained,
        device=cfg.device,
    )
    print()

    # Initialize VLM (Florence-2 or LLaVA)
    if vlm_type == "florence":
        florence_cfg = cfg.model.florence
        print(f"🤖 Initializing Florence-2 model: {florence_cfg.model_name}")
        vlm_model = FlorenceWrapper(
            model_name=florence_cfg.model_name,
            device=florence_cfg.device,
            torch_dtype=florence_cfg.get('torch_dtype', 'float16'),
        )
        caption_task = cfg.captioning.caption_task
        caption_query = None
        print()
    elif vlm_type == "llava":
        if not LLAVA_AVAILABLE:
            print("❌ Error: LLaVA is not available. Please install LLaVA and set LLAVA_PYTHON_PATH.")
            print("   Falling back to Florence-2...")
            vlm_type = "florence"
            florence_cfg = cfg.model.florence
            print(f"🤖 Initializing Florence-2 model: {florence_cfg.model_name}")
            vlm_model = FlorenceWrapper(
                model_name=florence_cfg.model_name,
                device=florence_cfg.device,
                torch_dtype=florence_cfg.get('torch_dtype', 'float16'),
            )
            caption_task = cfg.captioning.caption_task
            caption_query = None
            print()
        else:
            llava_cfg = cfg.model.llava
            print(f"🤖 Initializing LLaVA model")
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
        print(f"❌ Error: Unknown VLM type '{vlm_type}'. Use 'florence' or 'llava'")
        return

    # Initialize Object Tracker
    track_cfg = cfg.tracking
    print(f"🎯 Initializing Object Tracker")
    tracker = ObjectTracker(
        sim_threshold=track_cfg.sim_threshold,
        spatial_weight=track_cfg.spatial_weight,
        clip_weight=track_cfg.clip_weight,
        max_missing_frames=track_cfg.max_missing_frames,
        min_observations=track_cfg.min_observations,
    )
    print()

    # Initialize Object Captioner
    caption_cfg = cfg.captioning
    print(f"💬 Initializing Object Captioner ({vlm_type.upper()})")
    if vlm_type == "florence":
        print(f"   Caption task: {caption_task}")
    else:
        print(f"   Caption query: {caption_query}")
    print(f"   Caption interval: every {caption_cfg.caption_interval} frames")
    captioner = ObjectCaptioner(
        vlm_model=vlm_model,
        vlm_type=vlm_type,
        caption_task=caption_task,
        caption_query=caption_query,
        max_captions_per_object=caption_cfg.max_captions_per_object,
        min_caption_length=caption_cfg.min_caption_length,
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

    print("🎬 Processing frames with tracking + captioning...")
    print(f"   Frames: {len(dataset)}")
    print(f"   Stride: {cfg.dataset.stride}")
    print()

    start_time = time.time()

    for frame_idx in range(len(dataset)):
        if frame_idx % 5 == 0 or frame_idx == len(dataset) - 1:
            print(f"Frame {frame_idx+1}/{len(dataset)}")

        # Load frame
        color, depth, intrinsics, pose = dataset[frame_idx]
        color_np = color.cpu().numpy().astype(np.uint8)
        depth_np = depth.cpu().numpy().astype(np.float32)

        # Run SAM segmentation
        masks, boxes, scores = sam_model.generate_masks(
            color_np,
            min_mask_region_area=seg_cfg.min_mask_area,
        )

        if len(masks) == 0:
            # Update tracker with empty detections
            tracker.update(frame_idx, np.array([]), np.array([]), np.array([]), np.array([]))
            continue

        # Merge overlapping masks
        if seg_cfg.get('merge_masks', True):
            masks, boxes, scores = merge_overlapping_masks(
                masks, boxes, scores,
                iou_threshold=seg_cfg.get('merge_iou_threshold', 0.5),
                containment_threshold=seg_cfg.get('merge_containment_threshold', 0.85),
            )

        # Limit number of masks
        if len(masks) > seg_cfg.max_masks_per_frame:
            sorted_idx = np.argsort(scores)[::-1][:seg_cfg.max_masks_per_frame]
            masks = masks[sorted_idx]
            boxes = boxes[sorted_idx]
            scores = scores[sorted_idx]

        # Extract CLIP features
        crops, features = clip_model.extract_mask_features(
            color_np,
            boxes,
            padding=clip_cfg.padding,
            batch_size=clip_cfg.batch_size,
        )

        # Update tracker
        object_ids, match_types = tracker.update(
            frame_idx=frame_idx,
            masks=masks,
            bboxes=boxes,
            scores=scores,
            clip_features=features,
        )

        # Generate captions for confirmed objects
        # Only caption every N frames to save computation
        if frame_idx % caption_cfg.caption_interval == 0:
            # Filter for confirmed objects (active with sufficient observations)
            confirmed_indices = []
            confirmed_ids = []
            confirmed_boxes = []

            for i, obj_id in enumerate(object_ids):
                # Find object in list
                obj = next((o for o in tracker.objects if o.object_id == obj_id), None)
                if obj and obj.num_observations >= track_cfg.min_observations:
                    confirmed_indices.append(i)
                    confirmed_ids.append(obj_id)
                    confirmed_boxes.append(boxes[i])

            if len(confirmed_ids) > 0:
                confirmed_boxes_np = np.array(confirmed_boxes)

                # Generate captions
                frame_captions = captioner.caption_frame_objects(
                    image=color_np,
                    object_ids=confirmed_ids,
                    bboxes=confirmed_boxes_np,
                    padding=caption_cfg.crop_padding,
                )

                # Log captions to Rerun
                for obj_id, caption in frame_captions.items():
                    rr.log(
                        f"world/captions/obj_{obj_id:03d}/frame_{frame_idx:03d}",
                        rr.TextDocument(f"**Frame {frame_idx}**: {caption}", media_type=rr.MediaType.MARKDOWN)
                    )

        # Log camera trajectory every 5 frames
        if frame_idx % 5 == 0:
            rerun_utils.log_camera_pose(
                f"world/camera_trajectory/frame_{frame_idx:03d}",
                pose,
                intrinsics,
                width=640,
                height=480,
            )

    total_time = time.time() - start_time

    print()
    print(f"✓ Processed {len(dataset)} frames in {total_time:.2f}s ({total_time/len(dataset):.3f}s per frame)")
    print()

    # Get final statistics
    tracker_stats = tracker.get_statistics()
    caption_stats = captioner.get_statistics()

    print(f"📊 Tracking Statistics:")
    print(f"   Total objects tracked: {tracker_stats['total_objects']}")
    print(f"   Confirmed objects: {tracker_stats['confirmed_objects']}")
    print(f"   Active objects: {tracker_stats['active_objects']}")
    print()

    print(f"💬 Captioning Statistics:")
    print(f"   Objects with captions: {caption_stats['num_captioned_objects']}")
    print(f"   Total captions generated: {caption_stats['total_captions']}")
    print(f"   Avg captions per object: {caption_stats['avg_captions_per_object']:.1f}")
    print()

    # Generate labels for all objects
    print("🏷️  Generating object labels...")
    all_labels = captioner.get_all_labels()
    print(f"✓ Generated {len(all_labels)} object labels")
    print()

    # Visualize top labeled objects
    vis_cfg = cfg.visualization
    if vis_cfg.get('show_labels', True):
        print(f"🏷️  Logging object labels to Rerun...")

        # Get confirmed objects sorted by observation count
        confirmed_objects = [
            (obj.object_id, obj) for obj in tracker.objects
            if obj.num_observations >= track_cfg.min_observations
        ]
        confirmed_objects.sort(key=lambda x: x[1].num_observations, reverse=True)

        objects_to_show = confirmed_objects[:vis_cfg.max_objects_to_show]

        for obj_id, obj in objects_to_show:
            # Get label
            label = captioner.get_object_label(obj_id)
            if not label:
                continue

            # Get all captions for this object
            caption_history = captioner.get_object_caption_history(obj_id)

            # Create label card
            label_text = f"""
**Object {obj_id}**

**Label**: {label}

**Tracking Info**:
- Observations: {obj.num_observations}
- First seen: Frame {obj.first_seen_frame}
- Last seen: Frame {obj.last_seen_frame}
- Status: {obj.status.value}

**Caption History** ({len(caption_history)} captions):
"""
            for i, caption in enumerate(caption_history, 1):
                label_text += f"{i}. {caption}\n"

            rr.log(
                f"world/labels/obj_{obj_id:03d}",
                rr.TextDocument(label_text, media_type=rr.MediaType.MARKDOWN)
            )

        print(f"✓ Logged {len(objects_to_show)} object labels")
        print()

    # Log completion summary
    completion_text = f"""
# Demo 4.1 Complete! ✓

Successfully tracked and labeled {len(all_labels)} objects using Florence-2 VLM.

## Pipeline Summary:
1. **SAM Segmentation**: Detected objects in each frame
2. **CLIP Embeddings**: Extracted visual features for tracking
3. **Object Tracking**: Associated detections across frames
4. **Florence-2 Captioning**: Generated semantic descriptions

## Tracking Results:
- **Total objects tracked:** {tracker_stats['total_objects']}
- **Confirmed objects:** {tracker_stats['confirmed_objects']}
- **Total detections:** {tracker_stats['total_detections']}
- **Match rate:** {tracker_stats['total_matches']/tracker_stats['total_detections']*100:.1f}%

## Captioning Results:
- **Objects captioned:** {caption_stats['num_captioned_objects']}
- **Total captions:** {caption_stats['total_captions']}
- **Objects labeled:** {len(all_labels)}

## VLM Model:
- **Model:** {cfg.model.florence.model_name if vlm_type == 'florence' else 'LLaVA'}
- **Task:** {caption_cfg.caption_task if vlm_type == 'florence' else 'Image Captioning'}
- **Caption interval:** Every {caption_cfg.caption_interval} frames

## Top Labeled Objects:
"""

    # Show top 10 objects by observation count
    top_objects = sorted(
        [(obj.object_id, obj) for obj in tracker.objects if obj.num_observations >= track_cfg.min_observations],
        key=lambda x: x[1].num_observations,
        reverse=True
    )[:10]

    for i, (obj_id, obj) in enumerate(top_objects, 1):
        label = captioner.get_object_label(obj_id)
        if label:
            completion_text += f"""
{i}. **Object {obj_id}**: "{label}" ({obj.num_observations} observations)
"""

    completion_text += f"""
## Performance:
- **Total time:** {total_time:.2f}s
- **Per-frame time:** {total_time/len(dataset):.3f}s
- **Frames processed:** {len(dataset)}

## What you're seeing:
- **Object captions**: Per-frame captions at 'world/captions/obj_XXX/frame_YYY'
- **Object labels**: Consolidated labels at 'world/labels/obj_XXX'
- **Camera trajectory**: Camera poses during capture

## Key Features:
- VLM-based semantic understanding (Florence-2)
- Multi-view caption aggregation
- Automatic object labeling
- Zero-shot object recognition

## Next Steps:
- Object-level 3D reconstruction with semantic labels (Phase 4.2)
- Scene graph construction (Phase 4.3)
- Natural language queries (Phase 5)

---
*Dataset: {cfg.dataset.dataset_path}*
*Frames: {len(dataset)}*
*Model: {cfg.model.florence.model_name}*
*Objects labeled: {len(all_labels)}*
    """

    rr.log("world/summary", rr.TextDocument(completion_text, media_type=rr.MediaType.MARKDOWN))

    print("=" * 70)
    print("✅ Demo 4.1 completed successfully!")
    print("=" * 70)
    print()
    print(f"📊 Final Results:")
    print(f"   Frames processed: {len(dataset)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Per-frame time: {total_time/len(dataset):.3f}s")
    print(f"   Objects tracked: {tracker_stats['total_objects']}")
    print(f"   Objects labeled: {len(all_labels)}")
    print()
    print("📊 Check the Rerun viewer:")
    print("   - Object labels at 'world/labels/obj_XXX'")
    print("   - Caption history at 'world/captions/obj_XXX'")
    print("   - Camera trajectory at 'world/camera_trajectory'")
    print()
    print("🎯 Complete pipeline demonstrated:")
    print("   SAM → CLIP → Tracking → Florence-2 Captioning")
    print()


if __name__ == "__main__":
    main()
