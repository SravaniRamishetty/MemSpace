#!/usr/bin/env python3
"""
Generate visual examples of LLaVA spatial reasoning.

Creates annotated images showing the input to LLaVA for different
spatial relationship types (on, in, near).
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# Load scene graph
with open('outputs/scene_graph.json', 'r') as f:
    scene_graph = json.load(f)

# Load semantic objects
with open('outputs/semantic_objects.json', 'r') as f:
    objects_data = json.load(f)

# Create object lookup
objects = {obj['object_id']: obj for obj in objects_data['objects']}

# Group relations by type
relationships_by_type = defaultdict(list)
for rel in scene_graph['relations']:
    if rel.get('metadata', {}).get('method') == 'llava_visual_reasoning':
        relationships_by_type[rel['relation_type']].append(rel)

print(f"Found {len(relationships_by_type)} relationship types from LLaVA visual reasoning")
for rel_type, rels in relationships_by_type.items():
    print(f"  {rel_type}: {len(rels)} relationships")

# Dataset path
from memspace.dataset.replica_dataset import ReplicaDataset
from omegaconf import OmegaConf

# Load dataset config
cfg = OmegaConf.load('memspace/configs/dataset/replica.yaml')
dataset = ReplicaDataset(
    dataset_path=cfg.dataset_path,
    stride=cfg.stride,
    start=0,
    end=1000,
    height=480,
    width=640,
    device='cpu',
)

print(f"\nDataset loaded: {len(dataset)} frames")

def find_common_frame(obj1, obj2):
    """Find frame where both objects are visible."""
    start1 = obj1.get('first_seen_frame', 0)
    end1 = obj1.get('last_seen_frame', 0)
    start2 = obj2.get('first_seen_frame', 0)
    end2 = obj2.get('last_seen_frame', 0)

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start > overlap_end:
        return None

    return (overlap_start + overlap_end) // 2

def create_annotated_image(frame_id, obj1, obj2, output_path, rel_type):
    """Create annotated image with RED and BLUE boxes."""
    # Load frame
    color, _, _, _ = dataset[frame_id]
    if hasattr(color, 'cpu'):
        rgb = color.cpu().numpy().astype(np.uint8)
    else:
        rgb = np.array(color).astype(np.uint8)

    # Convert to BGR for OpenCV
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    # Draw Object 1 (RED)
    bbox1 = obj1.get('current_bbox_2d')
    if bbox1 and len(bbox1) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox1]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(image, "Object 1 (RED)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw Object 2 (BLUE)
    bbox2 = obj2.get('current_bbox_2d')
    if bbox2 and len(bbox2) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(image, "Object 2 (BLUE)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Add relationship label
    h, w = image.shape[:2]
    cv2.putText(image, f"Detected: Object 1 '{rel_type}' Object 2", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save
    cv2.imwrite(str(output_path), image)
    print(f"  Saved: {output_path}")

# Generate examples for each relationship type
output_dir = Path('outputs/llava_visual_examples')
output_dir.mkdir(exist_ok=True)

# Create README
readme_content = """# LLaVA Visual Reasoning Examples

This folder contains example images showing the input to LLaVA for spatial relationship detection.

## Annotation Convention:
- **Object 1 (RED box)**: The subject object
- **Object 2 (BLUE box)**: The reference object
- **Green text**: Detected spatial relationship

## Relationship Types:

"""

for rel_type in ['on', 'in', 'near']:
    print(f"\n=== Generating examples for '{rel_type}' ===")

    if rel_type not in relationships_by_type:
        print(f"  No {rel_type} relationships found from LLaVA")
        continue

    rels = relationships_by_type[rel_type][:2]  # Get first 2 examples

    readme_content += f"### {rel_type.upper()}\n\n"

    for i, rel in enumerate(rels, 1):
        obj1_id = rel['object_a_id']
        obj2_id = rel['object_b_id']

        obj1 = objects.get(obj1_id)
        obj2 = objects.get(obj2_id)

        if not obj1 or not obj2:
            print(f"  Skipping: objects not found")
            continue

        # Find common frame
        frame_id = find_common_frame(obj1, obj2)
        if frame_id is None:
            print(f"  Skipping: no common frame")
            continue

        # Generate image
        output_path = output_dir / f"{rel_type}_example_{i}.png"
        create_annotated_image(frame_id, obj1, obj2, output_path, rel_type)

        # Add to README
        readme_content += f"**Example {i}**: Object {obj1_id} is '{rel_type}' Object {obj2_id}\n"
        readme_content += f"- Confidence: {rel['confidence']:.2f}\n"
        readme_content += f"- Frame: {frame_id}\n"
        readme_content += f"- ![{rel_type} example {i}]({output_path.name})\n\n"

# Save README
with open(output_dir / 'README.md', 'w') as f:
    f.write(readme_content)

print(f"\n✓ Examples generated in {output_dir}")
print(f"✓ README created")
