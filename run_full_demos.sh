#!/bin/bash
# Run all demos on full room0 dataset and save recordings

set -e

OUTPUT_DIR="outputs/videos"
mkdir -p $OUTPUT_DIR

echo "===== Running MemSpace Demos on Full Room0 Dataset ====="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Activate virtual environment
source venv/bin/activate

# Demo 2.1: SAM Segmentation (40 frames, stride 50)
echo "===== Demo 2.1: SAM Segmentation ====="
echo "Frames: 40 (stride 50, total 2000 frames in dataset)"
python demos/demo_2_1_sam_segmentation.py \
    dataset.stride=50 \
    dataset.max_frames=40 \
    use_rerun=false \
    > $OUTPUT_DIR/demo_2_1.log 2>&1
echo "✓ Demo 2.1 completed. Log saved to $OUTPUT_DIR/demo_2_1.log"
echo ""

# Demo 2.2: CLIP Embeddings (40 frames, stride 50)
echo "===== Demo 2.2: CLIP Embeddings ====="
echo "Frames: 40 (stride 50)"
python demos/demo_2_2_clip_embeddings.py \
    dataset.stride=50 \
    dataset.max_frames=40 \
    use_rerun=false \
    > $OUTPUT_DIR/demo_2_2.log 2>&1
echo "✓ Demo 2.2 completed. Log saved to $OUTPUT_DIR/demo_2_2.log"
echo ""

# Demo 2.3: Object Tracking (100 frames, stride 20)
echo "===== Demo 2.3: Object Tracking ====="
echo "Frames: 100 (stride 20, better for tracking)"
python demos/demo_2_3_object_tracking.py \
    dataset.stride=20 \
    dataset.max_frames=100 \
    use_rerun=false \
    > $OUTPUT_DIR/demo_2_3.log 2>&1
echo "✓ Demo 2.3 completed. Log saved to $OUTPUT_DIR/demo_2_3.log"
echo ""

echo "===== All Demos Completed ====="
echo "Logs saved to $OUTPUT_DIR/"
echo ""
echo "To view with Rerun:"
echo "  - Run demos interactively with use_rerun=true"
echo "  - Use 'rerun' command line tool to view saved RRD files (if generated)"
