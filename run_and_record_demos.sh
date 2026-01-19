#!/bin/bash
# Run all demos on full room0 dataset and save RRD recordings

set -e

OUTPUT_DIR="outputs/videos"
mkdir -p $OUTPUT_DIR

echo "===== Running MemSpace Demos with RRD Recording ====="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Activate virtual environment
source venv/bin/activate

# Demo 2.1: SAM Segmentation (40 frames, stride 50)
echo "===== Demo 2.1: SAM Segmentation ====="
echo "Frames: 40 (stride 50, every 50th frame from 2000 total)"
python demos/demo_2_1_sam_segmentation.py \
    dataset.stride=50 \
    dataset.max_frames=40 \
    rerun_save_path=$OUTPUT_DIR/demo_2_1_sam_segmentation.rrd \
    rerun_spawn=false \
    2>&1 | tee $OUTPUT_DIR/demo_2_1.log
echo "✓ Demo 2.1 completed"
echo "  Recording: $OUTPUT_DIR/demo_2_1_sam_segmentation.rrd"
echo "  Log: $OUTPUT_DIR/demo_2_1.log"
echo ""

# Demo 2.2: CLIP Embeddings (40 frames, stride 50)
echo "===== Demo 2.2: CLIP Embeddings ====="
echo "Frames: 40 (stride 50)"
python demos/demo_2_2_clip_embeddings.py \
    dataset.stride=50 \
    dataset.max_frames=40 \
    rerun_save_path=$OUTPUT_DIR/demo_2_2_clip_embeddings.rrd \
    rerun_spawn=false \
    2>&1 | tee $OUTPUT_DIR/demo_2_2.log
echo "✓ Demo 2.2 completed"
echo "  Recording: $OUTPUT_DIR/demo_2_2_clip_embeddings.rrd"
echo "  Log: $OUTPUT_DIR/demo_2_2.log"
echo ""

# Demo 2.3: Object Tracking (100 frames, stride 20)
echo "===== Demo 2.3: Object Tracking ====="
echo "Frames: 100 (stride 20, better for tracking)"
python demos/demo_2_3_object_tracking.py \
    dataset.stride=20 \
    dataset.max_frames=100 \
    rerun_save_path=$OUTPUT_DIR/demo_2_3_object_tracking.rrd \
    rerun_spawn=false \
    2>&1 | tee $OUTPUT_DIR/demo_2_3.log
echo "✓ Demo 2.3 completed"
echo "  Recording: $OUTPUT_DIR/demo_2_3_object_tracking.rrd"
echo "  Log: $OUTPUT_DIR/demo_2_3.log"
echo ""

echo "===== All Demos Completed ====="
echo ""
echo "RRD recordings saved to $OUTPUT_DIR/"
ls -lh $OUTPUT_DIR/*.rrd
echo ""
echo "To view recordings:"
echo "  rerun $OUTPUT_DIR/demo_2_1_sam_segmentation.rrd"
echo "  rerun $OUTPUT_DIR/demo_2_2_clip_embeddings.rrd"
echo "  rerun $OUTPUT_DIR/demo_2_3_object_tracking.rrd"
echo ""
echo "To run with live viewer (no recording):"
echo "  python demos/demo_2_1_sam_segmentation.py dataset.stride=50 dataset.max_frames=40"
