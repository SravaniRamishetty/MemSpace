#!/usr/bin/env python3
"""
Demo 1.1: Basic Setup and Environment Verification

This demo verifies:
- PyTorch and CUDA installation
- Hydra configuration loading
- Rerun visualization initialization
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import rerun as rr


@hydra.main(config_path="../memspace/configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Main demo function"""

    print("=" * 60)
    print("MemSpace Demo 1.1: Basic Setup and Environment Verification")
    print("=" * 60)
    print()

    # Display loaded configuration
    print("📋 Loaded Configuration:")
    print("-" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("-" * 60)
    print()

    # Check PyTorch installation
    print("🔧 PyTorch Information:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  Device count: {torch.cuda.device_count()}")
    else:
        print("  ⚠️  No CUDA device found - will use CPU")
    print()

    # Initialize Rerun visualization
    if cfg.use_rerun:
        print("📊 Initializing Rerun visualization...")
        rr.init("memspace/demo_1_1", spawn=True)

        # Set up coordinate system (right-hand, Z-up)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)

        # Log welcome message
        welcome_text = """
# MemSpace Initialized ✓

Environment setup successful!

## System Information
- PyTorch: {}
- CUDA: {}
- Device: {}

## Configuration
- Experiment: {}
- Data root: {}
- Output dir: {}

## Next Steps
1. Run Demo 1.2 to load and visualize RGB-D data
2. Continue with Phase 2 for object detection

---
*Demo 1.1 - Basic Setup Complete*
        """.format(
            torch.__version__,
            "Available" if torch.cuda.is_available() else "Not available",
            cfg.device,
            cfg.exp_name,
            cfg.data_root,
            cfg.output_dir
        )

        rr.log("world/status", rr.TextDocument(welcome_text, media_type=rr.MediaType.MARKDOWN))

        # Log a simple 3D coordinate frame
        rr.log("world/origin", rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["X", "Y", "Z"]
        ))

        print("  ✓ Rerun visualization initialized")
        print("  👀 Check the Rerun viewer window")
        print()

    # Summary
    print("✅ All systems operational!")
    print("=" * 60)
    print()
    print("Next: Run demo_1_2_data_pipeline.py to load RGB-D data")
    print()


if __name__ == "__main__":
    main()
