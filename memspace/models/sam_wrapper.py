"""
SAM (Segment Anything Model) wrapper
Uses Ultralytics SAM for simplicity (same as ConceptGraphs)
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from ultralytics import SAM
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. SAM will not be available.")


class SAMWrapper:
    """
    Wrapper for Segment Anything Model using Ultralytics

    Supports:
    - MobileSAM (fast, lightweight)
    - SAM-L (large, high quality)
    - Automatic mask generation
    - Prompted segmentation with bounding boxes
    """

    def __init__(
        self,
        model_type: str = "mobile_sam",
        device: str = "cuda",
    ):
        """
        Args:
            model_type: One of 'mobile_sam', 'sam_l', 'sam_b'
            device: Device to run model on ('cuda' or 'cpu')
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

        self.model_type = model_type
        self.device = device

        # Map model types to Ultralytics model names
        model_map = {
            'mobile_sam': 'mobile_sam.pt',
            'sam_l': 'sam_l.pt',
            'sam_b': 'sam_b.pt',
        }

        if model_type not in model_map:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Choose from: {list(model_map.keys())}"
            )

        model_name = model_map[model_type]

        print(f"Loading SAM model: {model_name}")
        self.model = SAM(model_name)

        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            self.model.to(device)
        elif device == 'cuda':
            print("Warning: CUDA not available, using CPU")
            self.device = 'cpu'

        print(f"✓ SAM model loaded on {self.device}")

    def segment_image(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image with SAM

        Args:
            image: RGB image (H, W, 3), uint8, range [0, 255]
            bboxes: Bounding boxes (N, 4) in xyxy format, optional
            points: Point prompts (N, 2) in xy format, optional
            labels: Point labels (N,) 1=foreground, 0=background, optional

        Returns:
            masks: Binary masks (N, H, W), bool
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
        """
        # Ultralytics SAM expects image as numpy array or path
        # and returns a Results object

        kwargs = {}
        if bboxes is not None:
            # Convert numpy to tensor if needed
            if isinstance(bboxes, np.ndarray):
                bboxes = torch.from_numpy(bboxes).float()
            kwargs['bboxes'] = bboxes

        if points is not None:
            kwargs['points'] = points
            if labels is not None:
                kwargs['labels'] = labels

        # Run prediction
        results = self.model.predict(
            image,
            verbose=False,
            **kwargs
        )

        # Extract results
        result = results[0]  # First (and only) image result

        if hasattr(result, 'masks') and result.masks is not None:
            # Get masks as numpy arrays
            masks = result.masks.data.cpu().numpy()  # (N, H, W)

            # Get bounding boxes
            if hasattr(result.masks, 'boxes'):
                boxes = result.masks.boxes.xyxy.cpu().numpy()  # (N, 4)
            else:
                boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)

            # Get confidence scores
            if hasattr(result.boxes, 'conf'):
                scores = result.boxes.conf.cpu().numpy()  # (N,)
            else:
                # If no confidence scores, set to 1.0
                scores = np.ones(len(masks))

            return masks, boxes, scores
        else:
            # No masks found, return empty arrays
            return np.array([]), np.array([]), np.array([])

    def generate_masks(
        self,
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks automatically (no prompts)

        Args:
            image: RGB image (H, W, 3), uint8
            points_per_side: Number of points per side for grid
            pred_iou_thresh: IoU threshold for filtering
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask area in pixels

        Returns:
            masks: Binary masks (N, H, W)
            boxes: Bounding boxes (N, 4)
            scores: Confidence scores (N,)
        """
        # For automatic mask generation, just run without prompts
        # Ultralytics SAM will generate masks automatically
        results = self.model.predict(
            image,
            verbose=False,
        )

        result = results[0]

        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            if hasattr(result.boxes, 'conf'):
                scores = result.boxes.conf.cpu().numpy()
            else:
                scores = np.ones(len(masks))

            # Filter by area
            areas = np.array([mask.sum() for mask in masks])
            valid_idx = areas >= min_mask_region_area

            masks = masks[valid_idx]
            boxes = boxes[valid_idx]
            scores = scores[valid_idx]

            return masks, boxes, scores
        else:
            return np.array([]), np.array([]), np.array([])

    def __call__(self, image: np.ndarray, **kwargs):
        """Convenience method - same as segment_image"""
        return self.segment_image(image, **kwargs)
