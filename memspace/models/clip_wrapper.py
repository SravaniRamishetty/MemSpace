"""
CLIP (Contrastive Language-Image Pre-training) wrapper
Uses OpenCLIP for extracting visual and text embeddings
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Union

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("Warning: open_clip_torch not installed. CLIP will not be available.")


class CLIPWrapper:
    """
    Wrapper for CLIP model using OpenCLIP

    Supports:
    - Image encoding (crop-based features)
    - Text encoding (for queries and object descriptions)
    - Batch processing for efficiency
    - Multiple CLIP architectures (ViT-B-32, ViT-H-14, etc.)
    """

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: str = "cuda",
    ):
        """
        Args:
            model_name: CLIP architecture (e.g., 'ViT-H-14', 'ViT-B-32')
            pretrained: Pretrained weights (e.g., 'laion2b_s32b_b79k', 'openai')
            device: Device to run model on ('cuda' or 'cpu')
        """
        if not OPENCLIP_AVAILABLE:
            raise ImportError(
                "open_clip_torch not installed. Install with: pip install open-clip-torch"
            )

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device

        print(f"Loading CLIP model: {model_name} ({pretrained})")

        # Create model and preprocessing transforms
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.to(device)
        elif device == 'cuda':
            print("Warning: CUDA not available, using CPU")
            self.device = 'cpu'

        # Set to eval mode
        self.model.eval()

        print(f"✓ CLIP model loaded on {self.device}")

    def encode_image(
        self,
        image: Union[np.ndarray, Image.Image],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a single image to CLIP embedding

        Args:
            image: RGB image (H, W, 3) as numpy array or PIL Image
            normalize: Whether to L2-normalize the embedding

        Returns:
            embedding: CLIP embedding (D,) where D is embedding dimension
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Preprocess and add batch dimension
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Encode
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)

            # Normalize
            if normalize:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().squeeze()

    def encode_images_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple images to CLIP embeddings (batched for efficiency)

        Args:
            images: List of RGB images
            normalize: Whether to L2-normalize the embeddings

        Returns:
            embeddings: CLIP embeddings (N, D)
        """
        # Convert all to PIL and preprocess
        image_tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            image_tensors.append(self.preprocess(img))

        # Stack into batch
        image_batch = torch.stack(image_tensors).to(self.device)

        # Encode
        with torch.no_grad():
            embeddings = self.model.encode_image(image_batch)

            # Normalize
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy()

    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text to CLIP embedding

        Args:
            text: Single text string or list of strings
            normalize: Whether to L2-normalize the embedding

        Returns:
            embedding: CLIP embedding (D,) for single text, (N, D) for list
        """
        # Convert single string to list
        if isinstance(text, str):
            text = [text]
            squeeze_output = True
        else:
            squeeze_output = False

        # Tokenize
        text_tokens = self.tokenizer(text).to(self.device)

        # Encode
        with torch.no_grad():
            embeddings = self.model.encode_text(text_tokens)

            # Normalize
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        embeddings = embeddings.cpu().numpy()

        # Squeeze if single text
        if squeeze_output:
            embeddings = embeddings.squeeze()

        return embeddings

    def compute_similarity(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text features

        Args:
            image_features: Image embeddings (N, D) or (D,)
            text_features: Text embeddings (M, D) or (D,)

        Returns:
            similarity: Cosine similarity scores (N, M) or (N,) or (M,) or scalar
        """
        # Ensure 2D
        if image_features.ndim == 1:
            image_features = image_features[np.newaxis, :]
        if text_features.ndim == 1:
            text_features = text_features[np.newaxis, :]

        # Compute cosine similarity (dot product of normalized vectors)
        similarity = image_features @ text_features.T

        return similarity.squeeze()

    def extract_mask_features(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        padding: int = 20,
        batch_size: int = 32,
    ) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Extract CLIP features for masks using bounding box crops

        Args:
            image: RGB image (H, W, 3), uint8
            boxes: Bounding boxes (N, 4) in xyxy format
            padding: Padding around bounding box (pixels)
            batch_size: Batch size for processing

        Returns:
            crops: List of cropped PIL Images
            features: CLIP embeddings (N, D)
        """
        # Convert to PIL
        pil_image = Image.fromarray(image)
        image_width, image_height = pil_image.size

        crops = []

        # Extract crops with padding
        for box in boxes:
            x_min, y_min, x_max, y_max = box

            # Adjust padding to avoid going beyond image borders
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Apply padding
            x_min = max(0, x_min - left_padding)
            y_min = max(0, y_min - top_padding)
            x_max = min(image_width, x_max + right_padding)
            y_max = min(image_height, y_max + bottom_padding)

            # Crop
            cropped = pil_image.crop((x_min, y_min, x_max, y_max))
            crops.append(cropped)

        # Extract features in batches
        all_features = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_features = self.encode_images_batch(batch_crops)
            all_features.append(batch_features)

        # Concatenate
        features = np.concatenate(all_features, axis=0)

        return crops, features

    def get_embedding_dim(self) -> int:
        """Get the dimension of CLIP embeddings"""
        # Encode a dummy image to get dimension
        dummy = Image.new('RGB', (224, 224))
        embedding = self.encode_image(dummy)
        return embedding.shape[0]
