"""
CLIP Vision Encoder Wrapper
Extracts patch-level embeddings and pooled image representation from CLIP vision tower.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


class CLIPImageEncoder(nn.Module):
    """Wrapper for CLIP vision encoder that exposes both patch and pooled embeddings."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
        extract_layer: int = -1
    ):
        """
        Initialize CLIP vision encoder.

        Args:
            model_name: HuggingFace model identifier or local path
            freeze: Whether to freeze the encoder weights
            extract_layer: Which transformer layer to extract features from (-1 for last)
        """
        super().__init__()

        # Load pretrained CLIP model
        import os
        is_local = os.path.exists(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name, local_files_only=is_local)
        self.vision_model = self.clip_model.vision_model
        self.visual_projection = self.clip_model.visual_projection

        # Configuration
        self.hidden_dim = self.vision_model.config.hidden_size
        self.projection_dim = self.clip_model.config.projection_dim
        self.extract_layer = extract_layer

        # Freeze if specified
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the vision encoder."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the vision encoder."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_block(self):
        """Unfreeze only the last transformer block for fine-tuning."""
        # Freeze everything first
        self.freeze()

        # Unfreeze the last encoder layer
        for param in self.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True

        # Unfreeze post-layernorm and projection
        for param in self.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        if self.visual_projection is not None:
            for param in self.visual_projection.parameters():
                param.requires_grad = True

    def forward(self, pixel_values):
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: [batch_size, channels, height, width] preprocessed images

        Returns:
            dict with:
                - vision_patches: [batch_size, num_patches, hidden_dim] patch embeddings
                - vision_pooled: [batch_size, projection_dim] pooled image embedding
                - vision_features: [batch_size, hidden_dim] pre-projection pooled features
        """
        # Get vision encoder outputs
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract patch-level features
        # Shape: [batch_size, num_patches + 1, hidden_dim]
        # Note: includes [CLS] token at position 0
        vision_patches = vision_outputs.last_hidden_state

        # Extract pooled features (from [CLS] token)
        # Shape: [batch_size, hidden_dim]
        pooled_output = vision_outputs.pooler_output  # [CLS] token

        # Project to common space
        # Shape: [batch_size, projection_dim]
        if self.visual_projection is not None:
            vision_pooled = self.visual_projection(pooled_output)
        else:
            vision_pooled = pooled_output

        return {
            "vision_patches": vision_patches,      # [B, P+1, D] (includes CLS)
            "vision_pooled": vision_pooled,        # [B, projection_dim]
            "vision_features": pooled_output,      # [B, hidden_dim]
        }


class CLIPImageProcessor:
    """Wrapper for CLIP image preprocessing."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize image processor.

        Args:
            model_name: HuggingFace model identifier or local path
        """
        import os
        is_local = os.path.exists(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=is_local)
        # For transformers 4.18.0 compatibility
        self.image_processor = getattr(self.processor, 'image_processor', getattr(self.processor, 'feature_extractor', self.processor))

    def __call__(self, images, device='cpu'):
        """
        Process image inputs.

        Args:
            images: List of PIL Images, numpy arrays, or single image
            device: Target device for tensors

        Returns:
            dict with pixel_values tensor
        """
        if not isinstance(images, list):
            images = [images]

        # Ensure all images are PIL Images
        processed_images = []
        for img in images:
            if isinstance(img, str):
                # Load from path
                img = Image.open(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                # Assume numpy array or tensor
                img = Image.fromarray(img)
            processed_images.append(img)

        # Preprocess
        inputs = self.image_processor(
            images=processed_images,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"].to(device)
        }
