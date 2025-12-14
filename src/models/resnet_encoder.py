"""
ResNet Image Encoder Wrapper
Extracts visual features from ResNet for image representation.
Used as auxiliary encoder alongside CLIP for CDAN 2025 architecture.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class ResNetImageEncoder(nn.Module):
    """
    Wrapper for ResNet image encoder that extracts visual features.

    Used alongside CLIP vision encoder for dual-encoder fusion as per CDAN 2025 paper.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        freeze: bool = True,
        pretrained: bool = True
    ):
        """
        Initialize ResNet image encoder.

        Args:
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            freeze: Whether to freeze the encoder weights
            pretrained: Whether to load ImageNet pretrained weights
        """
        super().__init__()

        # Load pretrained ResNet model
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet34(weights=weights)
            self.feature_dim = 512
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.resnet = models.resnet50(weights=weights)
            self.feature_dim = 2048
        elif model_name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            self.resnet = models.resnet101(weights=weights)
            self.feature_dim = 2048
        elif model_name == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            self.resnet = models.resnet152(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")

        # Remove the final classification layer
        # Keep everything up to avgpool
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Store config
        self.hidden_dim = self.feature_dim
        self.model_name = model_name

        # Freeze if specified
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the ResNet encoder."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the ResNet encoder."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_block(self):
        """Unfreeze only the last residual block (layer4) for fine-tuning."""
        # Freeze everything first
        self.freeze()

        # Find and unfreeze layer4 (last residual block)
        for name, param in self.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor):
        """
        Forward pass through ResNet encoder.

        Args:
            pixel_values: [batch_size, channels, height, width] preprocessed images
                         Expected to be normalized for ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         CLIP preprocessing is compatible with ResNet (224x224, normalized)

        Returns:
            dict with:
                - features: [batch_size, feature_dim] global pooled features
                - spatial: [batch_size, feature_dim, h, w] spatial features (before pooling)
        """
        # Extract features through all layers except final FC
        # Output before avgpool: [B, feature_dim, 7, 7] for 224x224 input
        spatial_features = None

        # Manual forward to get spatial features
        x = pixel_values
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Store features before final avgpool (layer index -1)
            if i == len(self.features) - 2:
                spatial_features = x  # [B, feature_dim, 7, 7]

        # x after avgpool is [B, feature_dim, 1, 1]
        features = x.squeeze(-1).squeeze(-1)  # [B, feature_dim]

        return {
            "features": features,         # [B, 2048] for resnet50
            "spatial": spatial_features,  # [B, 2048, 7, 7] for resnet50
        }


class ResNetImageProcessor:
    """
    Wrapper for ResNet image preprocessing.

    Note: CLIP's preprocessing (224x224, normalized) is already compatible with ResNet.
    This processor provides an alternative if needed.
    """

    def __init__(self, image_size: int = 224):
        """
        Initialize ResNet image processor.

        Args:
            image_size: Target image size (default 224 for ResNet)
        """
        self.image_size = image_size

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, images, device='cpu'):
        """
        Process image inputs for ResNet.

        Args:
            images: List of PIL Images, numpy arrays, or single image
            device: Target device for tensors

        Returns:
            dict with pixel_values tensor
        """
        if not isinstance(images, list):
            images = [images]

        # Process each image
        processed = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")

            processed.append(self.transform(img))

        # Stack into batch
        pixel_values = torch.stack(processed)

        return {
            "resnet_pixel_values": pixel_values.to(device)
        }
