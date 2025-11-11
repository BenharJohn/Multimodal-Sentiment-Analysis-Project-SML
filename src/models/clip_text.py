"""
CLIP Text Encoder Wrapper
Extracts token-level embeddings and pooled text representation from CLIP text tower.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPTextEncoder(nn.Module):
    """Wrapper for CLIP text encoder that exposes both token and pooled embeddings."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
        extract_layer: int = -1
    ):
        """
        Initialize CLIP text encoder.

        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze the encoder weights
            extract_layer: Which transformer layer to extract features from (-1 for last)
        """
        super().__init__()

        # Load pretrained CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.text_model = self.clip_model.text_model
        self.text_projection = self.clip_model.text_projection

        # Configuration
        self.hidden_dim = self.text_model.config.hidden_size
        self.projection_dim = self.clip_model.config.projection_dim
        self.extract_layer = extract_layer

        # Freeze if specified
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the text encoder."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the text encoder."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_block(self):
        """Unfreeze only the last transformer block for fine-tuning."""
        # Freeze everything first
        self.freeze()

        # Unfreeze the last encoder layer
        for param in self.text_model.encoder.layers[-1].parameters():
            param.requires_grad = True

        # Unfreeze final layer norm and projection
        for param in self.text_model.final_layer_norm.parameters():
            param.requires_grad = True
        if self.text_projection is not None:
            for param in self.text_projection.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through text encoder.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            dict with:
                - text_tokens: [batch_size, seq_len, hidden_dim] token embeddings
                - text_pooled: [batch_size, projection_dim] pooled text embedding
                - text_features: [batch_size, hidden_dim] pre-projection pooled features
        """
        # Get text encoder outputs
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract token-level features
        # Shape: [batch_size, seq_len, hidden_dim]
        text_tokens = text_outputs.last_hidden_state

        # Extract pooled features (from [EOS] token)
        # Shape: [batch_size, hidden_dim]
        pooled_output = text_outputs.pooler_output  # Already extracts [EOS] token

        # Project to common space
        # Shape: [batch_size, projection_dim]
        if self.text_projection is not None:
            text_pooled = self.text_projection(pooled_output)
        else:
            text_pooled = pooled_output

        return {
            "text_tokens": text_tokens,          # [B, T, D]
            "text_pooled": text_pooled,          # [B, projection_dim]
            "text_features": pooled_output,      # [B, hidden_dim]
            "attention_mask": attention_mask     # [B, T]
        }


class CLIPTextProcessor:
    """Wrapper for CLIP text preprocessing."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 77):
        """Initialize text processor.

        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length (CLIP default is 77)
        """
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.max_length = max_length

    def __call__(self, texts, device='cpu'):
        """
        Process text inputs.

        Args:
            texts: List of text strings or single string
            device: Target device for tensors

        Returns:
            dict with input_ids and attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device)
        }
