"""
BERT Text Encoder Wrapper
Extracts token-level embeddings and pooled text representation from BERT.
Used as auxiliary encoder alongside CLIP for CDAN 2025 architecture.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BERTTextEncoder(nn.Module):
    """
    Wrapper for BERT text encoder that exposes both token and pooled embeddings.

    Used alongside CLIP text encoder for dual-encoder fusion as per CDAN 2025 paper.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = True,
        pooling_strategy: str = "cls"  # 'cls', 'mean', or 'max'
    ):
        """
        Initialize BERT text encoder.

        Args:
            model_name: HuggingFace model identifier (e.g., 'bert-base-uncased')
            freeze: Whether to freeze the encoder weights
            pooling_strategy: How to pool token embeddings ('cls', 'mean', 'max')
        """
        super().__init__()

        # Load pretrained BERT model
        self.bert_model = BertModel.from_pretrained(model_name)

        # Configuration
        self.hidden_dim = self.bert_model.config.hidden_size  # 768 for bert-base
        self.pooling_strategy = pooling_strategy

        # Freeze if specified
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the BERT encoder."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the BERT encoder."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_block(self, num_layers: int = 2):
        """
        Unfreeze only the last N transformer blocks for fine-tuning.

        Args:
            num_layers: Number of layers from the end to unfreeze
        """
        # Freeze everything first
        self.freeze()

        # Unfreeze the last N encoder layers
        for layer in self.bert_model.encoder.layer[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze pooler
        if self.bert_model.pooler is not None:
            for param in self.bert_model.pooler.parameters():
                param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass through BERT encoder.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            dict with:
                - tokens: [batch_size, seq_len, hidden_dim] token embeddings
                - pooled: [batch_size, hidden_dim] pooled text embedding
        """
        # Get BERT outputs
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Token-level features: [batch_size, seq_len, hidden_dim]
        token_embeddings = outputs.last_hidden_state

        # Pooled features based on strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            pooled = outputs.pooler_output  # [batch_size, hidden_dim]
        elif self.pooling_strategy == "mean":
            # Mean pooling over valid tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = token_embeddings.mean(dim=1)
        elif self.pooling_strategy == "max":
            # Max pooling over valid tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                token_embeddings[mask_expanded == 0] = -1e9
            pooled = torch.max(token_embeddings, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return {
            "tokens": token_embeddings,  # [B, T, 768]
            "pooled": pooled,            # [B, 768]
        }


class BERTTextProcessor:
    """Wrapper for BERT text preprocessing."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128
    ):
        """
        Initialize BERT text processor.

        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, texts, device='cpu'):
        """
        Process text inputs for BERT.

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
            "bert_input_ids": encoded["input_ids"].to(device),
            "bert_attention_mask": encoded["attention_mask"].to(device)
        }
