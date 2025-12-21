"""
CDAN: Cross-Domain Attention Network for Multimodal Sentiment Analysis
Main model that integrates CLIP encoders, cross-attention fusion, gating, auxiliary decoder, and classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_text import CLIPTextEncoder
from models.clip_image import CLIPImageEncoder
from models.cross_attention import BidirectionalCrossAttention
from models.gating import ModalityGate, AttentionGate, HierarchicalGate
from models.aux_decoder import DualAuxiliaryDecoder
from models.classifier import MLPClassifier
from models.losses import FocalLoss
from models.bert_encoder import BERTTextEncoder
from models.resnet_encoder import ResNetImageEncoder
from models.multi_encoder_fusion import AttentionEncoderFusion, SelfAdaptiveAggregation


class CDANModel(nn.Module):
    """
    CDAN Model for Multimodal Sentiment Analysis.

    Architecture:
        1. CLIP Text & Image Encoders (frozen initially)
        2. Bidirectional Cross-Attention Fusion
        3. Modality Gating
        4. Auxiliary Reconstruction Decoder
        5. Sentiment Classification Head
    """

    def __init__(
        self,
        # Model configuration
        clip_model_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 3,
        freeze_clip: bool = True,

        # Cross-attention configuration
        cross_attn_layers: int = 2,
        cross_attn_heads: int = 8,
        cross_attn_dropout: float = 0.1,
        pool_method: str = "mean",

        # Gating configuration
        use_gating: bool = True,
        gate_type: str = "simple",  # 'simple', 'attention', 'hierarchical'
        gate_activation: str = "sigmoid",

        # Auxiliary decoder configuration
        use_aux_decoder: bool = True,
        aux_loss_weight: float = 0.05,
        aux_loss_type: str = "cosine",
        aux_num_layers: int = 2,

        # Classifier configuration
        classifier_hidden_dims: list = None,
        classifier_dropout: float = 0.3,

        # Training configuration
        label_smoothing: float = 0.0,

        # Loss configuration
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: list = None,  # Class weights [w0, w1, w2]

        # Dual encoder configuration (CDAN 2025)
        use_dual_encoders: bool = False,
        bert_model_name: str = "bert-base-uncased",
        resnet_model: str = "resnet50",
        freeze_aux_encoders: bool = True,

        # Self-adaptive aggregation (CDAN 2025 key innovation)
        use_decoder_feedback: bool = True,  # Feed decoder output back into fusion
    ):
        """
        Initialize CDAN model.

        Args:
            clip_model_name: HuggingFace CLIP model identifier
            num_classes: Number of sentiment classes
            freeze_clip: Whether to initially freeze CLIP encoders
            cross_attn_layers: Number of cross-attention layers
            cross_attn_heads: Number of attention heads
            cross_attn_dropout: Dropout in cross-attention
            pool_method: Pooling method for sequences ('mean', 'max', 'cls')
            use_gating: Whether to use modality gating
            gate_type: Type of gating mechanism
            gate_activation: Activation for gates
            use_aux_decoder: Whether to use auxiliary decoder
            aux_loss_weight: Weight for auxiliary reconstruction loss
            aux_loss_type: Type of auxiliary loss ('cosine', 'l2')
            aux_num_layers: Number of layers in auxiliary decoder
            classifier_hidden_dims: Hidden dimensions for classifier
            classifier_dropout: Dropout in classifier
            label_smoothing: Label smoothing factor
            use_focal_loss: Whether to use focal loss for class imbalance
            focal_gamma: Focusing parameter for focal loss (higher = more focus on hard examples)
            focal_alpha: Class weights for focal loss [w0, w1, w2]
            use_dual_encoders: Whether to use dual encoders (CLIP + BERT/ResNet) per CDAN 2025
            bert_model_name: BERT model name for text auxiliary encoder
            resnet_model: ResNet model name for image auxiliary encoder
            freeze_aux_encoders: Whether to freeze auxiliary encoders initially
            use_decoder_feedback: Whether to use decoder output in final fusion (CDAN 2025)
        """
        super().__init__()

        # Store configuration
        self.num_classes = num_classes
        self.use_gating = use_gating
        self.use_aux_decoder = use_aux_decoder
        self.aux_loss_weight = aux_loss_weight
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.use_dual_encoders = use_dual_encoders
        self.use_decoder_feedback = use_decoder_feedback and use_aux_decoder  # Requires aux decoder

        # Initialize loss function
        if use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
        else:
            self.loss_fn = None  # Will use F.cross_entropy

        # CLIP Encoders
        self.text_encoder = CLIPTextEncoder(
            model_name=clip_model_name,
            freeze=freeze_clip
        )
        self.image_encoder = CLIPImageEncoder(
            model_name=clip_model_name,
            freeze=freeze_clip
        )

        # Get dimensions
        self.text_hidden_dim = self.text_encoder.hidden_dim
        self.image_hidden_dim = self.image_encoder.hidden_dim
        self.projection_dim = self.text_encoder.projection_dim

        # Use projection_dim as common dimension for cross-attention
        # CLIP text and vision have different hidden dims (512 vs 768)
        # but both project to the same projection_dim (512)
        self.hidden_dim = self.projection_dim

        # Add projection layers to align dimensions if needed
        if self.text_hidden_dim != self.hidden_dim:
            self.text_projection_layer = nn.Linear(self.text_hidden_dim, self.hidden_dim)
        else:
            self.text_projection_layer = nn.Identity()

        if self.image_hidden_dim != self.hidden_dim:
            self.image_projection_layer = nn.Linear(self.image_hidden_dim, self.hidden_dim)
        else:
            self.image_projection_layer = nn.Identity()

        # Dual Encoders (CDAN 2025): BERT + ResNet alongside CLIP
        if self.use_dual_encoders:
            # BERT text encoder
            self.bert_encoder = BERTTextEncoder(
                model_name=bert_model_name,
                freeze=freeze_aux_encoders
            )
            # ResNet image encoder
            self.resnet_encoder = ResNetImageEncoder(
                model_name=resnet_model,
                freeze=freeze_aux_encoders
            )

            # Attention-based fusion modules
            # Fuse CLIP text (512) with BERT (768) -> 512
            self.text_fusion = AttentionEncoderFusion(
                clip_dim=self.projection_dim,  # 512
                aux_dim=self.bert_encoder.hidden_dim,  # 768
                output_dim=self.projection_dim,  # 512
                num_heads=8,
                dropout=cross_attn_dropout
            )
            # Fuse CLIP vision (512) with ResNet (2048) -> 512
            self.image_fusion = AttentionEncoderFusion(
                clip_dim=self.projection_dim,  # 512
                aux_dim=self.resnet_encoder.hidden_dim,  # 2048
                output_dim=self.projection_dim,  # 512
                num_heads=8,
                dropout=cross_attn_dropout
            )

        # Cross-Attention Fusion
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=self.hidden_dim,
            num_heads=cross_attn_heads,
            num_layers=cross_attn_layers,
            dropout=cross_attn_dropout,
            pool_method=pool_method
        )

        # Modality Gating
        if self.use_gating:
            if gate_type == "simple":
                self.gate = ModalityGate(
                    embed_dim=self.hidden_dim,
                    num_modalities=2,
                    gate_activation=gate_activation,
                    dropout=cross_attn_dropout
                )
            elif gate_type == "attention":
                self.gate = AttentionGate(
                    embed_dim=self.hidden_dim,
                    num_modalities=2,
                    dropout=cross_attn_dropout
                )
            elif gate_type == "hierarchical":
                self.gate = HierarchicalGate(
                    embed_dim=self.hidden_dim,
                    num_modalities=2,
                    dropout=cross_attn_dropout
                )
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")

            # Fusion dimension after gating (concatenated)
            self.fusion_dim = self.hidden_dim * 2
        else:
            # Simple concatenation
            self.fusion_dim = self.hidden_dim * 2

        # Auxiliary Decoder
        if self.use_aux_decoder:
            self.aux_decoder = DualAuxiliaryDecoder(
                input_dim=self.hidden_dim,
                text_output_dim=self.projection_dim,
                image_output_dim=self.projection_dim,
                hidden_dim=self.hidden_dim,
                num_layers=aux_num_layers,
                dropout=cross_attn_dropout,
                loss_type=aux_loss_type
            )

        # Self-Adaptive Aggregation (CDAN 2025 key innovation)
        # Feeds decoder output back into fusion for refined representations
        if self.use_decoder_feedback:
            # Aggregate: [text_fused, image_fused, text_reconstructed, image_reconstructed]
            # All are projection_dim (512)
            self.adaptive_aggregation = SelfAdaptiveAggregation(
                input_dims=[self.projection_dim] * 4,  # 4 streams of 512 each
                output_dim=self.hidden_dim * 2,  # Match fusion_dim
                dropout=cross_attn_dropout
            )
            # Override fusion_dim when using decoder feedback
            self.fusion_dim = self.hidden_dim * 2

        # Classifier
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [512, 256]

        self.classifier = MLPClassifier(
            input_dim=self.fusion_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout=classifier_dropout,
            use_batch_norm=True
        )

    def freeze_clip(self):
        """Freeze all CLIP encoder parameters."""
        self.text_encoder.freeze()
        self.image_encoder.freeze()

    def unfreeze_clip(self):
        """Unfreeze all CLIP encoder parameters."""
        self.text_encoder.unfreeze()
        self.image_encoder.unfreeze()

    def unfreeze_clip_last_block(self):
        """Unfreeze only the last block of CLIP encoders."""
        self.text_encoder.unfreeze_last_block()
        self.image_encoder.unfreeze_last_block()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        bert_input_ids: Optional[torch.Tensor] = None,
        bert_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch_size, seq_len] CLIP text token IDs
            attention_mask: [batch_size, seq_len] CLIP text attention mask
            pixel_values: [batch_size, C, H, W] image pixels
            labels: [batch_size] ground truth labels (optional)
            return_attention: Whether to return attention weights
            bert_input_ids: [batch_size, seq_len] BERT token IDs (for dual encoders)
            bert_attention_mask: [batch_size, seq_len] BERT attention mask (for dual encoders)

        Returns:
            dict with:
                - logits: [batch_size, num_classes] classification logits
                - loss: scalar total loss (if labels provided)
                - ce_loss: scalar classification loss (if labels provided)
                - aux_loss: scalar auxiliary loss (if labels provided and aux enabled)
                - gate_weights: [batch_size, 2] modality gate weights (if gating enabled)
                - attention_weights: dict of attention maps (if return_attention=True)
        """
        batch_size = input_ids.size(0)

        # Extract CLIP features
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_outputs = self.image_encoder(
            pixel_values=pixel_values
        )

        # Get token-level and pooled features
        text_tokens = text_outputs['text_tokens']          # [B, T, text_hidden_dim]
        text_pooled = text_outputs['text_pooled']          # [B, projection_dim]
        vision_patches = image_outputs['vision_patches']   # [B, P, image_hidden_dim]
        vision_pooled = image_outputs['vision_pooled']     # [B, projection_dim]

        # Dual encoder fusion (CDAN 2025)
        if self.use_dual_encoders and bert_input_ids is not None:
            # Extract BERT features
            bert_outputs = self.bert_encoder(
                input_ids=bert_input_ids,
                attention_mask=bert_attention_mask
            )
            bert_pooled = bert_outputs['pooled']  # [B, 768]

            # Extract ResNet features
            resnet_outputs = self.resnet_encoder(pixel_values)
            resnet_features = resnet_outputs['features']  # [B, 2048]

            # Fuse CLIP + auxiliary features using attention
            text_pooled = self.text_fusion(text_pooled, bert_pooled)      # [B, 512]
            vision_pooled = self.image_fusion(vision_pooled, resnet_features)  # [B, 512]

        # Project tokens to common dimension for cross-attention
        text_tokens_proj = self.text_projection_layer(text_tokens)       # [B, T, hidden_dim]
        vision_patches_proj = self.image_projection_layer(vision_patches) # [B, P, hidden_dim]

        # Cross-attention fusion
        cross_attn_output = self.cross_attention(
            text_tokens=text_tokens_proj,
            vision_patches=vision_patches_proj,
            text_mask=attention_mask,
            return_attention=return_attention
        )

        text_fused = cross_attn_output['text_fused']       # [B, D]
        image_fused = cross_attn_output['image_fused']     # [B, D]

        # Modality gating (optional)
        if self.use_gating:
            gate_output = self.gate(text_fused, image_fused)
            gated_features = gate_output['gated_features']  # [B, D*2]
            gate_weights = gate_output['gate_weights']      # [B, 2]
        else:
            # Simple concatenation
            gated_features = torch.cat([text_fused, image_fused], dim=-1)
            gate_weights = None

        # Self-Adaptive Aggregation with decoder feedback (CDAN 2025)
        # Feed decoder reconstructions back into fusion for refined representations
        aux_loss = None
        adaptive_weights = None
        if self.use_decoder_feedback:
            # Run auxiliary decoder to get reconstructed features
            aux_output = self.aux_decoder(
                text_fused=text_fused,
                image_fused=image_fused,
                text_target=text_pooled.detach() if labels is not None else None,
                image_target=vision_pooled.detach() if labels is not None else None
            )
            text_reconstructed = aux_output['text_reconstructed']    # [B, 512]
            image_reconstructed = aux_output['image_reconstructed']  # [B, 512]

            # Store aux loss for later
            if 'total_loss' in aux_output:
                aux_loss = aux_output['total_loss']

            # Self-adaptive aggregation: combine original + reconstructed features
            # The model learns optimal weights for each feature stream
            fused_features, adaptive_weights = self.adaptive_aggregation(
                text_fused, image_fused, text_reconstructed, image_reconstructed
            )
        else:
            fused_features = gated_features

        # Classification
        logits = self.classifier(fused_features)  # [B, num_classes]

        # Prepare output
        output = {
            'logits': logits,
            'text_fused': text_fused,
            'image_fused': image_fused,
            'fused_features': fused_features
        }

        if gate_weights is not None:
            output['gate_weights'] = gate_weights

        if adaptive_weights is not None:
            output['adaptive_weights'] = adaptive_weights

        if return_attention:
            output['attention_weights'] = cross_attn_output['attention_weights']

        # Compute losses if labels are provided
        if labels is not None:
            # Classification loss (Focal Loss or Cross-Entropy)
            if self.use_focal_loss and self.loss_fn is not None:
                ce_loss = self.loss_fn(logits, labels)
            elif self.label_smoothing > 0:
                ce_loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=self.label_smoothing
                )
            else:
                ce_loss = F.cross_entropy(logits, labels)

            output['ce_loss'] = ce_loss
            total_loss = ce_loss

            # Auxiliary reconstruction loss
            if self.use_aux_decoder:
                if aux_loss is None:
                    # Decoder not yet run (use_decoder_feedback=False but use_aux_decoder=True)
                    aux_output = self.aux_decoder(
                        text_fused=text_fused,
                        image_fused=image_fused,
                        text_target=text_pooled.detach(),
                        image_target=vision_pooled.detach()
                    )
                    aux_loss = aux_output['total_loss']
                    output['text_aux_loss'] = aux_output['text_loss']
                    output['image_aux_loss'] = aux_output['image_loss']
                else:
                    # Already computed during decoder feedback
                    output['text_aux_loss'] = aux_output.get('text_loss', aux_loss / 2)
                    output['image_aux_loss'] = aux_output.get('image_loss', aux_loss / 2)

                output['aux_loss'] = aux_loss

                # Add weighted auxiliary loss to total
                total_loss = total_loss + self.aux_loss_weight * aux_loss

            output['loss'] = total_loss

        return output

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode: predict class labels and probabilities.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size, C, H, W]

        Returns:
            predictions: [batch_size] predicted class indices
            probabilities: [batch_size, num_classes] class probabilities
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None
            )

            logits = output['logits']
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

        return predictions, probabilities

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings at different stages of the model.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size, C, H, W]

        Returns:
            dict with embeddings from different layers
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None
            )

        return {
            'text_fused': output['text_fused'],
            'image_fused': output['image_fused'],
            'fused_features': output['fused_features']
        }


def build_cdan_model(config: dict) -> CDANModel:
    """
    Build CDAN model from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized CDANModel
    """
    return CDANModel(
        clip_model_name=config.get('clip_model', config.get('clip_model_name', 'openai/clip-vit-base-patch32')),
        num_classes=config.get('num_classes', 3),
        freeze_clip=config.get('freeze_clip', True),
        cross_attn_layers=config.get('cross_attn_layers', 2),
        cross_attn_heads=config.get('cross_attn_heads', 8),
        cross_attn_dropout=config.get('cross_attn_dropout', 0.1),
        pool_method=config.get('pool_method', 'mean'),
        use_gating=config.get('use_gating', True),
        gate_type=config.get('gate_type', 'simple'),
        gate_activation=config.get('gate_activation', 'sigmoid'),
        use_aux_decoder=config.get('use_aux_decoder', True),
        aux_loss_weight=config.get('aux_loss_weight', 0.05),
        aux_loss_type=config.get('aux_loss_type', 'cosine'),
        aux_num_layers=config.get('aux_num_layers', 2),
        classifier_hidden_dims=config.get('classifier_hidden_dims', [512, 256]),
        classifier_dropout=config.get('classifier_dropout', 0.3),
        label_smoothing=config.get('label_smoothing', 0.0),
        # Focal loss for class imbalance
        use_focal_loss=config.get('use_focal_loss', False),
        focal_gamma=config.get('focal_gamma', 2.0),
        focal_alpha=config.get('focal_alpha', None),
        # Dual encoder configuration (CDAN 2025)
        use_dual_encoders=config.get('use_dual_encoders', False),
        bert_model_name=config.get('bert_model_name', 'bert-base-uncased'),
        resnet_model=config.get('resnet_model', 'resnet50'),
        freeze_aux_encoders=config.get('freeze_aux_encoders', True),
        # Self-adaptive aggregation (CDAN 2025 key innovation)
        use_decoder_feedback=config.get('use_decoder_feedback', True)
    )
