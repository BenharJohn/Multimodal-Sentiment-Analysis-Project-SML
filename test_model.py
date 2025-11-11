"""
Quick test script to verify model architecture and forward pass.
"""

import torch
import sys
import os

sys.path.append('src')

from systems.cdan_model import CDANModel
from models.clip_text import CLIPTextProcessor
from models.clip_image import CLIPImageProcessor


def test_model():
    """Test model instantiation and forward pass."""
    print("="*60)
    print("Testing CDAN Model Architecture")
    print("="*60)

    # Configuration
    config = {
        'clip_model_name': 'openai/clip-vit-base-patch32',
        'num_classes': 3,
        'freeze_clip': True,
        'cross_attn_layers': 2,
        'cross_attn_heads': 8,
        'cross_attn_dropout': 0.1,
        'pool_method': 'mean',
        'use_gating': True,
        'gate_type': 'simple',
        'gate_activation': 'sigmoid',
        'use_aux_decoder': True,
        'aux_loss_weight': 0.05,
        'aux_loss_type': 'cosine',
        'aux_num_layers': 2,
        'classifier_hidden_dims': [512, 256],
        'classifier_dropout': 0.3,
        'label_smoothing': 0.0
    }

    print("\n1. Building model...")
    try:
        model = CDANModel(**config)
        print("✓ Model built successfully")
    except Exception as e:
        print(f"✗ Error building model: {e}")
        return False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n2. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")

    # Create dummy batch
    print(f"\n3. Creating dummy batch...")
    batch_size = 2
    seq_len = 77
    image_size = 224

    # Text inputs
    input_ids = torch.randint(0, 49408, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Image inputs
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)

    # Labels
    labels = torch.randint(0, 3, (batch_size,))

    print(f"   Batch size: {batch_size}")
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Pixel values shape: {pixel_values.shape}")
    print(f"   Labels shape: {labels.shape}")

    # Forward pass
    print(f"\n4. Running forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                return_attention=True
            )
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check outputs
    print(f"\n5. Checking outputs...")
    expected_keys = ['logits', 'loss', 'ce_loss', 'aux_loss', 'gate_weights', 'attention_weights']
    for key in expected_keys:
        if key in outputs:
            if isinstance(outputs[key], torch.Tensor):
                print(f"   ✓ {key}: shape {outputs[key].shape}")
            elif isinstance(outputs[key], dict):
                print(f"   ✓ {key}: dict with {len(outputs[key])} entries")
            else:
                print(f"   ✓ {key}: {type(outputs[key])}")
        else:
            print(f"   ✗ Missing: {key}")

    # Check logits shape
    if outputs['logits'].shape != (batch_size, 3):
        print(f"✗ Incorrect logits shape: {outputs['logits'].shape}")
        return False

    # Test prediction
    print(f"\n6. Testing prediction mode...")
    try:
        predictions, probabilities = model.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        print(f"   ✓ Predictions shape: {predictions.shape}")
        print(f"   ✓ Probabilities shape: {probabilities.shape}")
        print(f"   Predicted classes: {predictions.tolist()}")
    except Exception as e:
        print(f"✗ Error in prediction: {e}")
        return False

    # Test freezing/unfreezing
    print(f"\n7. Testing freeze/unfreeze...")
    initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.freeze_clip()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   After freeze_clip: {frozen_trainable:,} trainable params")

    model.unfreeze_clip_last_block()
    unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   After unfreeze_last_block: {unfrozen_trainable:,} trainable params")

    if frozen_trainable >= unfrozen_trainable:
        print(f"   ✗ Unfreezing didn't work correctly")
        return False

    print("   ✓ Freeze/unfreeze working correctly")

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nModel is ready to use. You can now:")
    print("  1. Prepare data: bash scripts/prepare_data.sh")
    print("  2. Train model: bash scripts/train_cdan.sh")
    print("  3. Evaluate model: bash scripts/eval.sh --checkpoint <path>")

    return True


if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)
