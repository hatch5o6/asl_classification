# Configuration Recommendations for ASL Feature Ablation

## Overview
This document contains Claude's recommendations for model configurations in the Turkish Sign Language classification project. These recommendations are designed for a class assignment proof-of-concept without hyperparameter optimization.

## Design Principles

1. **Balance model capacity** across modalities so no single encoder dominates
2. **Scale architectures** to be efficient but expressive enough for sign language
3. **Match dimensions** for clean fusion
4. **Account for training from scratch vs. pretrained** - give more capacity to encoders trained from scratch

## Recommended Configuration Parameters

### RGB Encoder (Pretrained VideoMAE)
```yaml
# Uses pretrained "MCG-NJU/videomae-base-finetuned-kinetics"
# Keep default config from pretrained model
# hidden_size: 768 (from pretrained model)
pretrained_learning_rate: 5e-05
```

**Rationale:** Pretrained on Kinetics dataset, already has good video understanding. Use lower learning rate to preserve learned features.

### Depth Encoder (VideoMAE architecture, trained from scratch)
```yaml
depth_image_size: 224
depth_hidden_dim: 512          # Increased from 384
depth_hidden_layers: 8         # Increased from 6
depth_att_heads: 8             # Increased from 6
depth_intermediate_size: 2048  # Standard 4x hidden_dim (512 * 4)
depth_patch_size: 16
new_learning_rate: 1e-04
```

**Rationale:**
- Depth is highly informative for hand shapes and gestures in sign language
- Trained from scratch, so needs more capacity than a pretrained model
- 8 layers and 512 hidden dim provide sufficient expressiveness
- Intermediate size follows standard Transformer convention (4x hidden_dim)
- Attention heads must divide hidden_dim evenly (512 / 8 = 64)

### Skeleton Encoder (BERT architecture)
```yaml
bert_hidden_dim: 384           # Increased from 256
bert_hidden_layers: 6          # Increased from 4
bert_att_heads: 6              # Increased from 4
bert_intermediate_size: 1536   # Standard 4x hidden_dim (384 * 4)
num_pose_points: 543
skel_learning_rate: 1e-04
```

**Rationale:**
- Skeleton data is lower-dimensional than video but temporal patterns are crucial
- More layers (6) help capture motion dynamics and gesture sequences
- 384 hidden dim balances capacity with efficiency
- Attention heads must divide hidden_dim evenly (384 / 6 = 64)

### Fusion & Classifier
```yaml
fusion_dim: 384                # Increased from 256
classifier_dropout: 0.1
class_learning_rate: 1e-04
```

**Rationale:**
- Larger fusion dimension (384) reduces bottleneck when combining RGB (768), depth (512), and skeleton (384) features
- Weighted fusion uses learnable modality weights (softmax normalized)
- Moderate dropout (0.1) for regularization without being too aggressive

### Training Configuration
```yaml
effective_batch_size: 32
max_steps: 200000
early_stop: 10
val_interval: 0.25
weight_decay: 0.01
gradient_clip_val: 1.0
warmup_steps: 10000           # 5% of max_steps
```

**Rationale:**
- Batch size of 32 is standard for video models
- 200k steps with early stopping provides sufficient training
- Linear warmup (5%) followed by linear decay for stable training
- Gradient clipping prevents training instabilities

## Summary of Changes from Current Config

| Component | Current | Recommended | Change Reason |
|-----------|---------|-------------|---------------|
| **Depth Encoder** | | | |
| depth_hidden_dim | 384 | **512** | More capacity for learning from scratch |
| depth_hidden_layers | 6 | **8** | Depth is very informative for sign language |
| depth_att_heads | 6 | **8** | Match increased capacity, maintain head_dim=64 |
| depth_intermediate_size | 1536 | **2048** | Standard 4x hidden_dim |
| **Skeleton Encoder** | | | |
| bert_hidden_dim | 256 | **384** | Better temporal modeling capacity |
| bert_hidden_layers | 4 | **6** | Capture complex motion patterns |
| bert_att_heads | 4 | **6** | Match increased capacity, maintain head_dim=64 |
| bert_intermediate_size | 512 | **1536** | Standard 4x hidden_dim |
| **Fusion** | | | |
| fusion_dim | 256 | **384** | Reduce fusion bottleneck |

## Learning Rate Strategy

**Fixed learning rates (no hyperparameter search needed):**

1. **Pretrained RGB encoder**: `5e-5` (0.00005)
   - Lower rate to prevent catastrophic forgetting
   - Standard practice for fine-tuning vision transformers

2. **New components** (depth, skeleton, heads): `1e-4` (0.0001)
   - 2x higher than pretrained components
   - Appropriate for training from scratch

3. **Classifier and fusion weights**: `1e-4` (0.0001)
   - Same as new components

This follows the standard practice where pretrained layers use ~2x lower learning rates than newly initialized layers.

## Architecture Details

### Model Flow
```
RGB Video (224x224x3x16) → VideoMAE Encoder (pretrained) → RGB features (768)
                                                                ↓
Depth Video (224x224x1x16) → VideoMAE Encoder (scratch) → Depth features (512)
                                                                ↓
Skeleton (543 keypoints x 2 coords x 16 frames) → BERT Encoder → Skel features (384)
                                                                ↓
                                    Weighted Fusion (learnable weights) → Fused features (384)
                                                                ↓
                                            Classifier (2-layer MLP with GELU, dropout)
                                                                ↓
                                            Logits (226 classes for AUTSL)
```

### Feature Extraction
- **RGB & Depth**: Use mean pooling over sequence dimension from VideoMAE's last hidden state
- **Skeleton**: Use CLS token (first token) from BERT's last hidden state
- **Fusion**: Softmax-normalized weighted sum of modality features

## Experimental Ablations

With these configurations, run experiments on:
1. **RGB only** - Baseline with pretrained VideoMAE
2. **Skeleton only** - Test pose-based recognition
3. **RGB + Depth** - Evaluate depth contribution
4. **RGB + Skeleton** - Evaluate pose contribution
5. **RGB + Depth + Skeleton** - Full multimodal model

This allows systematic analysis of each modality's contribution to sign language classification performance.

## Notes

- All configs use the same seed (4000) for reproducibility
- Class-weighted cross-entropy loss to handle class imbalance in AUTSL dataset
- Configurations are balanced to ensure fair comparison across ablation conditions
- No hyperparameter search performed due to time constraints (class assignment)
- For published work, recommend Optuna-based hyperparameter optimization on validation set
