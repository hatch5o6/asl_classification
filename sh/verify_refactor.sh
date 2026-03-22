#!/bin/bash
# Verification script for src/ reorganization and encoder refactor.
# Run these commands to verify everything works after the refactor.
#
# Step 1: Import check (run on login node, no GPU needed)
# Step 2: SLURM smoke tests (submit when queue has capacity)

echo "=========================================="
echo "Step 1: Import verification (login node)"
echo "=========================================="

cd /home/ccoulson/asl_classification

# Activate the conda environment
source ~/.bashrc
conda activate asl

# Test core model imports
echo "Testing core imports..."
python -c "
import sys
sys.path.insert(0, 'src')

# Core model imports
from models.lightning_asl import SignClassificationLightning
from models.joint_pruning import JointPruningModule, l0_penalty
from models.skeleton_encoders import build_skeleton_encoder, SKELETON_ENCODERS

# Data imports
from data.asl_dataset import RGBDSkel_Dataset

# Verify all encoders are registered
print(f'Available encoders: {list(SKELETON_ENCODERS.keys())}')

# Verify factory works with minimal config (no GPU needed for init check)
import torch
for enc_name in SKELETON_ENCODERS:
    config = {
        'skeleton_encoder': enc_name,
        'num_pose_points': 48,
        'num_coords': 2,
        'fusion_dim': 256,
        'num_frames': 16,
        'bert_hidden_dim': 256,
        'bert_hidden_layers': 4,
        'bert_att_heads': 4,
        'bert_intermediate_size': 1024,
        'bert_dropout': 0.0,
    }
    encoder = build_skeleton_encoder(config)
    # Test forward pass on CPU
    dummy = torch.randn(2, 16, 48, 2)
    out = encoder(dummy)
    assert out.shape == (2, 256), f'{enc_name}: expected (2, 256), got {out.shape}'
    print(f'  {enc_name}: OK (output shape {out.shape})')

print()
print('All import and forward pass checks passed!')
"

if [ $? -ne 0 ]; then
    echo "FAILED: Import verification failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: SLURM smoke tests"
echo "=========================================="
echo ""
echo "Submit these manually when the queue has capacity:"
echo ""
echo "  # Test existing BERT config still works:"
echo "  sbatch --job-name=verify_bert sbatch/train_informed_selection.sh configs/informed_selection/topk_48.yaml"
echo ""
echo "  # Test each new encoder (create smoke configs first, see docs/experiments.md):"
echo "  sbatch --job-name=verify_gru sbatch/train_informed_selection.sh configs/encoder_comparison/gru_autsl_48.yaml"
echo "  sbatch --job-name=verify_stgcn sbatch/train_informed_selection.sh configs/encoder_comparison/stgcn_autsl_48.yaml"
echo "  sbatch --job-name=verify_spoter sbatch/train_informed_selection.sh configs/encoder_comparison/spoter_autsl_48.yaml"
echo ""
echo "  Check SLURM output logs for import errors in the first few minutes."
echo "  Cancel jobs once training starts successfully (scancel <job_id>)."
