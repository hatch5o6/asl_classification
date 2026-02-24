#!/bin/bash

# Submit all 3 gating experiments
# 1. gating_only: Test gating without L0 pruning (baseline)
# 2. gating_l0: Test gating + L0 (no random init)
# 3. gating_hybrid: Test gating + L0 + random init (3 runs with different L0 weights)

echo "=================================================="
echo "Submitting GATING mechanism experiments"
echo "=================================================="
echo ""
echo "This tests the solution to the gradient uniformity problem."
echo ""
echo "Experiments:"
echo "  1. gating_only    - Gating without L0 (baseline)"
echo "  2. gating_l0      - Gating + L0, no random init"
echo "  3. gating_hybrid  - Gating + L0 + random init (RECOMMENDED)"
echo ""
echo "Key features:"
echo "  - Sample-specific gating network (~257 params)"
echo "  - Creates 10-100x gradient differentiation"
echo "  - Breaks symmetric initialization problem"
echo ""

# Option 1: Gating only (baseline)
echo "----------------------------------------"
echo "Option 1: Submitting gating_only (baseline)..."
echo "  - NO L0 pruning"
echo "  - Tests if gating helps classification"
sbatch sbatch/train_gating_sweep.sh gating_only
echo ""

# Option 2: Gating + L0 (no random init)
echo "----------------------------------------"
echo "Option 2: Submitting gating_l0..."
echo "  - Gating + L0 pruning"
echo "  - NO random initialization"
echo "  - Tests if gating provides enough differentiation"
sbatch sbatch/train_gating_sweep.sh gating_l0
echo ""

# Option 3: Gating + L0 + Random Init (HYBRID - RECOMMENDED)
echo "----------------------------------------"
echo "Option 3: Submitting gating_hybrid (RECOMMENDED)..."
echo "  - Gating + L0 + random init"
echo "  - Multiple safety mechanisms"
echo "  - Sweeps L0 weights: 15.0, 20.0, 25.0"
sbatch sbatch/train_gating_sweep.sh gating_hybrid
echo ""

echo "=================================================="
echo "All jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs at:"
echo "  /home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/"
echo ""
echo "What to look for (at step ~50k):"
echo "  ✅ gate_std > 0.20       (strong differentiation)"
echo "  ✅ joint_prob spread > 0.3  (bimodal emerging)"
echo "  ✅ pruning_ratio: 0.3-0.7   (30-70% pruned)"
echo "=================================================="
