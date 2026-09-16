#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=163840M
#SBATCH --gpus=8
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%x_%A_%a.out
#SBATCH --job-name=vfTrain
#SBATCH --partition=cs
#SBATCH --qos=cs
# Log naming: %x (job name) MUST come first. clean_slurm_outputs.py -- which is
# shared with other users -- groups files by everything after the first '_' and
# keeps only the newest per group, so the old %A_%a_%x scheme made task N of every
# array share the key 'N_<jobname>.out' and a new submission deleted a RUNNING
# job's log. Leading with a non-numeric token makes its isdigit() guard skip ours.
#
# NOTE: skeleton configs should set `cache_val_skeletons: true`. Validation clips are
# preprocessed identically on every pass, and on ASL Citizen that is ~51% of all clip
# preprocessing (10,304-clip val split x4 passes/epoch at val_interval=0.25). The cache
# is ~60 MB, bit-identical, and self-disables if augmentation is configured.

# Stage-2 retraining on visibility-filtered landmark subsets.
#
# Each array task reads one tab-separated line (config, job name) from the list
# produced by scripts/generate_visfilt_configs.py. The array index is 0-based.
#
# Usage:
#   python scripts/generate_visfilt_configs.py
#   sbatch --array=0-53%8 sbatch/train_visfilt_array.sh configs/visfilt/visfilt_configs.txt
#   sbatch --array=0-53%8 sbatch/train_visfilt_array.sh configs/visfilt/visfilt_configs.txt TEST
#
# Runs on cs/cs: 5 A100 nodes, QoS MaxWall 1 day (the cap, so time is padded to it).
# cs frees up faster than dw/matrix, and the shorter reservation avoids tripping
# AssocGrpBillingRunMinutes, which blocks starts when 3-day x 8-GPU jobs are queued.
# Do NOT mirror the same config across dw and cs: unlike idempotent extraction jobs,
# two training runs sharing one `save` directory will clobber each other's checkpoints.
#
# Modes: TRAIN (default), RESUME, TEST

CONFIG_LIST=${1:?"Error: config list file required"}
MODE=${2:-TRAIN}

if [ ! -f "$CONFIG_LIST" ]; then
    echo "Error: Config list not found: $CONFIG_LIST"
    exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")
CONFIG=$(echo "$LINE" | awk '{print $1}')
JOBNAME=$(echo "$LINE" | awk '{print $2}')

if [ -z "$CONFIG" ]; then
    echo "Error: No config found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

scontrol update jobid="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" jobname="$JOBNAME" 2>/dev/null || true

echo "=========================================="
echo "Visibility-Filtered Stage-2 Run"
echo "Array ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Job name: $JOBNAME"
echo "Config:   $CONFIG"
echo "Mode:     $MODE"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo "=========================================="

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source ~/.bashrc
conda init
conda activate asl

python src/utils/clean_slurm_outputs.py --user "$USER"

nvidia-smi

if [ "$MODE" = "TEST" ]; then
    # Lightning picks its strategy from `n_gpus`, not from the SLURM allocation.
    # Running a config written for 8-GPU training under a single process makes it
    # open a DDP rendezvous and block for 30 min waiting on 7 peers that never
    # arrive, so force single-device for the test pass.
    TEST_CONFIG="${TMPDIR:-/tmp}/$(basename "$CONFIG" .yaml)_1gpu_$$.yaml"
    sed 's/^n_gpus:.*/n_gpus: 1/' "$CONFIG" > "$TEST_CONFIG"
    python src/train.py -c "$TEST_CONFIG" -m "$MODE"
    STATUS=$?
    rm -f "$TEST_CONFIG"
else
    srun python src/train.py -c "$CONFIG" -m "$MODE"
    STATUS=$?
fi

# Keep the real exit status: this cleanup and the trailing echo below both
# succeed even when training failed, which otherwise reports COMPLETED to SLURM.
python src/utils/clean_slurm_outputs.py --user "$USER" || true

echo "=========================================="
echo "End time: $(date)"
echo "Exit status: $STATUS"
echo "=========================================="
exit $STATUS
