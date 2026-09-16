#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32768M
#SBATCH --gpus=1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%x_%A_%a.out
#SBATCH --job-name=cascade
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --exclude=dw-2-4,cs-1-2
# dw-2-4 fails every job at CUDA init (torch._C._cuda_init) within ~30 s and, because it frees
# up so fast, the scheduler keeps feeding it work: on 2026-09-14 it swallowed 25 of 54 jobs
# while every other node succeeded. docs/submission_instructions.md already listed both nodes
# as known-bad; putting the exclusion here stops it depending on remembering a CLI flag.

# PARTITION POLICY (2026-09-04): do not use cs3. That workload is CPU-bound and uses ~1% of
# a B200's memory at 0-25% utilization; leave those cards for jobs that can fill them.
#
# Iterative-cascade landmark selection, one language per array task. Each task runs FOUR
# gate trainings sequentially (270 -> 100 -> 48 -> 24, each selecting the next subset), so
# this is a single long job rather than an array of independent ones -- every stage depends
# on the previous stage's output subset.
#
# 32 CPUs, unlike the 16 used for the bulk stage-2 queue. Throughput here is set by
# DataLoader worker count, not GPU count: 16 CPU -> 12.02 it/s, 32 CPU -> 22.40 it/s on one
# A100. For the bulk queue 16 wins because it fits 8 jobs per node and maximises aggregate
# throughput, but the cascade is SEQUENTIAL and latency-bound -- only 3 tasks run at once,
# one per language, and halving each stage's wall clock shortens the critical path directly.
# train.py derives num_workers from SLURM_CPUS_PER_TASK - 2, so this needs no config change.
#
# Walltime: 4 stages at up to ~2 h each, so 1 day is ample. `matrix` QoS allows 3 days if a
# language turns out slower than expected.
#
# Usage:
#   sbatch --array=0-2 sbatch/run_cascade.sh

LANGS=(autsl asl_citizen gsl)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$LANG" ]; then
    echo "Error: no language for array index $SLURM_ARRAY_TASK_ID (expected 0-2)"
    exit 1
fi

scontrol update jobid="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    jobname="casc_${LANG:0:2}" 2>/dev/null || true

echo "=========================================="
echo "Iterative cascade: $LANG"
echo "Array:    ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:     $SLURMD_NODENAME"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}  (num_workers will be $((SLURM_CPUS_PER_TASK - 2)))"
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

# Bash drives the chain so that every training step is launched exactly the way the working
# scripts launch it: `srun python src/train.py`. The first attempt (13585411) had a Python
# parent call `python src/train.py` via subprocess and all three chains died in 30 s with
#   Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1
# I could not reproduce that from a Python parent on the login node, so the root cause is not
# confirmed -- the one clear difference from every job that does work is the missing `srun`.
# Rather than keep guessing at MKL internals on a queue we need this weekend, the orchestration
# moved to bash and matches the proven invocation. Python now only writes configs and reads
# checkpoints, never spawns training.

set -o pipefail
STATUS=0

STAGES=($(python scripts/run_cascade.py --lang "$LANG" --stages))
IDX_FILE=$(python scripts/run_cascade.py --lang "$LANG" --seed-pool) || exit 1
echo "[cascade:$LANG] starting pool: $IDX_FILE"

for ((i = 0; i < ${#STAGES[@]} - 1; i++)); do
    POOL_K=${STAGES[$i]}
    NEXT_K=${STAGES[$((i + 1))]}
    IDX_DIR=$(dirname "$IDX_FILE")

    CFG=$(python scripts/run_cascade.py --lang "$LANG" --emit-config "$POOL_K" "$NEXT_K") || { STATUS=1; break; }
    echo "[cascade:$LANG] stage $POOL_K -> $NEXT_K : $CFG   $(date +%H:%M:%S)"

    srun python src/train.py -c "$CFG" -m TRAIN
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "[cascade:$LANG] TRAINING FAILED at stage $POOL_K (rc=$STATUS); stopping chain"
        break
    fi

    CKPT="$(python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['save'])")/checkpoints/last.ckpt"
    if [ ! -f "$CKPT" ]; then
        echo "[cascade:$LANG] missing $CKPT; stopping chain"
        STATUS=1
        break
    fi

    python scripts/extract_gate_ranking.py \
        --checkpoint "$CKPT" \
        --pool "$IDX_DIR/iter_${POOL_K}_indices.json" \
        --keep "$NEXT_K" \
        --out "$IDX_DIR/iter_${NEXT_K}_indices.json"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "[cascade:$LANG] SELECTION FAILED at stage $POOL_K (rc=$STATUS); stopping chain"
        break
    fi
done

if [ $STATUS -eq 0 ]; then
    echo "[cascade:$LANG] chain complete"
fi

echo "=========================================="
echo "End:      $(date)"
echo "Exit:     $STATUS"
echo "=========================================="

# Propagate the real status; a trailing command here would mask a failed chain and make the
# job report COMPLETED (this bit us before with the TEST-mode DDP hang).
exit $STATUS
