#!/bin/bash

# PARTITION POLICY (2026-09-04): **do not use cs3.** cs-3-1's B200s are wasted on this
# workload -- measured 843-2098 MiB of 183,359 MiB GPU memory (~1%) and 0-25% utilization,
# because the BERT skeleton encoder flattens landmarks into one per-frame vector, so the
# transformer sees only ~16 tokens over 2 layers and the GPU is never the bottleneck (CPU
# preprocessing is, dominated by NaN interpolation). B200 also bills 2.86x an A100 for ~1.6x
# the speed. Use dw (A100, ~2 TiB RAM/node) first, then cs (A100). Leave the B200s for jobs
# that can actually fill them.

#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=24576M
#SBATCH --gpus=1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%x_%A_%a.out
#SBATCH --job-name=vf1gpu
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=dw-2-4,cs-1-2
# dw-2-4 fails every job at CUDA init (torch._C._cuda_init) within ~30 s and, because it frees
# up so fast, the scheduler keeps feeding it work: on 2026-09-14 it swallowed 25 of 54 jobs
# while every other node succeeded. docs/submission_instructions.md already listed both nodes
# as known-bad; putting the exclusion here stops it depending on remembering a CLI flag.
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
# 1-GPU / 16-CPU variant, and the default for this project from 2026-09-04 on.
#
# Throughput is set by DataLoader WORKER COUNT, not GPU count. Measured on one A100 with an
# otherwise identical GSL config (median training optimizer steps/s):
#
#   2 GPU, 16 CPU, 12 workers -> 10.96 it/s   billing 448   4 jobs/node   43.8 node it/s
#   1 GPU,  8 CPU,  6 workers ->  5.29 it/s   billing 224   8 jobs/node   42.3 node it/s
#   1 GPU, 16 CPU, 14 workers -> 12.02 it/s   billing 224   8 jobs/node   96.2 node it/s  <-- this
#   1 GPU, 32 CPU, 30 workers -> 22.40 it/s   billing 448   4 jobs/node   89.6 node it/s
#
# 16 CPU + 1 GPU wins on aggregate throughput AND on billing per unit work (207 vs 223 vs 457
# billing-hours per 40k steps), and it saturates a 128-core/8-GPU node exactly at 8 jobs. Use
# 32 CPU only for latency-critical SEQUENTIAL work -- e.g. the iterative cascade chains, where
# halving per-stage wall clock shortens the critical path and only 3 jobs run at once.
#
# Beware: `--cpus-per-task` is PER TASK, so halving the ranks halves the workers. That is what
# made an 8-CPU 1-GPU run look 2.2x slower than its 2-GPU counterpart and briefly convinced us
# the GPU count mattered. train.py now derives num_workers from SLURM_CPUS_PER_TASK - 2, so the
# two settings can no longer drift apart. Host RSS is 11.3 GiB/rank, so 24 GiB is ~2x headroom;
# GPU memory use was 2,098 MiB of 183,359 MiB -- the GPUs were starved, not spare.
#
# Superseded note (kept for the billing figures): AllocTRES bills 2 B200 at 1280 and 2 A100 at 448,
# i.e. B200 costs 2.86x for roughly 1.6x the speed -- a net loss per billing unit, and
# billing is what AssocGrpBillingRunMinutes caps. Leave the B200s for jobs that can fill them.
#
# Single-GPU jobs also schedule far better: whole-cluster GPUs are frequently allocated
# 7/8 or 8/8 per node, so free capacity is stranded one GPU at a time and a 2-GPU request
# waits on (Resources) while a 1-GPU request starts immediately.
#
# Superseded rationale from the 2-GPU variant: MaxRSS on completed runs is ~4.6 GB, so the inherited
# --mem=1024000M (1000 GiB) over-requested by ~225x and, together with 8 whole GPUs,
# made the job unschedulable. The model is tiny (2-layer/512 BERT over 48x2 inputs) and
# effective_batch_size is divided by n_gpus in train.py, so fewer ranks preserves the
# optimization and cuts DDP allreduce overhead.
#
# Runs on cs/cs: 5 A100 nodes, QoS MaxWall 1 day (the cap, so time is padded to it).
# cs frees up faster than dw/matrix, and the shorter reservation avoids tripping
# AssocGrpBillingRunMinutes, which blocks starts when 3-day x 8-GPU jobs are queued.
# Do NOT mirror the same config across dw and cs: unlike idempotent extraction jobs,
# two training runs sharing one `save` directory will clobber each other's checkpoints.
#
# PARTITIONS. cs and dw are served by different, partition-locked QoSes -- cs takes
# --qos=cs (MaxWall 1 day), dw takes --qos=matrix (MaxWall 3 days) -- so one submission
# cannot span both. Override on the command line:
#
#   sbatch --partition=dw --qos=matrix --array=0-N%8 sbatch/train_visfilt_1gpu.sh LIST_A
#   sbatch --partition=cs --qos=cs     --array=0-M%8 sbatch/train_visfilt_1gpu.sh LIST_B
#
# **SPLIT the work across two disjoint config lists; do NOT mirror the same list.** Mirroring
# is safe for idempotent jobs (e.g. pose extraction) but two training runs that share a `save:`
# directory will clobber each other's checkpoints -- this has already happened once here
# (13581745 vs 13582050_0). One config, one partition, one job.
#
# dw is usually the better bet: it is 7 A100 nodes with ~2 TiB RAM each, and cs nodes routinely
# sit at 0 GiB free memory with idle GPUs because other jobs have reserved all the RAM. A 1-GPU
# job that pended indefinitely on cs started on dw-1-1 in 8 seconds (13582040 -> 13582044).
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
echo "Stage Run (1 GPU)"
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
