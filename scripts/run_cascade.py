"""Run the iterative-cascade landmark selection for one language, end to end.

The cascade descends through target sizes K in {270, 100, 48, 24, 10}. At each step it
retrains the L0 gate *restricted to the previous step's surviving subset*, then keeps that
step's top-K. The point is resolution: a single 543-wide gate has one penalty weight to
separate 543 landmarks, so the tail of its ranking -- exactly the part that determines a
K=24 or K=10 subset -- is its least reliable region. Measured on the 543-wide rankings,
old-vs-new agreement is 44-46/48 at K=48 but only 7-9/10 at K=10. Re-ranking within each
reduced pool gives the gate its full dynamic range to separate the survivors.

The first subset (top-270) comes from the existing 543-wide stage-1 run, so this script
performs 4 gate trainings per language, sequentially -- each depends on the previous one's
output, which is why this is one long job rather than an array.

Usage:
    python scripts/run_cascade.py --lang gsl
    python scripts/run_cascade.py --lang gsl --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys

# Deliberately NO numpy/torch import here. Importing them in the orchestrator leaves
# MKL_THREADING_LAYER=INTEL in os.environ, which the `src/train.py` child inherits and
# which then collides with the libgomp torch loads ("MKL_THREADING_LAYER=INTEL is
# incompatible with libgomp.so.1"). That killed all three chains 30 s in on the first
# attempt (13585411). Ranking extraction is delegated to
# scripts/extract_gate_ranking.py so it runs in its own process.

MODELS = "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models"

LANGS = {
    "autsl": dict(
        gate_base="configs/gate_stage1/autsl.yaml",
        rank="data/informed_selection_hc",
        save=f"{MODELS}/cascade_hc",
    ),
    "asl_citizen": dict(
        gate_base="configs/gate_stage1/asl_citizen.yaml",
        rank="data/asl_citizen/informed_selection_hc",
        save=f"{MODELS}/asl_citizen/cascade_hc",
    ),
    "gsl": dict(
        gate_base="configs/gate_sweep/gsl/hc2_init0.5_lam2_2gpu.yaml",
        rank="data/gsl/informed_selection_hc",
        save=f"{MODELS}/gsl/cascade_hc",
    ),
}

STAGES = [270, 100, 48, 24, 10]


def set_key(text, key, value):
    pattern = rf"^{re.escape(key)}:.*$"
    if re.search(pattern, text, flags=re.M):
        return re.sub(pattern, f"{key}: {value}", text, flags=re.M)
    return text.rstrip() + f"\n{key}: {value}\n"


def emit_config(lang, pool_k, next_k):
    """Write the config for one cascade stage and print its path. Used by the sbatch driver."""
    meta = LANGS[lang]
    cfg_dir = f"configs/cascade_hc/{lang}"
    idx_dir = f"{meta['rank']}/cascade"
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(idx_dir, exist_ok=True)

    pool_file = f"{idx_dir}/iter_{pool_k}_indices.json"
    pool = json.load(open(pool_file))
    if len(pool) != pool_k:
        raise SystemExit(f"pool {pool_file} has {len(pool)} indices, expected {pool_k}")

    s = open(meta["gate_base"]).read()
    s = set_key(s, "save", f"{meta['save']}/iter_{pool_k}")
    s = set_key(s, "joint_pruning", "True")
    s = set_key(s, "n_gpus", 1)
    s = set_key(s, "num_pose_points", pool_k)
    s = set_key(s, "joint_indices_file", pool_file)
    s = set_key(s, "normalization_scope", "global")
    s = set_key(s, "cache_val_skeletons", "true")
    # Gate settings that produced usable rankings (see EXPERIMENT_PROVENANCE.md).
    s = set_key(s, "gate_type", "hard_concrete")
    s = set_key(s, "init_keep_probability", 0.5)
    s = set_key(s, "l0_warmup_steps", 0)
    s = set_key(s, "l0_anneal_steps", 5000)
    s = set_key(s, "l0_end_weight", 2.0)
    s = set_key(s, "max_steps", 40000)
    # Stage-1's product is the ranking, not accuracy, so do not early-stop on val_acc.
    s = set_key(s, "early_stop", 1000)
    s = set_key(s, "seed", 4000)
    s += (f"\n# Cascade stage: gate trained over the {pool_k} landmarks surviving the previous\n"
          f"# stage, to select the top {next_k}. Restricting the pool lets the gate spend its\n"
          f"# full dynamic range separating candidates a 543-wide gate leaves nearly tied.\n")
    cfg_path = f"{cfg_dir}/iter_{pool_k}.yaml"
    open(cfg_path, "w").write(s)
    print(cfg_path)


def seed_pool(lang):
    """Copy the 543-wide top-270 ranking in as the cascade's starting pool."""
    meta = LANGS[lang]
    idx_dir = f"{meta['rank']}/cascade"
    os.makedirs(idx_dir, exist_ok=True)
    src = f"{meta['rank']}/topk/top_270_indices.json"
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}; run the 543-wide stage-1 first")
    json.dump(json.load(open(src)), open(f"{idx_dir}/iter_270_indices.json", "w"))
    print(f"{idx_dir}/iter_270_indices.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(LANGS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed-pool", action="store_true",
                    help="write the starting top-270 pool and exit")
    ap.add_argument("--emit-config", nargs=2, type=int, metavar=("POOL_K", "NEXT_K"),
                    help="write one stage's config, print its path, and exit")
    ap.add_argument("--stages", action="store_true", help="print the stage list and exit")
    args = ap.parse_args()

    if args.stages:
        print(" ".join(str(k) for k in STAGES)); return 0
    if args.seed_pool:
        seed_pool(args.lang); return 0
    if args.emit_config:
        emit_config(args.lang, *args.emit_config); return 0

    meta = LANGS[args.lang]
    cfg_dir = f"configs/cascade_hc/{args.lang}"
    idx_dir = f"{meta['rank']}/cascade"
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(idx_dir, exist_ok=True)

    # Step 0: the 543-wide stage-1 run already produced the top-270 pool.
    pool_file = f"{meta['rank']}/topk/top_270_indices.json"
    assert os.path.exists(pool_file), f"missing {pool_file}; run the 543-wide stage-1 first"
    json.dump(json.load(open(pool_file)), open(f"{idx_dir}/iter_270_indices.json", "w"))
    print(f"[cascade:{args.lang}] stage 270 seeded from the 543-wide ranking", flush=True)

    # Steps 1..4: train the gate on the current pool, keep its top-K_next.
    for pool_k, next_k in zip(STAGES[:-1], STAGES[1:]):
        pool = json.load(open(f"{idx_dir}/iter_{pool_k}_indices.json"))
        assert len(pool) == pool_k, f"pool has {len(pool)} indices, expected {pool_k}"

        name = f"iter_{pool_k}"
        save = f"{meta['save']}/{name}"
        cfg_path = f"{cfg_dir}/{name}.yaml"

        s = open(meta["gate_base"]).read()
        s = set_key(s, "save", save)
        s = set_key(s, "joint_pruning", "True")
        s = set_key(s, "n_gpus", 1)
        s = set_key(s, "num_pose_points", pool_k)
        s = set_key(s, "joint_indices_file", f"{idx_dir}/iter_{pool_k}_indices.json")
        s = set_key(s, "normalization_scope", "global")
        s = set_key(s, "cache_val_skeletons", "true")
        # Gate settings that produced usable rankings (see EXPERIMENT_PROVENANCE.md).
        s = set_key(s, "gate_type", "hard_concrete")
        s = set_key(s, "init_keep_probability", 0.5)
        s = set_key(s, "l0_warmup_steps", 0)
        s = set_key(s, "l0_anneal_steps", 5000)
        s = set_key(s, "l0_end_weight", 2.0)
        s = set_key(s, "max_steps", 40000)
        # Stage-1's product is the ranking, not accuracy, so do not early-stop on val_acc.
        s = set_key(s, "early_stop", 1000)
        s = set_key(s, "seed", 4000)
        s += (f"\n# Cascade stage: gate trained over the {pool_k} landmarks surviving the previous\n"
              f"# stage, to select the top {next_k}. Restricting the pool lets the gate spend its\n"
              f"# full dynamic range separating candidates a 543-wide gate leaves nearly tied.\n")
        open(cfg_path, "w").write(s)

        print(f"[cascade:{args.lang}] stage {pool_k} -> {next_k}: training gate over {pool_k} "
              f"landmarks ({cfg_path})", flush=True)
        if args.dry_run:
            print(f"[cascade:{args.lang}]   (dry run, skipping training)", flush=True)
            json.dump(pool[:next_k], open(f"{idx_dir}/iter_{next_k}_indices.json", "w"))
            continue

        rc = subprocess.call([sys.executable, "src/train.py", "-c", cfg_path, "-m", "TRAIN"])
        if rc != 0:
            print(f"[cascade:{args.lang}] TRAINING FAILED at stage {pool_k} (rc={rc}); "
                  f"stopping the chain", flush=True)
            return rc

        ckpt = f"{save}/checkpoints/last.ckpt"
        if not os.path.exists(ckpt):
            print(f"[cascade:{args.lang}] missing {ckpt}; stopping", flush=True)
            return 1

        rc = subprocess.call([
            sys.executable, "scripts/extract_gate_ranking.py",
            "--checkpoint", ckpt,
            "--pool", f"{idx_dir}/iter_{pool_k}_indices.json",
            "--keep", str(next_k),
            "--out", f"{idx_dir}/iter_{next_k}_indices.json",
        ])
        if rc != 0:
            print(f"[cascade:{args.lang}] SELECTION FAILED at stage {pool_k} (rc={rc}); "
                  f"stopping the chain", flush=True)
            return rc
        print(f"[cascade:{args.lang}] stage {pool_k} -> {next_k} done", flush=True)

    print(f"[cascade:{args.lang}] chain complete; subsets in {idx_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
