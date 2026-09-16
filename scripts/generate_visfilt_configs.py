"""
Generate stage-2 configs for retraining on visibility-filtered landmark subsets.

Each config reuses the corresponding informed-selection config for its dataset and
overrides only what the corrected run changes: the landmark index file, the subset
size, the seed, and the output directory. Everything else (architecture, optimizer,
schedule) is inherited so the corrected runs stay comparable to the originals.

Also emits a tab-separated config list for the job-array submitter.

Usage:
    python scripts/generate_visfilt_configs.py
    python scripts/generate_visfilt_configs.py --k 48 24 10 --seeds 4000 4001 4002
"""

import argparse
from pathlib import Path

ARCHIVE = "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models"

# dataset -> (template config, corrected-indices dir, archive subdir, short tag)
DATASETS = {
    "autsl": ("configs/informed_selection/topk_{k}.yaml",
              "data/informed_selection_visfilt/topk",
              "", "au"),
    "gsl": ("configs/gsl/informed_selection/topk_{k}.yaml",
            "data/gsl/informed_selection_visfilt/topk",
            "gsl/", "gs"),
    "asl_citizen": ("configs/asl_citizen/informed_selection/topk_{k}.yaml",
                    "data/asl_citizen/informed_selection_visfilt/topk",
                    "asl_citizen/", "ac"),
}

# Template K to borrow settings from when the target K has no direct template.
TEMPLATE_FALLBACK = {543: 270}


def overwrite_key(lines, key, value):
    """Replace a top-level `key: ...` line, appending it if absent."""
    prefix = f"{key}:"
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{key}: {value}"
            return lines
    lines.append(f"{key}: {value}")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate visibility-filtered stage-2 configs"
    )
    parser.add_argument("--k", type=int, nargs="+",
                        default=[543, 270, 100, 48, 24, 10],
                        help="Subset sizes to generate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[4000, 4001, 4002],
                        help="Random seeds (default: the original 4000 plus two)")
    parser.add_argument("--output-dir", type=str, default="configs/visfilt",
                        help="Where to write configs")
    parser.add_argument("--list-out", type=str,
                        default="configs/visfilt/visfilt_configs.txt",
                        help="Where to write the job-array config list")
    args = parser.parse_args()

    entries = []
    for ds, (tpl, idx_dir, archive_sub, tag) in DATASETS.items():
        for k in args.k:
            template = Path(tpl.format(k=TEMPLATE_FALLBACK.get(k, k)))
            if not template.exists():
                print(f"  SKIP {ds} k={k}: no template at {template}")
                continue
            base = template.read_text().splitlines()

            for seed in args.seeds:
                lines = list(base)
                name = f"visfilt_top{k}_seed{seed}"
                save = f"{ARCHIVE}/{archive_sub}informed_selection_visfilt/{name}"

                lines = overwrite_key(lines, "save", save)
                lines = overwrite_key(lines, "seed", seed)
                lines = overwrite_key(lines, "num_pose_points", k)
                # Validation clips are deterministic; memoizing them removes ~half of
                # ASL Citizen's clip preprocessing. See RGBDSkel_Dataset.cache_skeletons.
                lines = overwrite_key(lines, "cache_val_skeletons", "true")

                if k == 543:
                    # Unpruned baseline: no landmark subset, no gate.
                    lines = overwrite_key(lines, "joint_indices_file", "null")
                    lines = overwrite_key(lines, "joint_pruning", "False")
                else:
                    lines = overwrite_key(
                        lines, "joint_indices_file", f"{idx_dir}/top_{k}_indices.json"
                    )
                    lines = overwrite_key(lines, "joint_pruning", "True")

                out_dir = Path(args.output_dir) / ds
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"top{k}_seed{seed}.yaml"
                out_file.write_text("\n".join(lines) + "\n")
                entries.append((str(out_file), f"vf_{tag}{k}_s{seed}"))

    list_path = Path(args.list_out)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("".join(f"{c}\t{n}\n" for c, n in entries))

    print(f"Wrote {len(entries)} configs under {args.output_dir}/")
    print(f"Wrote config list -> {list_path}")
    print(f"\nSubmit with:")
    print(f"  sbatch --array=0-{len(entries) - 1}%8 "
          f"sbatch/train_visfilt_array.sh {list_path}")


if __name__ == "__main__":
    main()
