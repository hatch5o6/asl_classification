"""Generate the stage-2 config matrix for the ARR resubmission runs.

Four families, all sharing one recipe so every number is comparable:

  topk      K in {543,270,100,48,24,10} x 3 seeds x 3 languages  = 54
  randomk   K in {48,24,10} x 5 draws   x 3 languages            = 45
  normabl   K in {48,24,10} x 3 languages, normalization_scope=subset = 9
  matched   {top48 U lips (88), top88} x 3 languages             = 6

Recipe: `joint_pruning: False` (the subset is fixed, so no gate), seed 4000 unless the
family varies it, max_steps 40000, early_stop 30, n_gpus 1, normalization_scope global
(except normabl). This matches the region-ablation runs, so face-only/lips-only numbers
sit on the same scale as the top-K frontier.

Random draws are shared across languages (one RNG, seeded) so the random-K baseline is
the same subset everywhere and cross-language differences are not draw noise.

Usage:
    python scripts/generate_stage2_configs.py [--dry-run]
"""

import argparse
import json
import os
import re

import numpy as np

MODELS = ("/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models")

# (base config to inherit csvs/architecture from, ranking dir, model save root, short tag)
LANGS = {
    "autsl": dict(
        base="configs/gate_norm/autsl_pergroup.yaml",
        rank="data/informed_selection_hc",
        save=f"{MODELS}/stage2_hc",
        tag="au",
    ),
    "asl_citizen": dict(
        base="configs/gate_norm/asl_citizen_pergroup.yaml",
        rank="data/asl_citizen/informed_selection_hc",
        save=f"{MODELS}/asl_citizen/stage2_hc",
        tag="ac",
    ),
    "gsl": dict(
        base="configs/gate_norm/gsl_pergroup.yaml",
        rank="data/gsl/informed_selection_hc",
        save=f"{MODELS}/gsl/stage2_hc",
        tag="gs",
    ),
}

TOPK_K = [543, 270, 100, 48, 24, 10]
SEEDS = [4000, 4001, 4002]
RANDOM_K = [48, 24, 10]
RANDOM_DRAWS = 5
NORMABL_K = [48, 24, 10]
LIPS = "data/region_subsets/lips_indices.json"

CONFIG_ROOT = "configs/stage2_hc"


def set_key(text, key, value):
    """Set `key: value` in a YAML text blob, appending if absent."""
    pattern = rf"^{re.escape(key)}:.*$"
    if re.search(pattern, text, flags=re.M):
        return re.sub(pattern, f"{key}: {value}", text, flags=re.M)
    return text.rstrip() + f"\n{key}: {value}\n"


def make_config(lang, name, *, k, indices_file, seed, scope, note):
    """Render one stage-2 config from the language's base config."""
    meta = LANGS[lang]
    s = open(meta["base"]).read()
    s = set_key(s, "save", f"{meta['save']}/{name}")
    s = set_key(s, "joint_pruning", "False")
    s = set_key(s, "n_gpus", 1)
    s = set_key(s, "max_steps", 40000)
    s = set_key(s, "early_stop", 30)
    s = set_key(s, "seed", seed)
    s = set_key(s, "num_pose_points", k)
    s = set_key(s, "joint_indices_file", indices_file if indices_file else "null")
    s = set_key(s, "normalization_scope", scope)
    s = set_key(s, "cache_val_skeletons", "true")
    s += f"\n# {note}\n"
    path = f"{CONFIG_ROOT}/{lang}/{name}.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(s)
    return path


def build_random_indices(dry):
    """Shared random draws, one file per (K, draw). Same subset for every language."""
    out = "data/region_subsets/random"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20260904)
    made = {}
    for k in RANDOM_K:
        for draw in range(RANDOM_DRAWS):
            idx = sorted(rng.choice(543, size=k, replace=False).tolist())
            path = f"{out}/random_{k}_draw{draw}.json"
            if not dry:
                json.dump(idx, open(path, "w"))
            made[(k, draw)] = path
    return made


def build_matched_indices(dry):
    """top48 U lips (88 landmarks) and the gate's own top88, per language, at equal K."""
    lips = json.load(open(LIPS))
    made = {}
    for lang, meta in LANGS.items():
        probs = np.genfromtxt(
            f"{meta['rank']}/joint_probabilities.csv", delimiter=",", skip_header=1
        )
        order = np.argsort(probs)[::-1]
        top48 = json.load(open(f"{meta['rank']}/topk/top_48_indices.json"))
        union = sorted(set(top48) | set(lips))
        # Match the budget exactly: the gate's own top-|union| subset.
        topn = sorted(order[: len(union)].tolist())
        d = f"{meta['rank']}/matched"
        os.makedirs(d, exist_ok=True)
        u_path, t_path = f"{d}/top48_union_lips.json", f"{d}/top{len(union)}.json"
        if not dry:
            json.dump(union, open(u_path, "w"))
            json.dump(topn, open(t_path, "w"))
        made[lang] = (len(union), u_path, t_path)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rand_idx = build_random_indices(args.dry_run)
    matched = build_matched_indices(args.dry_run)

    jobs = {"topk": [], "randomk": [], "normabl": [], "matched": []}
    saves = {}

    for lang, meta in LANGS.items():
        tag = meta["tag"]

        # --- topk: the Pareto frontier, 3 seeds ---
        for k in TOPK_K:
            idx = None if k == 543 else f"{meta['rank']}/topk/top_{k}_indices.json"
            for seed in SEEDS:
                name = f"topk{k}_seed{seed}"
                p = make_config(
                    lang, name, k=k, indices_file=idx, seed=seed, scope="global",
                    note=f"Stage-2 top-{k} ({lang}), seed {seed}. Hard-concrete gate ranking.",
                ) if not args.dry_run else f"{CONFIG_ROOT}/{lang}/{name}.yaml"
                jobs["topk"].append((p, f"s2_{tag}t{k}_{seed % 100}"))
                saves.setdefault(f"{meta['save']}/{name}", []).append(p)

        # --- randomk: the lower bound, 5 draws ---
        for k in RANDOM_K:
            for draw in range(RANDOM_DRAWS):
                name = f"random{k}_draw{draw}"
                p = make_config(
                    lang, name, k=k, indices_file=rand_idx[(k, draw)], seed=4000,
                    scope="global",
                    note=f"Random-{k} baseline draw {draw} ({lang}). Draws shared across languages.",
                ) if not args.dry_run else f"{CONFIG_ROOT}/{lang}/{name}.yaml"
                jobs["randomk"].append((p, f"s2_{tag}r{k}_{draw}"))
                saves.setdefault(f"{meta['save']}/{name}", []).append(p)

        # --- normabl: subset-statistics normalization ---
        for k in NORMABL_K:
            name = f"normsub{k}_seed4000"
            p = make_config(
                lang, name, k=k,
                indices_file=f"{meta['rank']}/topk/top_{k}_indices.json",
                seed=4000, scope="subset",
                note=f"Normalization ablation, K={k} ({lang}): statistics pooled over the "
                     f"retained landmarks only. Pairs with topk{k}_seed4000.",
            ) if not args.dry_run else f"{CONFIG_ROOT}/{lang}/{name}.yaml"
            jobs["normabl"].append((p, f"s2_{tag}n{k}"))
            saves.setdefault(f"{meta['save']}/{name}", []).append(p)

        # --- matched: does adding lips at equal budget beat the gate's own choice? ---
        n, u_path, t_path = matched[lang]
        for name, idxf, note in (
            (f"union{n}_seed4000", u_path,
             f"top-48 U lip contour = {n} landmarks ({lang}), equal budget to top{n}."),
            (f"top{n}_seed4000", t_path,
             f"gate's own top-{n} ({lang}), equal budget to top-48 U lips."),
        ):
            p = make_config(
                lang, name, k=n, indices_file=idxf, seed=4000, scope="global", note=note,
            ) if not args.dry_run else f"{CONFIG_ROOT}/{lang}/{name}.yaml"
            jobs["matched"].append((p, f"s2_{tag}m{name[:5]}"))
            saves.setdefault(f"{meta['save']}/{name}", []).append(p)

    dupes = {s: c for s, c in saves.items() if len(c) > 1}
    assert not dupes, f"SAVE DIR COLLISION: {dupes}"

    os.makedirs(CONFIG_ROOT, exist_ok=True)
    for fam, items in jobs.items():
        if not args.dry_run:
            with open(f"{CONFIG_ROOT}/{fam}_configs.txt", "w") as f:
                f.writelines(f"{c}\t{n}\n" for c, n in items)
        print(f"{fam:9} {len(items):>3} runs -> {CONFIG_ROOT}/{fam}_configs.txt")
    print(f"{'TOTAL':9} {sum(len(v) for v in jobs.values()):>3} runs, "
          f"{len(saves)} distinct save dirs (no collisions)")
    for lang, (n, _, _) in matched.items():
        print(f"  matched budget {lang}: {n} landmarks")


if __name__ == "__main__":
    main()
