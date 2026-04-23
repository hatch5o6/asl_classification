"""Generate encoder comparison YAML configs for all language x encoder x K combinations.

Generates two config types:
  1. Informed selection (topk): language-specific L0-learned joint subsets
  2. Random baselines: shared random joint draws with joint_pruning=true, l0_end_weight=0.0

Re-run this script after updating ENCODER_PARAMS with Optuna best values.

Usage:
    python scripts/generate_encoder_configs.py
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGES = {
    "autsl": {
        "train_csv": "data/train.csv",
        "val_csv":   "data/val.csv",
        "test_csv":  "data/test.csv",
        "class_id_csv": "data/SignList_ClassId_TR_EN.csv",
        "topk_dir": "data/informed_selection/topk",
        "iterative_dir": "data/informed_selection/iterative",
        "iterative_k": [10, 24, 48, 100, 270],
        "max_steps": 200000,
    },
    "asl_citizen": {
        "train_csv": "data/asl_citizen/train.csv",
        "val_csv":   "data/asl_citizen/val.csv",
        "test_csv":  "data/asl_citizen/test.csv",
        "class_id_csv": "data/asl_citizen/class_ids.csv",
        "topk_dir": "data/asl_citizen/informed_selection/topk",
        "iterative_dir": "data/asl_citizen/informed_selection/iterative",
        "iterative_k": [10, 24, 48, 100],
        "max_steps": 400000,  # 2731 classes, ~15 samples/class — needs more steps
    },
    "gsl": {
        "train_csv": "data/gsl/train.csv",
        "val_csv":   "data/gsl/val.csv",
        "test_csv":  "data/gsl/test.csv",
        "class_id_csv": "data/gsl/class_ids.csv",
        "topk_dir": "data/gsl/informed_selection/topk",
        "iterative_dir": "data/gsl/informed_selection/iterative",
        "iterative_k": [10, 24, 48, 100],
        "max_steps": 200000,
    },
    "multilingual": {
        "train_csv": "data/multilingual/train.csv",
        "val_csv":   "data/multilingual/val.csv",
        "test_csv":  "data/multilingual/test.csv",
        "class_id_csv": "data/multilingual/class_ids.csv",
        "topk_dir": "data/multilingual/informed_selection/topk",
        "iterative_dir": "data/multilingual/informed_selection/iterative",
        "iterative_k": [10, 24, 48, 100],
        "max_steps": 500000,  # 103K samples, 1614 steps/epoch — needs more steps
    },
}

TOPK_K_VALUES = [10, 24, 48, 100]       # topk 270 is same as iterative 270 — skip
RANDOM_K_VALUES = [10, 24, 48, 100, 270]
N_RANDOM_DRAWS = 3

# ─────────────────────────────────────────────────────────────────────────────
# Encoder-specific params — UPDATE WITH OPTUNA BEST PARAMS
# ─────────────────────────────────────────────────────────────────────────────

ENCODER_PARAMS = {
    # BERT excluded — results already exist from informed selection experiments
    # GRU — Optuna best: trial 3, val_acc=79.82%
    "gru": {
        "skeleton_encoder": "gru",
        "params": {
            "gru_hidden_dim": 512,
            "gru_num_layers": 3,
            "gru_dropout": 0.2,
            "gru_bidirectional": True,
        },
        "skel_learning_rate": "5.235e-05",
        "classifier_dropout": 0.3,
    },
    # ST-GCN — Optuna best: trial 11, val_acc=54.82%
    "stgcn": {
        "skeleton_encoder": "stgcn",
        "params": {
            "stgcn_channels": [64, 128, 256, 512],
            "stgcn_kernel_size": 13,
            "stgcn_edges_file": "data/mediapipe_edges.json",
        },
        "skel_learning_rate": "4.737e-04",
        "classifier_dropout": 0.1,
    },
    # SPOTER — Optuna best: trial 18, val_acc=68.57%
    "spoter": {
        "skeleton_encoder": "spoter",
        "params": {
            "spoter_hidden_dim": 256,
            "spoter_spatial_layers": 1,
            "spoter_temporal_layers": 4,
            "spoter_att_heads": 8,
            "spoter_dropout": 0.1,
        },
        "skel_learning_rate": "2.084e-04",
        "classifier_dropout": 0.2,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared training settings
# ─────────────────────────────────────────────────────────────────────────────

SHARED = {
    "seed": 4000,
    "num_frames": "video_mae",
    "n_gpus": 8,
    "device": "cuda",
    "pretrained_learning_rate": "1e-04",
    "new_learning_rate": "2e-04",
    "class_learning_rate": "2e-04",
    "weight_decay": 0.01,
    "effective_batch_size": 64,
    "early_stop": 30,
    "save_top_k": 3,
    "val_interval": 1.0,
    "max_steps": 200000,
    "fusion_dim": 512,
    "depth_image_size": 224,
    "depth_hidden_dim": 384,
    "depth_hidden_layers": 6,
    "depth_att_heads": 6,
    "depth_intermediate_size": 1536,
    "depth_patch_size": 16,
    "label_smoothing": 0.1,
    "gradient_clip_val": 1.0,
}

# L0 params for random baselines (pruning present but disabled)
L0_PARAMS = {
    "init_keep_probability": 0.98,
    "use_random_init": False,
    "l0_warmup_steps": 10000,
    "l0_anneal_steps": 40000,
    "l0_end_weight": 0.0,
    "temp_start": 10.0,
    "temp_end": 0.01,
    "temp_anneal_steps": 50000,
}

SAVE_ROOT = "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/encoder_comparison"
CONFIG_ROOT = Path("configs/encoder_comparison")


def format_val(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, list):
        return repr(v)
    return v


def generate_config(lang, encoder, k, selection_type="topk", draw=None):
    """Generate a single YAML config string.

    Args:
        selection_type: "topk", "iterative", "random", or "full"
        draw: random draw index (0, 1, 2) — only used for selection_type="random"
    """
    lang_cfg = LANGUAGES[lang]
    enc_cfg = ENCODER_PARAMS[encoder]

    is_random = selection_type == "random"
    is_full = selection_type == "full"
    is_iterative = selection_type == "iterative"

    # Joint selection paths
    if is_full:
        joint_indices_file = "null"
        num_pose_points = 543
        save_suffix = f"{lang}/{encoder}/k543"
        comment = f"Encoder comparison: {encoder.upper()} @ K=543 ({lang.upper()}) — full skeleton"
    elif is_random:
        joint_indices_file = f"data/random_selection/random_{k}_draw{draw}_indices.json"
        num_pose_points = k
        save_suffix = f"{lang}/{encoder}/random_{k}_draw{draw}"
        comment = f"Random baseline: {encoder.upper()} @ K={k} draw {draw} ({lang.upper()})"
    elif is_iterative:
        joint_indices_file = f"{lang_cfg['iterative_dir']}/iter_{k}_indices.json"
        num_pose_points = k
        save_suffix = f"{lang}/{encoder}/iter_{k}"
        comment = f"Iterative selection: {encoder.upper()} @ K={k} ({lang.upper()})"
    else:
        joint_indices_file = f"{lang_cfg['topk_dir']}/top_{k}_indices.json"
        num_pose_points = k
        save_suffix = f"{lang}/{encoder}/k{k}"
        comment = f"Top-K selection: {encoder.upper()} @ K={k} ({lang.upper()})"

    save_path = f"{SAVE_ROOT}/{save_suffix}"

    lines = []
    lines.append(f"# {comment}")
    lines.append(f"save: {save_path}")
    lines.append("test_checkpoint: null")
    lines.append('modalities: ["skeleton"]')
    lines.append(f"joint_pruning: {'true' if is_random else 'False'}")
    lines.append(f"skeleton_encoder: {enc_cfg['skeleton_encoder']}")
    lines.append("")

    # Data paths
    lines.append(f"train_csv: {lang_cfg['train_csv']}")
    lines.append(f"val_csv: {lang_cfg['val_csv']}")
    lines.append(f"test_csv: {lang_cfg['test_csv']}")
    lines.append(f"class_id_csv: {lang_cfg['class_id_csv']}")
    lines.append("")

    # Shared settings
    lines.append(f"seed: {SHARED['seed']}")
    lines.append(f"num_frames: {SHARED['num_frames']}")
    lines.append(f"n_gpus: {SHARED['n_gpus']}")
    lines.append(f"device: {SHARED['device']}")
    lines.append("")

    # Learning rates
    lines.append("# Learning rates")
    lines.append(f"pretrained_learning_rate: {SHARED['pretrained_learning_rate']}")
    lines.append(f"new_learning_rate: {SHARED['new_learning_rate']}")
    lines.append(f"skel_learning_rate: {enc_cfg['skel_learning_rate']}")
    lines.append(f"class_learning_rate: {SHARED['class_learning_rate']}")
    lines.append(f"weight_decay: {SHARED['weight_decay']}")
    lines.append("")

    # Training schedule
    lines.append("# Training schedule")
    lines.append(f"effective_batch_size: {SHARED['effective_batch_size']}")
    lines.append(f"early_stop: {SHARED['early_stop']}")
    lines.append(f"save_top_k: {SHARED['save_top_k']}")
    lines.append(f"val_interval: {SHARED['val_interval']}")
    lines.append(f"max_steps: {lang_cfg.get('max_steps', SHARED['max_steps'])}")
    lines.append("")

    # Architecture (shared)
    lines.append("# Architecture")
    lines.append(f"fusion_dim: {SHARED['fusion_dim']}")
    lines.append(f"depth_image_size: {SHARED['depth_image_size']}")
    lines.append(f"depth_hidden_dim: {SHARED['depth_hidden_dim']}")
    lines.append(f"depth_hidden_layers: {SHARED['depth_hidden_layers']}")
    lines.append(f"depth_att_heads: {SHARED['depth_att_heads']}")
    lines.append(f"depth_intermediate_size: {SHARED['depth_intermediate_size']}")
    lines.append(f"depth_patch_size: {SHARED['depth_patch_size']}")
    lines.append("")

    # Encoder-specific params
    lines.append(f"# {encoder.upper()} Skeleton Encoder")
    for param_key, param_val in enc_cfg["params"].items():
        lines.append(f"{param_key}: {format_val(param_val)}")
    lines.append("")

    # Joint selection
    lines.append("# Joint selection")
    lines.append(f"num_pose_points: {num_pose_points}")
    lines.append(f"joint_indices_file: {joint_indices_file}")
    lines.append("")

    # L0 params for random baselines
    if is_random:
        lines.append("# L0 pruning (disabled via l0_end_weight=0.0)")
        for l0_key, l0_val in L0_PARAMS.items():
            lines.append(f"{l0_key}: {format_val(l0_val)}")
        lines.append("")

    # Final layers
    lines.append(f"classifier_dropout: {enc_cfg['classifier_dropout']}")
    lines.append(f"label_smoothing: {SHARED['label_smoothing']}")
    lines.append(f"gradient_clip_val: {SHARED['gradient_clip_val']}")

    return "\n".join(lines) + "\n"


def main():
    topk_count = 0
    iterative_count = 0
    full_count = 0
    random_count = 0

    for lang in LANGUAGES:
        lang_cfg = LANGUAGES[lang]
        for encoder in ENCODER_PARAMS:
            enc_dir = CONFIG_ROOT / lang / encoder
            enc_dir.mkdir(parents=True, exist_ok=True)

            # Top-K informed selection configs
            for k in TOPK_K_VALUES:
                content = generate_config(lang, encoder, k, selection_type="topk")
                (enc_dir / f"k{k}.yaml").write_text(content)
                topk_count += 1

            # Iterative informed selection configs
            for k in lang_cfg["iterative_k"]:
                content = generate_config(lang, encoder, k, selection_type="iterative")
                (enc_dir / f"iter_{k}.yaml").write_text(content)
                iterative_count += 1

            # Full skeleton (K=543)
            content = generate_config(lang, encoder, 543, selection_type="full")
            (enc_dir / "k543.yaml").write_text(content)
            full_count += 1

            # Random baseline configs
            random_dir = enc_dir / "random"
            random_dir.mkdir(parents=True, exist_ok=True)
            for k in RANDOM_K_VALUES:
                for draw in range(N_RANDOM_DRAWS):
                    content = generate_config(lang, encoder, k, selection_type="random", draw=draw)
                    (random_dir / f"random_{k}_draw{draw}.yaml").write_text(content)
                    random_count += 1

    total = topk_count + iterative_count + full_count + random_count
    print(f"Generated {total} configs under {CONFIG_ROOT}/")
    print(f"  Top-K informed:     {topk_count}")
    print(f"  Iterative informed: {iterative_count}")
    print(f"  Full skeleton:      {full_count}")
    print(f"  Random baseline:    {random_count}")
    print(f"\n  Languages: {list(LANGUAGES.keys())}")
    print(f"  Encoders:  {list(ENCODER_PARAMS.keys())}")
    print(f"  K values (topk):      {TOPK_K_VALUES} + [543]")
    print(f"  K values (random):    {RANDOM_K_VALUES}")
    print(f"  K values (iterative): per-language (see LANGUAGES dict)")
    print(f"  Random draws: {N_RANDOM_DRAWS}")


if __name__ == "__main__":
    main()
