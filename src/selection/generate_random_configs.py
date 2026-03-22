"""
Generate YAML config files for random joint selection baselines.

Creates configs for each (K, draw) combination, for each dataset,
by modifying the corresponding informed selection config template.

Usage:
    python src/selection/generate_random_configs.py
"""

import yaml
from pathlib import Path


# Base save directory on the cluster
BASE_SAVE = "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models"

# Dataset configs: (config_dir, template_prefix, save_subdir, csv_overrides)
DATASETS = {
    'autsl': {
        'config_dir': 'configs',
        'template_dir': 'configs/informed_selection',
        'save_prefix': f'{BASE_SAVE}/random_selection',
        'csv': {},  # uses defaults from template
    },
    'asl_citizen': {
        'config_dir': 'configs/asl_citizen',
        'template_dir': 'configs/asl_citizen/informed_selection',
        'save_prefix': f'{BASE_SAVE}/asl_citizen/random_selection',
        'csv': {},
    },
    'gsl': {
        'config_dir': 'configs/gsl',
        'template_dir': 'configs/gsl/informed_selection',
        'save_prefix': f'{BASE_SAVE}/gsl/random_selection',
        'csv': {},
    },
    'multilingual': {
        'config_dir': 'configs/multilingual',
        'template_dir': 'configs/multilingual/informed_selection',
        'save_prefix': f'{BASE_SAVE}/multilingual/random_selection',
        'csv': {},
    },
}

K_VALUES = [270, 100, 48, 24, 10]
N_DRAWS = 3


def main():
    indices_dir = Path('data/random_selection')

    for dataset_name, dataset_info in DATASETS.items():
        template_dir = Path(dataset_info['template_dir'])
        output_dir = Path(dataset_info['config_dir']) / 'random_selection'
        output_dir.mkdir(parents=True, exist_ok=True)

        for k in K_VALUES:
            # Load template from corresponding topk config
            template_path = template_dir / f'topk_{k}.yaml'
            if not template_path.exists():
                print(f"  WARNING: Template {template_path} not found, skipping")
                continue

            with open(template_path) as f:
                template = yaml.safe_load(f)

            for draw in range(N_DRAWS):
                config = dict(template)

                # Update paths
                config['save'] = f"{dataset_info['save_prefix']}/random_{k}_draw{draw}"
                config['joint_indices_file'] = f"data/random_selection/random_{k}_draw{draw}_indices.json"

                # Keep same num_pose_points, L0 settings, architecture
                # The model will train with L0 pruning on these random joints

                # Write config
                config_filename = f'random_{k}_draw{draw}.yaml'
                config_path = output_dir / config_filename
                with open(config_path, 'w') as f:
                    f.write(f"# Random selection baseline (K={k}, draw {draw})\n")
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"{dataset_name}: wrote {len(K_VALUES) * N_DRAWS} configs to {output_dir}")


if __name__ == "__main__":
    main()
