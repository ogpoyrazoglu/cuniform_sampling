# Single-Frame Planning Analysis (`map_conditioning/single_frame_analysis/`)

This is the C-Free-Uniform contribution: the experiment showing the map-conditioned
C-Free-Uniform (CFU) sampler beats C-Uniform on single-frame goal-finding success
rate at a smaller sampling budget. It compares samplers (`C-Free-Uniform`,
`C-Uniform`, `MPPI`, `Log-MPPI`) across a dataset of polygon scenarios.

It depends on the navigation framework controllers and on
`navigation_experiments/post_experiment_analysis_scripts/run_c_safe_uniformity_analysis.py`
(for `check_collision_interpolated_vectorized`), all of which are present in this
repository.

Install once from the repository root:

```bash
pip install -e .
```

## Entry point

### `run_planning_analysis.py`

The script takes a `--config` argument (path relative to the script directory;
defaults to `configs/planning_config.yaml`). Run it from the script directory so
the default resolves:

```bash
cd map_conditioning/single_frame_analysis
python run_planning_analysis.py --config configs/planning_config.yaml
```

## Configuration (`configs/planning_config.yaml`)

This config file is included in the repository. The important fields:

| Field | Meaning |
|---|---|
| `dataset_path` | Scenario dataset folder, **relative to the repository root**. Default `map_conditioning/dataset_polygon_uniformity_localized_gt_dt0.1`. |
| `samplers_to_test` | Which samplers to compare (must match keys in `CONTROLLERS_FACTORY` inside `run_planning_analysis.py`). |
| `sampling_budgets` | List of K values (default `512`). |
| `goal_tolerance` | Success threshold in meters. |
| `num_mc_trials` | Monte Carlo trials per (scenario, sampler, budget). Lower it for a quick run. |
| `visualization_debug` | If true, renders the first trial of each combo (slower). |
| `base_experiment_config_path` | Base controller config, relative to repo root. Default `map_conditioning/navigation_experiments/configs/experiment_config.yaml`. |

## Required data and how to provide it

This analysis needs two kinds of excluded artifacts:

1. **Scenario dataset** (`dataset_path`): the polygon uniformity ground-truth
   folders containing `costmap.npy`, `sdf.npy`, and `uniformity_gt.pkl` per
   scenario. Regenerate with
   `map_conditioning/generate_polygon_uniformity_groundtruth.py`, then set
   `dataset_path` to the folder it produced. (Note the generator's default output
   folder name may differ from the config default — point the config at whatever
   folder you generated.)
2. **Trained model weights** referenced by `base_experiment_config_path`
   (`experiment_config.yaml`): `map_conditioned_model_path`,
   `feature_extractor_path`, and `unsupervised_cuniform_model_path`. These are
   absolute `/home/mikasa/RSN/...` paths in the shipped config and must be updated
   to your locally trained checkpoints (from `supervised_training.py` /
   `unsupervised_training.py`). See [navigation_experiments.md](navigation_experiments.md)
   and [../data_artifacts.md](../data_artifacts.md).

## Quick toy run

For a fast end-to-end check, edit `configs/planning_config.yaml` to use a single
small budget, a small `num_mc_trials`, and `visualization_debug: false`, and point
`dataset_path` at a tiny generated scenario dataset.

See [../data_artifacts.md](../data_artifacts.md) for full regeneration details.
