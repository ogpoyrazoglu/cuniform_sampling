# Closed-Loop Navigation Framework (`map_conditioning/navigation_experiments/`)

The self-contained closed-loop simulator: `EnvironmentsManager` +
`utils/dynamics.py` (integrator) + `utils/geometry_utils.py` (collision) implement
the full sensor → control → dynamics loop in-process. No external simulator
(gym/CARLA/F1Tenth) is required.

> **Runs under `c_uniform` with no extra setup.** All controllers — including the MPPI
> variants (`controllers/mppi_pytorch_controller.py`) — run on **PyTorch**. There is **no
> `pycuda` dependency** and no separate CUDA toolkit install is required. A GPU is used
> automatically when available (`torch.cuda.is_available()`), otherwise the controllers fall
> back to CPU.
>
> ```bash
> conda activate c_uniform
> cd map_conditioning/navigation_experiments
> python run_experiments.py
> ```

This is also the canonical copy of the framework that
`single_frame_analysis/run_planning_analysis.py` and
`post_experiment_analysis_scripts/run_c_safe_uniformity_analysis.py` resolve
against.

Install once from the repository root:

```bash
pip install -e .
```

## Entry points

### `run_experiments.py` — run a closed-loop navigation experiment

```bash
cd map_conditioning/navigation_experiments
python run_experiments.py
```

Optional CLI flags:

| Flag | Meaning |
|---|---|
| `--num-rollouts <int>` | Override the rollout count from the config. |
| `--resume-dir <path>` | Resume a previous experiment from its results directory. |

### `run_multi_experiments.py` — drive multiple experiment suites

```bash
cd map_conditioning/navigation_experiments
python run_multi_experiments.py
```

Orchestrates batches of `run_experiments.py` runs (BARN / Polygon / DumbbellSweep
/ BudgetSweep suites). `resume_manager.py` provides resume support and is invoked
by the multi-experiment driver.

## Configuration (`configs/experiment_config.yaml`)

This config is included in the repository. Notable fields:

| Field | Meaning |
|---|---|
| `controllers_to_test` | Which controllers to run (e.g. `MPPI`, `LOGMPPI`, `cuniform_map_conditioned`, `cuniform_unsupervised_openspace`). |
| `horizon_T`, `dt` | Planning horizon and simulation timestep. |
| `max_steps` | Number of simulation steps per rollout. |
| `num_rollouts` | Control samples per step. |
| `experiment_suite.*.dataset_path` | Map dataset folders (relative to the `navigation_experiments` directory), e.g. `data/BARN_npy_files_0.05_padded`, `data/Minimum_Feature_25_Dataset_A_Star_Generated`. |
| `cuniform_controller.map_conditioned_model_path` etc. | Paths to trained model weights. |

> The shipped defaults are `horizon_T: 1.2` and `max_steps: 1000`. Adjust these
> (and the other fields above) for your own runs.

### Generating test maps

The map datasets referenced by `dataset_path` are excluded. You can:

- Generate synthetic test maps with `data/create_test_maps.py`:
  ```bash
  python map_conditioning/navigation_experiments/data/create_test_maps.py
  ```
- Convert BARN maps to the expected `.npy` format with `data/convert_barn_maps.py`
  (edit its input/output directories), then visualize with
  `data/visualize_converted_maps.py`.

See [../data_artifacts.md](../data_artifacts.md) for details on each map dataset
folder and the model checkpoints.

## Stale absolute paths to update

Before running the neural C-Uniform controllers, update these absolute
`/home/mikasa/RSN/...` paths in `configs/experiment_config.yaml` to your local
trained checkpoints:

| Field | Replace with |
|---|---|
| `map_conditioned_model_path` (~line 203) | your `results_<timestamp>/models/best_model.pth` from `supervised_training.py` |
| `feature_extractor_path` (~line 211) | your `results_<timestamp>/models/best_feature_extractor.pth` |
| `unsupervised_cuniform_model_path` (~line 214) | your `*_best_model_single_env.pt` from `unsupervised_training.py` |

(The `MPPI` / `LOGMPPI` controllers do not need a trained model, so a closed-loop
run using only those works without any weights.)

## Post-experiment analysis

### `post_experiment_analysis_scripts/run_c_safe_uniformity_analysis.py`

Exports `check_collision_interpolated_vectorized` (used by single-frame
analysis) and can be run directly to analyze C-safe uniformity. It accepts:

| Flag | Meaning |
|---|---|
| `--input_dir <path>` | Directory of per-environment ground-truth subfolders. |
| `--open_space_gt <path>` | Master open-space ground-truth `.pkl`. |

The `--open_space_gt` default is an absolute `/home/mikasa/RSN/...` processed
dataset path; override it on the command line with a processed dataset you
generated (see [../data_artifacts.md](../data_artifacts.md)).
