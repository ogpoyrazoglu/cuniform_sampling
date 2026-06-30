# Data-Artifact Regeneration Guide

No large data files or trained weights ship with this repository. Every dataset
(`.pkl`/`.pickle`), map array (`.npy`), and model checkpoint (`.pth`/`.pt`) is
**excluded by design** and must be regenerated locally (or obtained and placed
where a script expects it). This document lists each excluded artifact, the
script that produces it, where to place it, and any hardcoded path you must
update first.

All relative paths are from the repository root (`c_free_uniform_sampling/`).
Run `pip install -e .` once from the root before regenerating anything.

## Quick reference

| Artifact | Produced by | Consumed by |
|---|---|---|
| `C_Uniform_reachability_*.pkl` | `flow_Cuniform/c_uniform_sampling.py` | `c_uniform_sampling.py` (cache) |
| `C_Uniform_processed_*.pkl` | `flow_Cuniform/c_uniform_sampling.py` | `coverage_analysis.py`, training/analysis scripts |
| `C_Uniform_<N>_trajectories_*.pkl` | `flow_Cuniform/c_uniform_sampling.py` | `coverage_analysis.py`, `animation.py` |
| `uniform_sampled_actions_trajectories_*.pickle` | `c_uniform_sampling.py` (`UNIFORM_ACTION_TRAJ=True`) | `coverage_analysis.py` |
| `costmap.npy`, `sdf.npy` | `supervised_dataset_generation.py`, `generate_polygon_uniformity_groundtruth.py` | training, single-frame analysis, navigation |
| `action_probs.pkl`, `boundary_states_with_probs.pkl` | `supervised_dataset_generation.py` | `supervised_training.py` |
| `level_sets.pkl` / `fixed_oversampled_level_sets.pkl` | sampling / `unsupervised_dataset_generation.py` | `unsupervised_dataset_generation.py`, `unsupervised_training.py` |
| `uniformity_gt.pkl` | `generate_polygon_uniformity_groundtruth.py` | `single_frame_analysis/run_planning_analysis.py` |
| `best_model.pth`, `best_feature_extractor.pth` | `supervised_training.py` | navigation, single-frame analysis |
| `*_best_model.pt`, `*_best_model_single_env.pt` | `unsupervised_training.py` | navigation, single-frame analysis |
| `baseline_openspace_model.pth`, `baseline_trajectories.pkl` | `supervised_openspace_training.py` | `trajectory_uniformity_analysis.py`, `evaluate_baseline_model.py` |
| `results_<timestamp>/` folders | training / experiment scripts | analysis |
| BARN map dataset (`grid_files/`, `map_files/`) | external BARN dataset | `supervised_dataset_generation.py` (BARN), navigation |
| navigation map folders (`data/...`) | `data/create_test_maps.py` / `data/convert_barn_maps.py` | `run_experiments.py` |

---

## 1. Core sampling datasets (root `.pkl` / `.pickle`)

**Producer:** `flow_Cuniform/c_uniform_sampling.py` (run from the directory where
you want the files written — they are saved into the current working directory).

```bash
python flow_Cuniform/c_uniform_sampling.py
```

The filenames encode the run settings, e.g.
`C_Uniform_reachability_disjoint_1_init_<MODEL>_obs_none_perturb_<p>_seed_<s>_grid_<...>_t<total_t>_ts<dt>_vrange_<...>_steering_<n>.pkl`.

- **Reachability** (`C_Uniform_reachability_*.pkl`) and **processed**
  (`C_Uniform_processed_*.pkl`) files are cached: re-running with identical
  settings re-uses them.
- **Trajectory** files (`C_Uniform_<N>_trajectories_*.pkl`) are the sampled
  output, controlled by `num_trajectories_list`.
- To produce `uniform_sampled_actions_trajectories_<N>.pickle`, set
  `UNIFORM_ACTION_TRAJ = True` in `c_uniform_sampling.py`.

**Placement / pointing scripts at them:**

- `flow_Cuniform/coverage_analysis.py` — set the trajectory/processed filenames in
  `main()`. Also update the stale absolute path on **line ~359**:
  `uniform_action_file = "/home/mikasa/RSN/traj_sampling/flow_Cuniform/uniform_sampled_actions_trajectories_10000.pickle"`
  → the `.pickle` you generated locally.
- `flow_Cuniform/animation.py` — set the input trajectory filename in `main()`.

## 2. Map arrays: `costmap.npy` and `sdf.npy`

These per-environment arrays are written into each generated environment folder.

**Producers:**

- `map_conditioning/supervised_dataset_generation.py` → writes `costmap.npy`,
  `sdf.npy`, `action_probs.pkl` (and optionally `boundary_states_with_probs.pkl`)
  into `map_conditioning/dataset_supervised/<output_dataset_dir>/env_*/`.
- `map_conditioning/generate_polygon_uniformity_groundtruth.py` → writes
  `costmap.npy`, `sdf.npy`, `uniformity_gt.pkl` into
  `map_conditioning/dataset_polygon_uniformity_localized_gt_*/poly_env_*/`.

**Optional single-map SDF for core sampling:** `flow_Cuniform/c_uniform_sampling.py`
has a commented `sdf = np.load(".../sdf.npy")` line (~line 575). To sample inside a
map, uncomment it and point it at an `sdf.npy` you generated.

## 3. BARN map dataset (external input)

`supervised_dataset_generation.py`'s BARN path and the navigation BARN suite read
a BARN dataset directory containing `grid_files/grid_<id>.npy` and
`map_files/yaml_<id>.yaml` (loaded by
`map_conditioning/barn_dataset_helpers.py::load_barn_environment`).

- This dataset is **not** shipped. Obtain the BARN dataset (the public BARN
  navigation benchmark) and place it locally, then point the generator at it
  (the BARN branch in `supervised_dataset_generation.py::main()` uses a default
  absolute path construction — set it to your BARN directory).
- For navigation, convert BARN grids into the expected `.npy` map folder with
  `map_conditioning/navigation_experiments/data/convert_barn_maps.py`.

## 4. Level-set files

- `level_sets.pkl` is produced as part of the unsupervised pipeline
  (`map_conditioning/unsupervised_dataset_generation.py`).
- `unsupervised_training.py` loads
  `/home/mikasa/RSN/.../Kinematic_3D_trained_models/fixed_oversampled_level_sets.pkl`
  (**line ~100**) — update this to your local level-set `.pkl`. Also update its
  absolute `base_dir` (**line ~824**) and `experiment_folder` (**line ~841**) to a
  local dataset folder and a local output folder.

## 5. `uniformity_gt.pkl` (single-frame analysis input)

**Producer:** `map_conditioning/generate_polygon_uniformity_groundtruth.py`
(output folder set in `main()`, default
`dataset_polygon_uniformity_localized_gt_v_1.25_dt0.1_t2.41/`).

```bash
python map_conditioning/generate_polygon_uniformity_groundtruth.py
```

**Consumer:** `map_conditioning/single_frame_analysis/run_planning_analysis.py`.
Set `dataset_path` in `single_frame_analysis/configs/planning_config.yaml` to the
folder you generated (path is relative to the repository root). See
[modules/single_frame_analysis.md](modules/single_frame_analysis.md).

## 6. Model checkpoints (`.pth` / `.pt`) and `results_*` folders

### Supervised map-conditioned model

**Producer:** `map_conditioning/supervised_training.py` → writes
`map_conditioning/results_<timestamp>/models/best_model.pth` and
`best_feature_extractor.pth`.

```bash
python map_conditioning/supervised_training.py   # set num_epochs small for a quick run
```

It reads the dataset named by `hyperparams['dataset_name']` (default
`barn_dataset_vmax2.5`) from `map_conditioning/dataset_supervised/`, so generate
that dataset first (section 2).

### Unsupervised C-Uniform model

**Producer:** `map_conditioning/unsupervised_training.py` → produces
`*_best_model.pt` / `*_best_model_single_env.pt` (after you fix the absolute paths
in section 4).

### Open-space baseline

**Producer:** `map_conditioning/supervised_openspace_training.py` → produces
`baseline_openspace_model.pth`. Update its `data_file` absolute path (**line ~129**)
to a processed dataset you generated.

### Where the checkpoints are consumed (paths to update)

`map_conditioning/navigation_experiments/configs/experiment_config.yaml` holds
absolute `/home/mikasa/RSN/...` checkpoint paths that must be repointed to your
local files:

| Field (~line) | Replace with |
|---|---|
| `map_conditioned_model_path` (~203) | your `results_<timestamp>/models/best_model.pth` |
| `feature_extractor_path` (~211) | your `results_<timestamp>/models/best_feature_extractor.pth` |
| `unsupervised_cuniform_model_path` (~214) | your `*_best_model_single_env.pt` |

`map_conditioning/trajectory_uniformity_analysis.py` (**lines ~386–388**) also has
absolute `data_file_path` and `model_path`; set them to a processed dataset `.pkl`
and your `baseline_openspace_model.pth`.

`map_conditioning/evaluate_baseline_model.py` loads
`map_conditioning/baseline_models/baseline_trajectories.pkl` and the
`shepherd_dataset_supervised_cleaned` dataset — regenerate/obtain both and place
them where the script expects.

## 7. Navigation map dataset folders

`run_experiments.py` reads map folders named under `experiment_config.yaml`'s
`experiment_suite.*.dataset_path`, relative to the `navigation_experiments`
directory, e.g.:

- `data/BARN_npy_files_0.05_padded` (BARN suite)
- `data/Minimum_Feature_25_Dataset_A_Star_Generated` (Polygon suite)

These folders are excluded. Generate synthetic maps with
`navigation_experiments/data/create_test_maps.py`, or convert BARN maps with
`navigation_experiments/data/convert_barn_maps.py`, then set `dataset_path` to the
folder you produced. The `MPPI`/`LOGMPPI` controllers need no trained weights, so
a closed-loop smoke run is possible with just a generated map folder.

---

## Summary of stale absolute paths to fix

All originate from the original research machine (`/home/mikasa/RSN/...`) and
point at excluded artifacts. Update each to a locally regenerated file:

| File | Line | What to repoint |
|---|---|---|
| `flow_Cuniform/coverage_analysis.py` | ~359 | `uniform_action_file` → local `uniform_sampled_actions_trajectories_*.pickle` |
| `map_conditioning/trajectory_uniformity_analysis.py` | ~386 | `data_file_path` → local processed `.pkl` |
| `map_conditioning/trajectory_uniformity_analysis.py` | ~387 | `model_path` → local `baseline_openspace_model.pth` |
| `map_conditioning/supervised_openspace_training.py` | ~129 | `data_file` → local processed `.pkl` |
| `map_conditioning/unsupervised_training.py` | ~100 | level-set `.pkl` → local `fixed_oversampled_level_sets.pkl` |
| `map_conditioning/unsupervised_training.py` | ~824 | `base_dir` → local dataset folder |
| `map_conditioning/unsupervised_training.py` | ~841 | `experiment_folder` → local output folder |
| `navigation_experiments/post_experiment_analysis_scripts/run_c_safe_uniformity_analysis.py` | ~1019 | `--open_space_gt` default → local processed `.pkl` (or pass `--open_space_gt` on the CLI) |
| `navigation_experiments/configs/experiment_config.yaml` | ~203 | `map_conditioned_model_path` → local `best_model.pth` |
| `navigation_experiments/configs/experiment_config.yaml` | ~211 | `feature_extractor_path` → local `best_feature_extractor.pth` |
| `navigation_experiments/configs/experiment_config.yaml` | ~214 | `unsupervised_cuniform_model_path` → local `*_best_model_single_env.pt` |

(There are additional commented-out absolute paths in these files; they have no
runtime effect.)
