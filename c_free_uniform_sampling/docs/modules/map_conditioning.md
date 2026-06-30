# Map-Conditioning Module (`map_conditioning/`)

This module generates the supervised/unsupervised C-Free-Uniform datasets,
trains the map-conditioned model, and runs uniformity/baseline analysis. It
imports across packages: `map_conditioning.utility_helper_map` reaches into
`flow_Cuniform.c_uniform_sampling`/`dynamics_helpers`, and several scripts import
`classes.grid`.

Install once from the repository root:

```bash
pip install -e .
```

Most scripts are configured by editing the `main()` block (no CLI flags). The
recurring knobs are the **dataset name/location**, the **time horizon**, and the
**number of epochs**.

## Directory conventions

- Generated supervised datasets live under `map_conditioning/dataset_supervised/<dataset_name>/`
  (resolved relative to the script via `dataset_utils.get_dataset_files`).
- Each environment folder contains `costmap.npy`, `sdf.npy`, `action_probs.pkl`
  (and optionally `boundary_states_with_probs.pkl`) plus visualization PNGs.
- Training writes a timestamped `map_conditioning/results_<timestamp>/` folder
  containing `models/best_model.pth` and `models/best_feature_extractor.pth`.

## Dataset-generation entry points

### `supervised_dataset_generation.py` — build the supervised action-probability dataset

```bash
python map_conditioning/supervised_dataset_generation.py
```

- **What it does:** for each environment, computes C-Uniform action
  probabilities per state and writes `costmap.npy`, `sdf.npy`, `action_probs.pkl`
  into `dataset_supervised/<output_dataset_dir>/env_*/`.
- **Configure** in `main()`: which generator to call (the default enabled block
  is the BARN dataset, output `dataset_supervised/barn_dataset_vmax2.5`),
  `grid_size`, `resolution`, `num_trajectories`, and the horizon.
- **External input — BARN maps:** the BARN generator loads occupancy grids from a
  BARN dataset directory (`grid_files/grid_<id>.npy` + `map_files/yaml_<id>.yaml`,
  via `barn_dataset_helpers.load_barn_environment`). This directory is **not**
  shipped. Point the generator at your local BARN dataset (see
  [../data_artifacts.md](../data_artifacts.md)).

### `unsupervised_dataset_generation.py` — build the unsupervised dataset / level sets

```bash
python map_conditioning/unsupervised_dataset_generation.py
```

- Produces the per-environment dataset folders and the level-set file used by
  unsupervised training. Configure the environment list, horizon, and output
  directory in `main()`.

### `generate_polygon_uniformity_groundtruth.py` — uniformity ground truth

```bash
python map_conditioning/generate_polygon_uniformity_groundtruth.py
```

- **What it does:** for sampled robot poses in polygon environments, computes the
  localized uniformity ground truth and writes, per environment,
  `costmap.npy`, `sdf.npy`, and `uniformity_gt.pkl` into
  `map_conditioning/dataset_polygon_uniformity_localized_gt_v_1.25_dt0.1_t2.41/poly_env_*/`
  (output dir set in `main()`).
- **Consumed by:** `single_frame_analysis/run_planning_analysis.py` — generate
  this before running that analysis. See
  [single_frame_analysis.md](single_frame_analysis.md).

### `clean_supervised_dataset.py` — clean/post-process a supervised dataset

```bash
python map_conditioning/clean_supervised_dataset.py
```

- Reads an input dataset folder under `dataset_supervised/` and writes a cleaned
  copy. Set `input_dataset_dir` / `output_dataset_dir` in `main()`.

## Training entry points

### `supervised_training.py` — train the map-conditioned CFU model

```bash
python map_conditioning/supervised_training.py
```

- **Reads** the dataset named by `hyperparams['dataset_name']` (default
  `"barn_dataset_vmax2.5"`) from `map_conditioning/dataset_supervised/`. Generate
  that dataset first with `supervised_dataset_generation.py`.
- **Key knobs** in the `hyperparams` dict in `main()`: `dataset_name`,
  `num_epochs` (default `100`; set to `2` for a quick toy run), `batch_size`,
  `learning_rate`, `skip_training`.
- **Writes** `results_<timestamp>/models/best_model.pth` and
  `best_feature_extractor.pth` plus training/test evaluation folders.
- Uses `torch.compile`; first run may take a few minutes to compile.

### `unsupervised_training.py` — unsupervised C-Uniform training

```bash
python map_conditioning/unsupervised_training.py
```

- **Key knobs:** `num_epochs` (default `15`; lower for a toy run) and the input
  data location / output folder.
- **Heads-up — stale absolute paths:** this script reads
  `/home/mikasa/RSN/.../Kinematic_3D_trained_models/fixed_oversampled_level_sets.pkl`
  and uses absolute `base_dir` / `experiment_folder` paths. Update these to your
  local level-set file and dataset/output folders before running. See
  [../data_artifacts.md](../data_artifacts.md).

### `supervised_openspace_training.py` — open-space baseline training

```bash
python map_conditioning/supervised_openspace_training.py
```

- **Heads-up — stale absolute path:** `data_file` points at
  `/home/mikasa/RSN/.../saved_pickles/C_Uniform_processed_...t1.21...steering_31.pkl`.
  Update it to a processed dataset you generated. Produces
  `baseline_openspace_model.pth`.

## Analysis entry points

### `trajectory_uniformity_analysis.py` — analyze trajectory uniformity

```bash
python map_conditioning/trajectory_uniformity_analysis.py
```

- **Heads-up — stale absolute paths:** `data_file_path` and `model_path` (lines
  ~386–388) point at `/home/mikasa/RSN/...`. Set them to a processed dataset
  `.pkl` and the `baseline_openspace_model.pth` you produced. Writes a results
  folder.

### `evaluate_baseline_model.py` — evaluate the open-space baseline

```bash
python map_conditioning/evaluate_baseline_model.py
```

- Loads `map_conditioning/baseline_models/baseline_trajectories.pkl` and tests it
  against the costmaps/SDFs in the `shepherd_dataset_supervised_cleaned` dataset.
  Both the baseline `.pkl` and the dataset must be present/regenerated first.

## What to regenerate

The `.pkl` datasets, `.npy` maps, `.pth`/`.pt` checkpoints, and `results_*`
folders this module consumes are all excluded from the repository. See
[../data_artifacts.md](../data_artifacts.md) for how to regenerate or obtain each
one, and which script produces it.
