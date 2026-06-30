# Core C-Uniform Sampling (`flow_Cuniform/`)

This module generates the reachability sets and runs the Max-Flow C-Uniform
sampler, plus coverage analysis and visualization. It depends on the
`classes/` package (`classes.grid`, `classes.graph_structure`).

Run the editable install once from the repository root before using anything
here:

```bash
pip install -e .
```

## Entry points

All scripts are configured by editing constants/dicts inside their `main()` /
`if __name__ == "__main__"` block (there are no CLI flags). The most relevant
knobs are the model selection and the time horizon.

### `c_uniform_sampling.py` — reachability sets + C-Uniform trajectory sampling

```bash
python flow_Cuniform/c_uniform_sampling.py
```

- **What it does:** builds (or loads) the reachability level sets, runs the
  ortools max-flow C-Uniform solve, then samples trajectories from the resulting
  flow network and analyzes their distribution.
- **Key settings** (top of `main()` and the per-model `model_configs` dict):
  - `MODEL` — dynamics model (e.g. `KS_3D_STEERING_ANGLE`, `DUBINS`, ...).
  - `config["dt"]` and `config["total_t"]` — time step and horizon. The number
    of level sets is `total_t / dt`. For a fast toy run set a short horizon
    (e.g. `total_t` ≈ `0.4`, `dt` = `0.2` → 2 level sets).
  - `num_trajectories_list` — how many trajectories to sample.
  - `seed`, `DISJOINT_LEVEL_SET`, `MULTIPLE_INITIAL_CONFIG`.
- **Inputs consumed:** none required for the default open-space run (`obstacles`
  and `sdf` are `None`). An optional `sdf.npy` map can be loaded if you uncomment
  and point the `sdf = np.load(...)` line at a real SDF (see
  [../data_artifacts.md](../data_artifacts.md)).
- **Outputs produced (written to the current working directory):**
  - `C_Uniform_reachability_*.pkl` — cached reachability level sets.
  - `C_Uniform_processed_*.pkl` — pruned graph + max-flow result.
  - `C_Uniform_<N>_trajectories_*.pkl` — sampled trajectory set.
  - Optionally `uniform_sampled_actions_trajectories_<N>.pickle` if
    `UNIFORM_ACTION_TRAJ` is enabled.
  - The exact filenames encode the model, grid, horizon and velocity settings.

> Tip: because the reachability and processed files are cached by filename, a
> re-run with the same settings re-uses them instead of recomputing.

### `coverage_analysis.py` — coverage / uniformity analysis (single horizon)

```bash
python flow_Cuniform/coverage_analysis.py
```

- **Consumes** the trajectory/processed `.pkl` files produced by
  `c_uniform_sampling.py`. The filenames it loads are set inside `main()`.
- **Heads-up — stale absolute path:** `coverage_analysis.py` references
  `uniform_action_file = "/home/mikasa/RSN/traj_sampling/flow_Cuniform/uniform_sampled_actions_trajectories_10000.pickle"`.
  Update this to the file you generated locally (enable `UNIFORM_ACTION_TRAJ` in
  `c_uniform_sampling.py` to produce it). See [../data_artifacts.md](../data_artifacts.md).

### `animation.py` — trajectory animation

```bash
python flow_Cuniform/animation.py
```

- Loads a trajectory file and animates the particle distribution over the
  horizon. Set the input filename and output filename in `main()`.

## Typical workflow

1. Run `c_uniform_sampling.py` to produce reachability, processed, and
   trajectory `.pkl` files.
2. Point `coverage_analysis.py` at those files and run it.
3. Optionally visualize with `animation.py`.

See [../data_artifacts.md](../data_artifacts.md) for the naming scheme and
regeneration details of every `.pkl`/`.pickle` file referenced here.
