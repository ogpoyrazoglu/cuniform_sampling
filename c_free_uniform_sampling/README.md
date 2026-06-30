# C-Free-Uniform

This is the repository for **C-Free-Uniform (CFU)**. For convenience it also ships the latest implementations of **C-Uniform** and **Unsupervised C-Uniform**, so all three works live in one place.

Together these cover three peer-reviewed robotics conference papers. The main entry point for each is:

| Paper | Main entry point |
|---|---|
| C-Uniform | `flow_Cuniform/c_uniform_sampling.py` |
| Unsupervised C-Uniform | `map_conditioning/unsupervised_training.py` |
| C-Free-Uniform | `map_conditioning/supervised_training.py` |

Yukang's Master's thesis builds upon these three works and provides more detail than the conference papers. <!-- TODO: add Yukang's Master's thesis link once published -->

<!-- TODO: UGE-MPC -->

### How the three works relate

The three papers form a single line of work on uniform trajectory sampling for sampling-based control:

- **C-Uniform** generates dynamically feasible trajectories that are uniform over the reachable state space (configuration-space uniform sampling). It builds reachability sets over a discretized state lattice and draws control-uniform samples through a max-flow / level-set formulation. Entry point: `flow_Cuniform/c_uniform_sampling.py`.
- **Unsupervised C-Uniform (IROS 2025)** trains a neural-network sampling policy to achieve C-Uniformity by iterative level-set entropy maximization — the uniform distribution is the maximum-entropy one — which avoids the explicit max-flow computation and grid discretization of the original C-Uniform that scale poorly with state-space dimension. Entry point: `map_conditioning/unsupervised_training.py`.
- **C-Free-Uniform (CFU)** extends C-Uniform to be map/obstacle-conditioned, targeting uniform coverage of the collision-*free* space rather than the full reachable set. The sampler is trained supervised and map-conditioned, then used inside CFU-MPPI for navigation. Entry point: `map_conditioning/supervised_training.py`.

Paper: https://arxiv.org/abs/2510.16905

## Capabilities

This repository provides an end-to-end pipeline for collision-aware uniform trajectory sampling and sampling-based control:

- **Max-Flow C-Uniform sampling** (`flow_Cuniform/`) — generates reachability sets over a discretized state lattice and draws control-uniform trajectory samples via a max-flow formulation.
- **Coverage analysis and visualization** (`flow_Cuniform/`) — quantifies and visualizes how uniformly sampled trajectories cover the reachable set.
- **Map-conditioned neural samplers** (`map_conditioning/`) — supervised (C-Free-Uniform) and unsupervised (C-Uniform) training of a UNet-based model that predicts action distributions conditioned on a local occupancy/SDF map.
- **Dataset generation** (`map_conditioning/`) — builds supervised and unsupervised training datasets, including polygon-uniformity ground truth.
- **Single-frame planning analysis** (`map_conditioning/single_frame_analysis/`) — compares the C-Free-Uniform sampler against the C-Uniform baseline on goal-reaching success rate (the CFU contribution).
- **Closed-loop navigation experiments** (`map_conditioning/navigation_experiments/`) — a self-contained sensor → control → dynamics simulation loop with MPPI, CU-MPPI, and map-conditioned C-Uniform controllers. Runs on PyTorch under the `c_uniform` environment (GPU optional, with CPU fallback).

## Environment Setup

The repository targets the `c_uniform` conda environment, which uses **Python 3.8** (Torch 2.4.1, cu118).

```bash
# 1. Create the environment from the shipped specification
conda env create -f environment_c_uniform.yml

# 2. Activate it
conda activate c_uniform

# 3. Install this repository as an editable package
#    (registers `classes`, `flow_Cuniform`, and `map_conditioning` as importable top-level packages)
pip install -e .
```

The editable install is required: absolute imports such as `from classes.grid import Grid` and `from flow_Cuniform.dynamics_helpers import ...` resolve only after `pip install -e .`.

## Quick Start

Run each entry point from the indicated directory so its relative config and path lookups resolve.

### Core C-Uniform sampling

```bash
cd flow_Cuniform
python c_uniform_sampling.py        # generate reachability sets + Max-Flow C-Uniform samples
python coverage_analysis.py         # coverage analysis
python animation.py                 # trajectory animation
```

### Map-conditioning: dataset generation

```bash
cd map_conditioning
python supervised_dataset_generation.py            # supervised CFU dataset
python unsupervised_dataset_generation.py          # unsupervised C-Uniform dataset
python generate_polygon_uniformity_groundtruth.py  # polygon uniformity ground truth
```

### Map-conditioning: training

```bash
cd map_conditioning
python supervised_training.py            # train the supervised map-conditioned C-Free-Uniform model
python unsupervised_training.py          # train the unsupervised C-Uniform model
python supervised_openspace_training.py  # open-space baseline training
python evaluate_baseline_model.py        # evaluate a trained baseline model
```

### Single-frame planning analysis (C-Free-Uniform vs C-Uniform)

```bash
cd map_conditioning/single_frame_analysis
python run_planning_analysis.py                         # uses configs/planning_config.yaml
python run_planning_analysis.py --config configs/planning_config.yaml
```

### Closed-loop navigation experiments

```bash
cd map_conditioning/navigation_experiments
python run_experiments.py                       # single experiment suite (configs/experiment_config.yaml)
python run_experiments.py --num-rollouts 5      # override rollout count
python run_multi_experiments.py                 # multi-experiment driver
```

Most scripts read their settings from in-file constants or a YAML config in the module's `configs/` directory rather than from command-line flags. See the per-module guides under `docs/` (`docs/modules/flow_cuniform.md`, `docs/modules/map_conditioning.md`, `docs/modules/single_frame_analysis.md`, `docs/modules/navigation_experiments.md`) for configuration details, and `docs/data_artifacts.md` for regenerating the excluded datasets and weights.

> Many entry points consume large data artifacts (`.pkl` datasets, `.npy` maps, trained `.pth`/`.pt` weights) that are **not bundled** to keep the repository minimal. Regenerate them by running the corresponding dataset-generation and training scripts first. If a script reports a missing dataset or weight file, that is a missing artifact to regenerate, not a code defect.

## Results Summary

The C-Free-Uniform result is that the **map-conditioned C-Free-Uniform MPPI controller (CFU-MPPI) outperforms the C-Uniform MPPI baseline on goal-reaching success rate while using a smaller sampling budget.** By conditioning the uniform sampler on the local map, CFU concentrates samples in the collision-free reachable set, so fewer trajectory samples are needed to achieve higher success rates than the map-agnostic C-Uniform baseline.

## Limitations

1. **Non-uniformity accumulation over the horizon.** Sampling uniformity degrades as the planning horizon grows, because small per-step deviations from uniformity compound across level sets.
2. **Curse of dimensionality fixes a nominal training velocity.** To keep the state-space discretization tractable, the model is trained at a fixed nominal velocity rather than across the full velocity range, limiting generalization to other speeds.
3. **Sequential-inference latency scales linearly with the horizon.** Because samples are produced level set by level set, inference time grows linearly with horizon length, which constrains real-time use at long horizons.

## Citation

If you use this code, please cite the **C-Free-Uniform** paper (arXiv:2510.16905) as the primary reference: https://arxiv.org/abs/2510.16905

```bibtex
@article{cao2025c,
  title   = {C-Free-Uniform: A Map-Conditioned Trajectory Sampler for Model Predictive Path Integral Control},
  author  = {Cao, Yukang and Moorthy, Rahul and Poyrazoglu, O Goktug and Isler, Volkan},
  journal = {arXiv preprint arXiv:2510.16905},
  year    = {2025}
}
```

This repository also ships the earlier works in the same C-Uniform line of research; cite these if you use the corresponding components:

```bibtex
@inproceedings{poyrazoglu2024c,
  title        = {C-Uniform trajectory sampling for fast motion planning},
  author       = {Poyrazoglu, O Goktug and Cao, Yukang and Isler, Volkan},
  booktitle    = {2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages        = {9236--9242},
  year         = {2025},
  organization = {IEEE}
}

@inproceedings{poyrazoglu2025unsupervised,
  title        = {An unsupervised c-uniform trajectory sampler with applications to model predictive path integral control},
  author       = {Poyrazoglu, O Goktug and Moorthy, Rahul and Cao, Yukang and Chastek, William and Isler, Volkan},
  booktitle    = {2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages        = {5800--5807},
  year         = {2025},
  organization = {IEEE}
}
```

Yukang's Master's thesis builds on this work and provides more detail than the conference papers, so it is worth reading. <!-- TODO: add Yukang's Master's thesis link once published -->

## Repository Layout

```
c_free_uniform_sampling/
├── setup.py                       # editable-install package definition (name='traj_sampling')
├── environment_c_uniform.yml      # conda environment specification (Python 3.8)
├── README.md                      # this file
├── LICENSE                        # MIT license
├── docs/                          # per-module guides + data-artifact regeneration guide
├── classes/                       # shared Grid / Node data structures
├── flow_Cuniform/                 # core Max-Flow C-Uniform sampling + coverage analysis
└── map_conditioning/              # neural map-conditioned samplers, training, analysis
    ├── single_frame_analysis/     # CFU vs C-Uniform success-rate comparison
    └── navigation_experiments/    # closed-loop sensor-control-dynamics simulation
```

## Maintainer

Yukang Cao — [yukang.cao@austin.utexas.edu](mailto:yukang.cao@austin.utexas.edu). Questions, issues, and pull requests are welcome.

## License

Released under the [MIT License](LICENSE).
