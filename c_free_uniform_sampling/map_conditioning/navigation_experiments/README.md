# Navigation Controller Benchmarking Framework

A minimal framework for comparing MPPI and C-Uniform controllers in robotic navigation simulations.

## Environment Setup

This module runs entirely under the project's **`c_uniform`** conda environment (the old
`sim_exp` environment is no longer used). All controllers, including the MPPI variants, run
on **PyTorch** — there is **no `pycuda` dependency** and no separate CUDA toolkit install is
required. A GPU is used automatically when available (`torch.cuda.is_available()`), otherwise
the controllers fall back to CPU.

After creating and activating `c_uniform` (see the top-level README):

```bash
conda activate c_uniform
cd map_conditioning/navigation_experiments
```

## Project Structure

```
├── run_experiments.py          # Main orchestrator
├── run_multi_experiments.py    # Batch experiment runner
├── resume_manager.py           # Resume interrupted experiments
├── configs/
│   └── experiment_config.yaml  # Global experiment settings
├── controllers/
│   ├── base_controller.py      # Abstract controller interface
│   ├── mppi_pytorch_controller.py # PyTorch MPPI (standard + log-normal)
│   ├── cu_mppi_controller.py   # C-Uniform MPPI controller
│   └── nn_cuniform_controller.py # Neural network C-Uniform controller
├── core/
│   ├── environments_manager.py # Map loading and collision detection
│   └── experiment_runner.py    # Single trial execution
├── utils/
│   ├── results_analyzer.py     # Experiment result analysis
│   ├── visualizer.py           # Trajectory visualization
│   ├── dynamics.py             # Vehicle dynamics models
│   └── perception.py           # Sensor simulation
├── data/
│   ├── create_test_maps.py     # generate synthetic test maps
│   ├── convert_barn_maps.py    # convert BARN maps to .npy
│   └── visualize_converted_maps.py  # visualize converted maps
└── results/
    └── experiment_*/           # Timestamped experiment results (generated at runtime)
```

## Usage

### Single Experiment
```bash
conda activate c_uniform
python run_experiments.py
```

### Batch Experiments
```bash
python run_multi_experiments.py
```

### Resume Interrupted Experiments
```bash
python resume_manager.py
```

## Controllers

### Available Controllers
1. **MPPI Controller** - Model Predictive Path Integral control
2. **C-Uniform MPPI** - C-Uniform sampling with MPPI optimization
3. **Neural C-Uniform** - Map-conditioned neural network controller

### Configuration
Controller parameters are defined in `configs/experiment_config.yaml`:
- Sampling parameters (number of trajectories, horizon)
- Vehicle dynamics (wheelbase, velocity limits)
- Environment settings (maps, start/goal positions)

## Results Analysis

Experiment results are automatically saved with timestamps in `results/experiment_*/`:
- **Trajectory data** - Robot paths and control inputs
- **Performance metrics** - Success rate, path length, computation time
- **Comparison reports** - Statistical analysis across controllers
- **Visualizations** - Trajectory plots and heatmaps

## Adding New Controllers

1. **Extend `BaseController`** in `controllers/`
2. **Implement required methods:**
   - `plan(current_state, goal_state, environment)`
   - `get_action(current_state)`
3. **Add configuration** in `experiment_config.yaml`
4. **Register controller** in `run_experiments.py`

## Dependencies

This framework requires:
- **PyTorch** - Neural network inference and all controller sampling, including the
  MPPI variants (provided by `c_uniform`). A GPU is used when available, otherwise the
  controllers run on CPU. **No `pycuda` and no separate CUDA toolkit are required.**
- **NumPy/SciPy** - Numerical computations (provided by `c_uniform`)
- **OpenCV** - Image processing for maps (provided by `c_uniform`)

## Supported Environments

- **BARN Dataset** - Real-world inspired navigation scenarios
- **Dumbbell Maps** - Narrow passage navigation
- **Polygon Environments** - Custom geometric obstacles
- **Minimum Feature Maps** - Simplified test scenarios