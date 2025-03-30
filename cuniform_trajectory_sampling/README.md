See [C-Uniform Introduction](https://ogpoyrazoglu.github.io/cuniform_sampling/) for more details on the approach.

# C-Uniform Trajectory Sampling

This repository contains scripts for simulating and analyzing trajectories in both **random walk** system and a **Dubins car** system, using uniform action and C_Uniform sampling approaches.

## Overview of Files

### Main C-Uniform Logic

1. **`c_uniform_sampling.py`**
   - This script generates trajectories using the C_Uniform sampling approach. 
   - **High-Level Steps:**
      1. Calculate reachable cells based on linear approximation of system dynamics.
      2. Use the reachable cells result as nodes for graph, build graph connection based on feasibility.
      3. Run max flow algorithm on the resulting graph structure.
      4. Flow result yield action probability distribution for C-Uniformity.

2. **Other Key Files:**
- **`c_uniform_sampling.py`**: Main script for C-Uniform sampling 
- **`classes/grid.py`**: Defines the `Grid` class for discretizing the configuration space.
- **`classes/graph_structure.py`**: Defines the `Node` class for graph construction.
- **`dynamics_helpers.py`**: Implements dynamics functions for Dubins car and random walk systems.
- **`utility_helpers.py`**: Provides helper functions for trajectory generation, visualization, and analysis.
- **`mppi_trajs/*`**: Folder storing MPPI/Log-MPPI trajectories
- **`coverage_analysis.py`**: Experiment script for analyzing trajecoty coverage
- **`environment.yml`**: Conda environment file specifying dependencies.

## Configuration Guide

The first 50 lines in the main() function of **`c_uniform_sampling.py`** defines important configuration parameters.

### Global Settings
  - `UNIFORM_ACTION_TRAJ`: When set to `True`, the script will also generates trajectories by uniformly sampling from the discretized actions.  
  - `DISJOINT_LEVEL_SET`: Determines whether reachable grid cells are computed as disjoint sets (non-overlapping) or allowed to overlap. 

### Model Configurations

The main configurations are defined in the `model_configs` dictionary (inside `c_uniform_sampling.py`). Two models are provided: **DUBINS** and **2D_RANDOM_WALK**. Modify the values below to change system behavior.

- **Dynamics Functions**:  
  - `dynamics`: Uses `dynamics_*` to compute the state transition.  
  - `inverse_dynamics`: Uses `inverse_dynamics_*` for closed-form estimation of the control action.  
  - `vectorized_dynamics`: Uses `vectorized_dynamics_*` for efficient vectorized dynamic calculation.

- **Action Space Settings**:  
  - `vrange`: Specifies the velocity range.  
  - `arange`: Acceleration range (typically zero for constant speed).  
  - `num_a`: Number of acceleration values 
  - `steering_angle_range`: Range for steering angles in degrees.  
  - `num_steering_angle`: Number of discrete steering values.  
  - `actions`: Initially set to `None` – this is computed later using the `generate_actions()` function from `utility_helpers.py`. 

- **Grid & Time Settings**:  
  - `thresholds`: Determines the cell sizes for discretizing the configuration space. 
  - `state_dim`: Number of state dimensions.
  - `dt`: Time step (in seconds).  
  - `total_t`: Total trajectory length in seconds. It determines the planning horizon.

### Tips for Experimentation
- **Model Switching**:  
  To switch between the Dubins car model and the 2D random walk, change the `MODEL` variable (e.g., `MODEL = "DUBINS"` or `MODEL = "2D_RANDOM_WALK"`).

- **Grid Resolution**:  
  Increasing the resolution (by decreasing the threshold values) will yield more grid cells and may improve approximation fidelity but will require more computation.

- **Perturbation**:  
  Tweaking `perturbation_param` can help manage the trade-off between accuracy and feasibility.

- **Action Discretization**:  
  The number of steering angles (`num_steering_angle`) directly impacts the granularity of the control actions. Increase this value will also increase computational load.

- **Extension**:  
To extend the codebase for other dynamics, follow these steps:
   1. Define the dynamic functions in `dynamic_helpers.py`. Note that there is also code for kinematic single track dynamics available, which you can use as a reference or extend for other vehicle models.
   2. Define configurations in the `model_configs` dictionary (inside `c_uniform_sampling.py`).

## Setting Up the Environment

To ensure all dependencies are installed correctly, we recommend using the provided Conda environment file.

### Prerequisites
You will need to have **Conda** (or **Miniconda**) installed on your system. If you haven't installed it yet, follow the installation instructions from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

### Importing the Environment

The Conda environment used for this project has been saved to the `environment.yml` file. To import and recreate the environment on your machine, run:

   ```bash
   conda env create -f environment.yml
   ```
### Potential Additional Dependencies

```bash
pip install numba
pip install scipy
```

### Running the code

Simply 
```
python c_uniform_sampling.py
```
or
```
python coverage_analysis.py
```