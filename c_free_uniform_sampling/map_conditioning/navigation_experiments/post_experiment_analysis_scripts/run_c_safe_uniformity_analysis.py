import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE other imports
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pickle
import yaml
import json
import math
import time
from collections import defaultdict
import gc
import argparse
from matplotlib.colors import ListedColormap
import copy # Import copy for deepcopying configurations

# =============================================================================
# 1. Setup, Path Handling, and Initialization
# =============================================================================

# Set up paths relative to the script location
# Script location: PROJECT_ROOT/map_conditioning/navigation_experiments/post_experiment_analysis_scripts
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    print("Warning: Running in an environment where __file__ is not defined.")
    SCRIPT_DIR = os.getcwd()

# Navigate up to the PROJECT_ROOT (e.g., traj_sampling)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../'))
# Navigate to navigation_experiments root
NAV_EXP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if NAV_EXP_ROOT not in sys.path:
    sys.path.append(NAV_EXP_ROOT)

MAP_COND_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'map_conditioning'))
if MAP_COND_DIR not in sys.path:
    sys.path.append(MAP_COND_DIR)

# Add path for 'classes' directory containing grid.py
CLASSES_ROOT = os.path.join(PROJECT_ROOT, 'classes')
if CLASSES_ROOT not in sys.path:
    sys.path.append(CLASSES_ROOT)

print(f"Project Root set to: {PROJECT_ROOT}")

# Imports requiring initialization
import torch
from scipy.stats import entropy
from scipy.ndimage import distance_transform_edt

from classes.grid import Grid
# Import Controllers from 'navigation_experiments/controllers'
from controllers.nn_cuniform_controller import CUniformController
from controllers.mppi_pytorch_controller import MPPIPyTorchController
from controllers.torch_planner_base import TorchPlannerBase

# Constants
EPSILON = 1e-9
ANALYSIS_SEED = 2025
SAMPLING_BUDGET = 50000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTERPOLATION_POINTS = 10 

# NOTE: Run into seg fault when import from the utility_helper_map somehow, so directly copy it here
def visualize_trajectories_background(
    trajectories, costmap, resolution=0.05,
    show_vis=False, save_vis=False, vis_filepath=None,
    alpha=None, marker_size=None, title=None,
):
    visual_start = time.time()
    grid_size = costmap.shape[0]
    center_index = (grid_size - 1) / 2.0
    x_coords = (np.arange(grid_size) - center_index) * resolution
    y_coords = (center_index - np.arange(grid_size)) * resolution

    plt.figure(figsize=(10, 10))
    
    # Create a more visually distinct costmap - use bright red for obstacles
    cmap = ListedColormap(['white', 'red'])  # White for free space, bright red for obstacles
    plt.imshow(costmap, origin='lower',
               extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
               cmap=cmap, alpha=1.0, zorder=1, vmin=0, vmax=1)  # Lower zorder so trajectories are on top
    plt.colorbar(label='Costmap (White: Free, Red: Obstacle)')

    # Define a color map for different level sets (time steps)
    alpha = alpha if alpha is not None else 0.8 # default alpha value
    marker_size = marker_size if marker_size is not None else 1 # default marker size
    if trajectories and len(trajectories) > 0 and len(trajectories[0]) > 0:
        num_steps = len(trajectories[0])
        
        # Use the modern API to avoid MatplotlibDeprecationWarning, with fallback
        try:
            cmap_traj = matplotlib.colormaps.get_cmap('viridis', num_steps)
        except (AttributeError, TypeError):
            cmap_traj = plt.cm.get_cmap('viridis', num_steps)
        
        # First pass: Draw the black lines connecting the states
        for trajectory in trajectories:
            x_traj = [state[0] for state, _ in trajectory]
            y_traj = [state[1] for state, _ in trajectory]
            plt.plot(x_traj, y_traj, linewidth=0.1, color='black', alpha=alpha, zorder=100)  # High zorder
        
        # Second pass: Draw colored dots for each state
        # Create legend handles
        legend_handles = []
        
        for t in range(num_steps):
            # Use a consistent color for all states at the same time step
            color = cmap_traj(t)
            
            # Plot all states at time step t across all trajectories
            x_all_t = [trajectory[t][0][0] for trajectory in trajectories]
            y_all_t = [trajectory[t][0][1] for trajectory in trajectories]
            
            # Plot actual data points (small and transparent) with highest zorder
            plt.scatter(x_all_t, y_all_t, s=marker_size, color=color, alpha=alpha, zorder=999)
            
            # Smart legend: show representative level sets for better visualization
            should_show_in_legend = False
            if num_steps <= 10:
                # Show all level sets if we have 10 or fewer
                should_show_in_legend = True
            else:
                # For 10 or more level sets: show level 0, then every 3rd level (3, 6, 9, ...), plus the last level
                should_show_in_legend = (t == 0) or (t % 3 == 0 and t > 0) or (t == num_steps-1)
            
            if should_show_in_legend:
                # Create a separate, more visible marker for the legend (larger and opaque)
                legend_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                          markersize=8, alpha=1.0, label=f'Level Set {t}')
                legend_handles.append(legend_marker)

    # Use custom title if provided, otherwise use default
    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')
    else:
        plt.title("Sampled Trajectories with State Markers")
        
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.gca().invert_yaxis()
    
    # Add legend with adaptive layout based on number of level sets
    if 'legend_handles' in locals() and legend_handles:
        legend_title = f"Level Sets (Time Steps, Total: {num_steps})"
        if num_steps <= 10:
            # Simple single column for small numbers
            plt.legend(handles=legend_handles, loc='upper right', 
                      title=legend_title, fontsize='small')
        elif num_steps <= 20:
            # Two columns for medium numbers
            plt.legend(handles=legend_handles, loc='upper right', 
                      title=legend_title, fontsize='small', 
                      title_fontsize='small', ncol=2)
        else:
            # Compact multi-column layout for many level sets
            plt.legend(handles=legend_handles, loc='upper right', 
                      title=legend_title, fontsize='x-small', 
                      title_fontsize='small', ncol=3)

    plt.tight_layout()
    
    if save_vis and vis_filepath:
        plt.savefig(vis_filepath, dpi=300, bbox_inches='tight')
        # print(f"Saved visualization to {vis_filepath}")
    
    if show_vis:
        plt.show()
    else:
        plt.close()
    
    # Reduced verbosity
    # print(f"Visualization completed in {time.time() - visual_start:.2f} seconds")

# =============================================================================
# 2. Controller Wrappers for Analysis (Dependency Injection)
# =============================================================================

class AnalysisTorchPlannerBase(TorchPlannerBase):
    """
    A wrapper for TorchPlannerBase that overrides the perception pipeline.
    """
    def set_precomputed_perception(self, perception_data):
        self.precomputed_perception = perception_data
        # Update the internal state used by the cost calculation (required for MPPI)
        self.local_costmap_map = perception_data.get('inflated_costmap')
        # Clear tensor cache to force regeneration based on the new map
        self.local_costmap_tensor = None
        self.convolved_costmap_tensor = None

    def override_sampling_budget(self, new_budget):
        """Override the sampling budget at runtime."""
        self.num_rollouts = new_budget
        # Update all related sampling parameters
        if hasattr(self, 'num_trajectories'):
            self.num_trajectories = new_budget
        if hasattr(self, 'K'):  # For MPPI controllers
            self.K = new_budget
        
        # Update visualization limit to match sampling budget for analysis
        if hasattr(self, 'num_vis_rollouts'):
            self.num_vis_rollouts = new_budget
        if hasattr(self, 'num_vis_trajectories'):
            self.num_vis_trajectories = new_budget
        
        # Handle CUMPPIController special case (split budget)
        if hasattr(self, 'mppi_refiner') and hasattr(self, 'initialization_budget_ratio'):
            init_budget = int(new_budget * self.initialization_budget_ratio)
            mppi_budget = new_budget - init_budget
            self.num_trajectories_init = init_budget
            self.num_trajectories = self.num_trajectories_init
            self.num_trajectories_mppi = mppi_budget
            self.mppi_refiner.K = mppi_budget
            # Also update MPPI refiner's visualization limit
            if hasattr(self.mppi_refiner, 'num_vis_rollouts'):
                self.mppi_refiner.num_vis_rollouts = mppi_budget

    def _prepare_perception_inputs(self, global_occupancy_grid, current_state):
        # Override the base method to return the injected data
        if hasattr(self, 'precomputed_perception') and self.precomputed_perception:
            return self.precomputed_perception
        # Fallback (should not happen during analysis)
        print("WARNING: Pre-computed perception not used. Falling back to simulation.")
        return super()._prepare_perception_inputs(global_occupancy_grid, current_state)

# Define specific controller wrappers using Multiple Inheritance
class AnalysisCUniformController(CUniformController, AnalysisTorchPlannerBase):
    # Inherits initialization from CUniformController and perception override from AnalysisTorchPlannerBase
    def __init__(self, *args, **kwargs):
        CUniformController.__init__(self, *args, **kwargs)

class AnalysisMPPIPyTorchController(MPPIPyTorchController, AnalysisTorchPlannerBase):
     # Inherits initialization from MPPIPyTorchController and perception override from AnalysisTorchPlannerBase
     def __init__(self, *args, **kwargs):
        MPPIPyTorchController.__init__(self, *args, **kwargs)

# =============================================================================
# 3. Helper Functions
# =============================================================================
def load_yaml_config(config_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

def prepare_analysis_config(base_config):
    """Prepares the configuration for analysis by setting the high sampling budget."""
    # Use deepcopy to ensure nested dictionaries (like controller configs) are independent
    analysis_config = copy.deepcopy(base_config)
    
    # Set high budget for rollouts
    analysis_config['num_rollouts'] = SAMPLING_BUDGET
    
    # Ensure velocity mode is correctly detected (required by TorchPlannerBase)
    # This will be finalized in prepare_runtime_config.
    vrange = analysis_config.get('vrange', [])
    if len(vrange) == 2 and vrange[0] != vrange[1]:
        analysis_config['variable_velocity_mode'] = True
    else:
        analysis_config['variable_velocity_mode'] = False
    return analysis_config

def prepare_runtime_config(base_analysis_config, env_config):
    """
    Merges the ground truth dynamics configuration (env_config) into the
    base analysis configuration. Ensures controllers use the exact same dynamics as the GT generation.
    """
    # Use deepcopy to handle nested dictionaries and prevent side effects
    runtime_config = copy.deepcopy(base_analysis_config)
    
    try:
        # 1. Update shared parameters (experiment_config level)
        
        # Time discretization and horizon
        runtime_config['dt'] = float(env_config['dt'])
        # GT uses 'total_t', controllers use 'horizon_T'. Round slightly for float stability.
        runtime_config['horizon_T'] = round(float(env_config['total_t']), 3)
        
        # Velocity range (GT uses tuple, config expects list)
        runtime_config['vrange'] = [float(v) for v in env_config['vrange']]
        
        # Angular velocity range (wrange). 
        # GT config (steering_angle_range) uses degrees, config (wrange) uses radians.
        if 'steering_angle_range' in env_config:
            steering_range_deg = env_config['steering_angle_range']
            runtime_config['wrange'] = [np.deg2rad(float(s)) for s in steering_range_deg]

        # Update velocity mode based on the new vrange
        vrange = runtime_config['vrange']
        if len(vrange) == 2 and vrange[0] != vrange[1]:
            runtime_config['variable_velocity_mode'] = True
        else:
            runtime_config['variable_velocity_mode'] = False

        # 2. Update C-Uniform specific parameters (controller_config level)
        # This is crucial as CUniformController initializes its action space based on these.
        if 'cuniform_controller' in runtime_config:
            cu_config = runtime_config['cuniform_controller']
            if 'arange' in env_config:
                cu_config['arange'] = [float(a) for a in env_config['arange']]
            if 'num_a' in env_config:
                cu_config['num_a'] = int(env_config['num_a'])
            if 'num_steering_angle' in env_config:
                cu_config['num_steering_angle'] = int(env_config['num_steering_angle'])

    except (KeyError, ValueError, TypeError) as e:
        raise RuntimeError(f"Error processing GT configuration during merge: {e}")

    return runtime_config

def generate_inflated_costmap(binary_costmap, inflation_radius, max_inflation_value):
    """Generates an inflated costmap (required for MPPI cost function). Logic from perception.py."""
    # ... (Implementation remains the same as provided in the original file) ...
    binary_costmap = binary_costmap.astype(np.float32)
    obstacle_mask = binary_costmap > 0.5
    
    if obstacle_mask.all():
        return np.full_like(binary_costmap, max_inflation_value, dtype=np.float32)
    if not obstacle_mask.any():
        return np.zeros_like(binary_costmap, dtype=np.float32)

    distance_map = distance_transform_edt(~obstacle_mask)

    if inflation_radius > 0:
        inflated_costmap = np.clip(
            (inflation_radius - distance_map) / inflation_radius, 
            0, 1
        ) * max_inflation_value
    else:
        inflated_costmap = np.zeros_like(distance_map, dtype=np.float32)

    inflated_costmap = np.maximum(binary_costmap, inflated_costmap)
    return inflated_costmap

def calculate_metrics(Domain_Cells, CSafe_Cells, P_star, Q):
    """Calculates KL Divergence and C-Safe Entropy Ratio."""
    
    if not Domain_Cells:
        return float('nan'), float('nan')
        
    if not CSafe_Cells:
        # If the C-Safe set is empty, uniformity is 0. KL divergence is calculable because P* is smoothed.
        pass

    # 1. KL Divergence (D_KL(Q || P*))
    # Ensure P_star and Q are aligned based on Domain_Cells order
    P_values = []
    Q_values = []
    # Sort to ensure consistent iteration order
    Domain_Cells_list = sorted(list(Domain_Cells))
    
    for cell in Domain_Cells_list:
        # P_star is smoothed and covers the domain.
        P_values.append(P_star.get(cell, EPSILON))
        # Q is the empirical distribution normalized over the domain. Use 0.0 if no samples landed in the cell.
        # We do not smooth Q here. We calculate D_KL(Q_empirical || P*_smooth).
        Q_values.append(Q.get(cell, 0.0))

    # KL Divergence (base e, nats). scipy.stats.entropy(pk, qk) computes KL(P||Q).
    # We compute D_KL(Q || P*).
    kl_div = entropy(Q_values, P_values)

    # 2. C-Safe Entropy Ratio
    if not CSafe_Cells:
        return kl_div, 0.0

    Q_csafe_values = []
    CSafe_Cells_list = sorted(list(CSafe_Cells))

    for cell in CSafe_Cells_list:
        Q_csafe_values.append(Q.get(cell, 0.0)) # Use 0.0 here as we renormalize
    
    # Normalize the C-Safe portion of Q (distribution over the safe region only)
    Q_csafe_sum = sum(Q_csafe_values)
    if Q_csafe_sum > 0:
        Q_csafe_values_norm = [v / Q_csafe_sum for v in Q_csafe_values]
    else:
        # Handle case where no samples fell into C-Safe zone (0% uniformity)
        return kl_div, 0.0

    # H_empirical (base 2, bits)
    H_empirical = entropy(Q_csafe_values_norm, base=2)
    
    # H_max
    H_max = math.log2(len(CSafe_Cells_list))
    
    if H_max > 0:
        entropy_ratio = H_empirical / H_max
    else:
        # If only 1 cell, it's perfectly uniform by definition if samples landed there
        entropy_ratio = 1.0

    return kl_div, entropy_ratio

def check_collision_sdf_vectorized(states_xy, sdf, resolution):
    """
    Vectorized collision checking against a local SDF.
    """
    # ... (Implementation remains the same as provided in the original file) ...
    H, W = sdf.shape
    center_x = (W - 1) / 2.0
    center_y = (H - 1) / 2.0

    cols = np.round(center_x + states_xy[:, 0] / resolution).astype(int)
    rows = np.round(center_y - states_xy[:, 1] / resolution).astype(int)

    out_of_bounds = (cols < 0) | (cols >= W) | (rows < 0) | (rows >= H)
    
    cols_clipped = np.clip(cols, 0, W - 1)
    rows_clipped = np.clip(rows, 0, H - 1)
    
    sdf_values = sdf[rows_clipped, cols_clipped]
    
    collisions = out_of_bounds | (sdf_values < -(0.05+1e-3)) # checking collision against the original raw obstacle map
    return collisions

def check_collision_interpolated_vectorized(trajectories, sdf, resolution, num_interp_points=10):
    """
    Vectorized, interpolated collision checking (swept volume) for trajectories against a local SDF.
    Ensures that the path between discrete states is also collision-free.

    Args:
        trajectories: (K, T+1, D>=2) numpy array of trajectories.
        sdf: (H, W) numpy array representing the SDF.
        resolution: Grid resolution (meters/cell).
        num_interp_points: Number of intermediate points (P) to check between two consecutive states.

    Returns:
        (K,) boolean array, True if the trajectory collides at any point.
    """
    K, T_plus_1, D = trajectories.shape
    T = T_plus_1 - 1 # Number of segments
    
    if T == 0:
        # Handle case with only start state. Check the single point.
        return check_collision_sdf_vectorized(trajectories[:, 0, :2], sdf, resolution)

    # 1. Generate interpolation weights
    # We generate P+1 points, including the start (0.0) and end (1.0) of each segment.
    weights = np.linspace(0, 1, num_interp_points + 1) 
    P_plus_1 = len(weights)

    # 2. Define segments (start and end points)
    # Shape: (K, T, 2)
    starts = trajectories[:, :-1, :2] 
    ends = trajectories[:, 1:, :2]   
    
    # 3. Perform interpolation using broadcasting
    # We want the resulting shape (K, T, P+1, 2).
    
    # Reshape inputs for broadcasting:
    # starts_reshaped: (K, T, 1, 2)
    # weights_reshaped: (1, 1, P+1, 1)
    starts_reshaped = starts[:, :, np.newaxis, :]
    weights_reshaped = weights[np.newaxis, np.newaxis, :, np.newaxis]

    # Interpolation formula: p_interp = p_start * (1-w) + p_end * w
    # Shape: (K, T, P+1, 2)
    interpolated_points = starts_reshaped * (1 - weights_reshaped) + \
                          ends[:, :, np.newaxis, :] * weights_reshaped
    
    # 4. Flatten the points for efficient vectorized SDF lookup
    # Shape: (K*T*(P+1), 2)
    all_points_flat = interpolated_points.reshape(-1, 2)
    
    # 5. Check collisions using the discrete checker
    collisions_flat = check_collision_sdf_vectorized(all_points_flat, sdf, resolution)
    
    # 6. Reshape back and aggregate results
    # Shape: (K, T*(P+1))
    collisions_reshaped = collisions_flat.reshape(K, -1)
    
    # Determine if any point along the entire trajectory collided.
    # Shape: (K,)
    trajectory_collided = np.any(collisions_reshaped, axis=1)
    return trajectory_collided

# =============================================================================
# 4. Core Analysis Logic
# =============================================================================

def analyze_environment(env_dir, open_space_gt, analysis_config, controllers_factory, output_dir):
    """Analyzes a single environment (scenario) across all controllers."""
    env_name = os.path.basename(env_dir)
    print(f"\nAnalyzing environment: {env_name}")

    # --- 1. Load Environment Data ---
    try:
        gt_path = os.path.join(env_dir, "uniformity_gt.pkl")
        sdf_path = os.path.join(env_dir, "sdf.npy")
        costmap_path = os.path.join(env_dir, "costmap.npy")

        with open(gt_path, 'rb') as f:
            env_gt = pickle.load(f)
        
        local_sdf = np.load(sdf_path)
        local_binary_costmap = np.load(costmap_path) 
    except Exception as e:
        print(f"  Error loading data for {env_name}: {e}")
        return None

    # --- 2. Initialization and Setup ---
    env_config = env_gt['config']
    
    # Inject GT dynamics into the analysis configuration
    runtime_analysis_config = prepare_runtime_config(analysis_config, env_config)
    
    # Initialize Grid
    thresholds = np.array(env_config['thresholds'])
    grid = Grid(thresholds)
    
    # Determine resolution (Perception settings like resolution come from the base analysis_config)
    local_resolution = analysis_config.get('local_costmap_resolution', 0.05)

    # Define start state (origin in local frame)
    initial_velocity = env_config['vrange'][0]
    start_state = np.array([0.0, 0.0, 0.0, initial_velocity])
    dummy_goal = np.array([100.0, 0.0]) # Far away goal

    # Identify the number of levels (T+1)
    num_levels = len(env_gt['level_set_representatives'])
    
    # --- 3. Pre-compute GT Cell Sets and P* ---
    GT_Data_Per_Level = []
    for t in range(num_levels):
        # A. Identify Cell Sets
        env_states_raw = env_gt['level_set_representatives'][t]
        GT_Env_States_t = np.array(list(env_states_raw) if isinstance(env_states_raw, set) else env_states_raw)
        
        if t >= len(open_space_gt['level_set_representatives']):
            GT_Data_Per_Level.append(None); continue
             
        open_states_raw = open_space_gt['level_set_representatives'][t]
        GT_Open_States_t = np.array(list(open_states_raw) if isinstance(open_states_raw, set) else open_states_raw)

        if GT_Env_States_t.size == 0 or GT_Open_States_t.size == 0:
            GT_Data_Per_Level.append(None); continue

        CSafe_Cells_t = set(map(tuple, grid.get_index_vectorized(GT_Env_States_t)))
        Domain_Cells_t = set(map(tuple, grid.get_index_vectorized(GT_Open_States_t)))
        
        CSafe_Cells_t = CSafe_Cells_t.intersection(Domain_Cells_t)

        if not CSafe_Cells_t:
            GT_Data_Per_Level.append(None); continue

        # B. Define Target Distribution P_t* (Smoothed)
        P_star_t = defaultdict(lambda: EPSILON)
        p_safe = 1.0 / len(CSafe_Cells_t)
        for cell in CSafe_Cells_t:
            P_star_t[cell] = p_safe
            
        # Normalize P_t* over the whole domain
        P_sum = sum(P_star_t.values()) + EPSILON * (len(Domain_Cells_t) - len(CSafe_Cells_t))
        
        for cell in Domain_Cells_t:
             P_star_t[cell] /= P_sum

        GT_Data_Per_Level.append({
            'Domain': Domain_Cells_t,
            'CSafe': CSafe_Cells_t,
            'P_star': P_star_t
        })

    # --- 4. Prepare Perception Data for Injection ---
    perception_data = { # costmap is already inflated in the dataset generation
        'binary_costmap': local_binary_costmap.astype(np.float32),
        'inflated_costmap': local_binary_costmap.astype(np.float32),
        'sdf_tensor': torch.from_numpy(local_sdf).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    }

    # --- 5. Controller Loop ---
    results = {}
    dummy_map = np.zeros((10, 10), dtype=np.int8)

    for controller_name in controllers_factory.keys():
        print(f"  Testing Controller: {controller_name}...")
        start_time = time.time()

        ControllerClass, config_section, overrides = controllers_factory[controller_name]

        # Initialize Controller using the Analysis Wrapper
        try: # Initialize with the runtime config that includes GT dynamics
            controller = ControllerClass(
                controller_config=runtime_analysis_config[config_section],
                experiment_config=runtime_analysis_config,
                **overrides,
                seed=ANALYSIS_SEED,
            )
            controller.reset()
            # Override the sampling budget at runtime
            controller.override_sampling_budget(SAMPLING_BUDGET)
            # Inject the pre-computed perception data
            controller.set_precomputed_perception(perception_data)
        except Exception as e:
            print(f"    Error initializing controller {controller_name}: {e}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        try: # Sample Trajectories
            _, info = controller.get_control_action(
                start_state, dummy_goal, dummy_map, runtime_analysis_config['dt']
            )
            
            sampled_rollouts = info['state_rollouts']
            
            # Verify sample count
            actual_samples = sampled_rollouts.shape[0]
            print(f"    Generated {actual_samples} samples (expected {SAMPLING_BUDGET})")
            if actual_samples < SAMPLING_BUDGET * 0.95: # Allow slight variation
                print(f"    Warning: Expected {SAMPLING_BUDGET} samples, got {actual_samples}")

        except Exception as e:
            print(f"    Error during sampling for {controller_name}: {e}")
            # Memory cleanup
            del controller
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        # --- 6. Analysis ---
        
        # 6.1 Collision Rate (Vectorized)
        K = sampled_rollouts.shape[0]
        T_plus_1 = sampled_rollouts.shape[1]

        # This performs a swept volume check by densely sampling points between states.
        trajectory_collided = check_collision_interpolated_vectorized(
            sampled_rollouts, local_sdf, local_resolution, num_interp_points=INTERPOLATION_POINTS
        )
        collision_rate = np.sum(trajectory_collided) / K

        # 6.2 Per-Level Metrics
        level_metrics = {}
        # Iterate up to the minimum of GT levels and sampled levels.
        analysis_horizon = min(num_levels, T_plus_1)

        # Start analysis from t=1 (Level 1)
        for t in range(1, analysis_horizon):
                
            GT_Data_t = GT_Data_Per_Level[t]
            if GT_Data_t is None: continue

            Domain_Cells_t = GT_Data_t['Domain']
            CSafe_Cells_t = GT_Data_t['CSafe']
            P_star_t = GT_Data_t['P_star']

            # Define Empirical Distribution Q_t
            Samples_t = sampled_rollouts[:, t, :]
            Sampled_Indices_t = grid.get_index_vectorized(Samples_t)
            
            # Count occurrences
            unique_indices, counts = np.unique(Sampled_Indices_t, axis=0, return_counts=True)
            
            Q_t = defaultdict(float)
            
            # Calculate total samples that fell within the domain for normalization
            total_domain_samples = 0
            temp_Q = {}
            for idx_arr, count in zip(unique_indices, counts):
                idx_tuple = tuple(idx_arr)
                # Only count samples that fall within the reachable domain
                if idx_tuple in Domain_Cells_t:
                    temp_Q[idx_tuple] = count
                    total_domain_samples += count
            
            # Normalize Q_t over the domain (P(x | x in Domain)), not the total budget K
            if total_domain_samples > 0:
                for idx_tuple, count in temp_Q.items():
                    Q_t[idx_tuple] = count / total_domain_samples
            # If no samples fell in the domain, Q_t remains empty (defaultdict(float))

            # Calculate Metrics
            kl_div, entropy_ratio = calculate_metrics(Domain_Cells_t, CSafe_Cells_t, P_star_t, Q_t)
            
            level_metrics[t] = {
                "kl_divergence": kl_div,
                "c_safe_entropy_ratio": entropy_ratio
            }

        # Store results
        results[controller_name] = {
            "collision_rate": collision_rate,
            "level_metrics": level_metrics,
            "time_taken": time.time() - start_time
        }
        
        # --- 7. Visualization ---
        print(f"    Generating visualization for {controller_name}...")
        # Downsample for visualization
        num_vis_traj = min(1000, K)
        # Ensure deterministic visualization sampling
        vis_rng = np.random.RandomState(ANALYSIS_SEED)
        vis_indices = vis_rng.choice(K, num_vis_traj, replace=False)
        
        # The visualizer expects a list of lists of (state, action) tuples. Action can be None.
        trajectories_for_vis = [
            [(state, None) for state in rollout]
            for rollout in sampled_rollouts[vis_indices]
        ]

        # Build the dynamic title with key metrics
        title_parts = [
            f"{controller_name}",
            f"Env: {env_name} | Collision-Free Rate: {(1.0-collision_rate):.2%}"
        ]
        # Add metrics for a few representative levels
        # for t in sorted(level_metrics.keys()):
        #     metrics = level_metrics[t]
        #     title_parts.append(f"L{t}: KL={metrics['kl_divergence']:.2f}, Entr={metrics['c_safe_entropy_ratio']:.1%}")
        
        # Sanitize controller name for filename
        safe_controller_name = controller_name.replace(' ', '_').replace('(', '').replace(')', '')
        vis_filepath = os.path.join(output_dir, f"{env_name}_{safe_controller_name}_traj_vis.png")

        # Call the visualization function
        visualize_trajectories_background(
            trajectories=trajectories_for_vis, costmap=local_binary_costmap, resolution=local_resolution,
            save_vis=True, vis_filepath=vis_filepath, title="\n".join(title_parts), alpha=0.8, marker_size=0.5
        )

        # Memory cleanup
        del controller, sampled_rollouts, info
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

# =============================================================================
# 5. Reporting
# =============================================================================
def generate_report(all_results, output_dir, controllers_list):
    """Generates the final human-readable summary report."""
    print("\nGenerating final report...")
    
    # Aggregate results
    aggregated_data = defaultdict(lambda: defaultdict(list))
    num_environments = 0
    max_levels = 0

    for env_name, env_results in all_results.items():
        if env_results is None:
            continue
        num_environments += 1
        
        for controller_name, data in env_results.items():
            if data is None or "error" in data:
                continue
                
            aggregated_data[controller_name]["collision_rate"].append(data["collision_rate"])
            
            for t, metrics in data["level_metrics"].items():
                if not math.isnan(metrics["kl_divergence"]):
                    aggregated_data[controller_name][f"kl_div_L{t}"].append(metrics["kl_divergence"])
                if not math.isnan(metrics["c_safe_entropy_ratio"]):
                    aggregated_data[controller_name][f"entropy_ratio_L{t}"].append(metrics["c_safe_entropy_ratio"])
                max_levels = max(max_levels, t)

    # Calculate averages
    summary = defaultdict(dict)
    for controller_name, data in aggregated_data.items():
        for metric, values in data.items():
            if values:
                if 'kl_div' in metric and np.isinf(np.max(values)):
                     summary[controller_name][metric] = float('inf')
                else:
                    summary[controller_name][metric] = np.mean(values)
            else:
                summary[controller_name][metric] = float('nan')

    # Generate Report Text
    report = []
    report.append("="*70)
    report.append("           C-Safe Uniformity Analysis: Final Report")
    report.append("="*70)
    report.append(f"Total Environments Analyzed: {num_environments}")
    report.append(f"Sampling Budget per Controller: {SAMPLING_BUDGET}\n")

    for controller_name in controllers_list:
        display_name = controller_name

        if controller_name not in summary:
            report.append(f"----------------------------------------------------------------------")
            report.append(f"CONTROLLER: {display_name}")
            report.append(f"----------------------------------------------------------------------")
            report.append("  No successful results recorded.\n")
            continue

        data = summary[controller_name]
        
        report.append(f"----------------------------------------------------------------------")
        report.append(f"CONTROLLER: {display_name}")
        report.append(f"----------------------------------------------------------------------")
        
        collision_rate = data.get("collision_rate", float('nan'))
        report.append(f"Global Collision Rate: {collision_rate*100:.2f}%\n")
        
        report.append("Per-Level Metrics (Averaged):")
        
        for t in range(1, max_levels + 1):
            kl_div = data.get(f"kl_div_L{t}", float('nan'))
            entropy_ratio = data.get(f"entropy_ratio_L{t}", float('nan'))
            
            report.append(f"Level {t}:")
            # Explicitly state the KL direction D_KL(Q||P*)
            report.append(f"  - KL Divergence (Q||P*): {kl_div:.4f} nats")
            report.append(f"  - C-Safe Entropy Ratio: {entropy_ratio:.4f} ({entropy_ratio*100:.1f}% Uniformity)")
        
        report.append("\n")
    report.append("="*70)

    report_path = os.path.join(output_dir, "final_uniformity_report.txt")
    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        print(f"Final report saved to: {report_path}")
    except Exception as e:
        print(f"Error saving final report: {e}")

# =============================================================================
# 6. Main Execution
# =============================================================================
def main(input_dataset_dir, open_space_gt_path):
    # Define Output Directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(SCRIPT_DIR, f"analysis_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("C-SAFE UNIFORMITY ANALYSIS")
    print(f"Budget: {SAMPLING_BUDGET} | Seed: {ANALYSIS_SEED}")
    print("="*80)

    # --- 1. Load Master Data ---
    # Load Open Space GT
    if not os.path.exists(open_space_gt_path):
        print(f"Error: Open Space GT file not found at {open_space_gt_path}")
        return
    try:
        with open(open_space_gt_path, 'rb') as f:
            open_space_gt = pickle.load(f)
            if 'pruned_level_set_representatives_across_LS' in open_space_gt:
                open_space_gt['level_set_representatives'] = open_space_gt['pruned_level_set_representatives_across_LS']
    except Exception as e:
        print(f"Error loading Open Space GT: {e}"); return

    # Load Experiment Config (YAML)
    config_path = os.path.join(NAV_EXP_ROOT, "configs", "experiment_config.yaml")
    base_experiment_config = load_yaml_config(config_path)
    if not base_experiment_config:
        return
        
    # Prepare the config for analysis (set budget, etc.)
    analysis_config = prepare_analysis_config(base_experiment_config)

    # --- 2. Define Controllers Factory (Using Analysis Wrappers) ---
    CONTROLLERS_FACTORY = {
        "Map-Conditioned C-Safe Uniform": (
            AnalysisCUniformController, "cuniform_controller", {"type_override": 1}
        ),
        "C-Uniform Sampler": (
            AnalysisCUniformController, "cuniform_controller", {"type_override": 0}
        ),
        "Standard MPPI": (
            AnalysisMPPIPyTorchController, "mppi_controller", {"type_override": 0}
        ),
        "Log-MPPI": (
            AnalysisMPPIPyTorchController, "mppi_controller", {"type_override": 1}
        ),
    }
    # Maintain order for the report
    controllers_list = list(CONTROLLERS_FACTORY.keys())

    # --- 3. Discover Environments ---
    if not os.path.exists(input_dataset_dir):
        print(f"Error: Input dataset directory not found: {input_dataset_dir}")
        return

    env_dirs = []
    # Sort directory listing for deterministic processing order
    for item in sorted(os.listdir(input_dataset_dir)):
        full_path = os.path.join(input_dataset_dir, item)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "uniformity_gt.pkl")):
            env_dirs.append(full_path)
    
    if not env_dirs:
        print("No valid environment directories found in the input dataset.")
        return

    print(f"Found {len(env_dirs)} environments to analyze.")

    # --- 4. Run Analysis ---
    config_checked = False
    all_results = {}
    
    # Set numpy print options for cleaner configuration display during the check
    np.set_printoptions(precision=8, suppress=True, linewidth=120)

    for env_dir in env_dirs:
        # Perform a one-time, verbose check on the first valid environment
        # This verifies that the Open Space GT and the Environment GT used the same dynamics.
        if not config_checked:
            try:
                with open(os.path.join(env_dir, "uniformity_gt.pkl"), 'rb') as f:
                    first_env_gt = pickle.load(f)
                
                print("\n--- Performing one-time configuration consistency check (Open Space GT vs Env GT) ---")
                env_config = first_env_gt['config']
                open_space_config = open_space_gt['config']

                mismatched_keys = []
                keys_to_ignore = {'dynamics', 'inverse_dynamics', 'slack_parameter', 'vectorized_dynamics'}
                all_keys = sorted(list(set(open_space_config.keys()) | set(env_config.keys())))

                for key in all_keys:
                    if key in keys_to_ignore:
                        continue
                    val1, val2 = open_space_config.get(key), env_config.get(key)
                    are_equal = False
                    
                    # Robust comparison logic
                    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                        are_equal = np.array_equal(val1, val2)
                    elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                        are_equal = (tuple(map(float, val1)) == tuple(map(float, val2)))
                    elif isinstance(val1, (float, np.floating)) and isinstance(val2, (float, np.floating)):
                         are_equal = math.isclose(float(val1), float(val2), rel_tol=1e-5)
                    else:
                        are_equal = (val1 == val2)
                    
                    if not are_equal:
                        mismatched_keys.append(key)
                
                if not mismatched_keys:
                    print("\n[Result]: Configurations match perfectly. ✔️")
                else:
                    print(f"\n[Result]: Configurations DO NOT match. Mismatches in keys: {mismatched_keys} ❌")
                    print("FATAL: Open Space GT and Environment GT must use the same dynamics configuration.")
                    return # Exit if configs don't match
                
                config_checked = True
                print("--- Configuration check complete. Proceeding with analysis. ---\n")
            except Exception as e:
                print(f"Error during config check on {os.path.basename(env_dir)}: {e}")
                return

        # Analyze environment
        results = analyze_environment(
            env_dir, open_space_gt, analysis_config, CONTROLLERS_FACTORY, output_dir
        )
        
        env_name = os.path.basename(env_dir)
        all_results[env_name] = results

        # Save detailed results for this environment (JSON)
        if results:
            details_path = os.path.join(output_dir, f"{env_name}_analysis.json")
            try:
                with open(details_path, 'w') as f:
                    # Use default=str to handle potential non-serializable types (e.g. errors)
                    json.dump(results, f, indent=4, default=str)
            except TypeError as e:
                 print(f"Warning: Could not serialize detailed results for {env_name}. Error: {e}")

    # Save comprehensive JSON
    analysis_filename = f"analysis_complete.json"
    analysis_filepath = os.path.join(output_dir, analysis_filename)
    try:
        with open(analysis_filepath, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving comprehensive JSON: {e}")


    # --- 5. Generate Final Report ---
    generate_report(all_results, output_dir, controllers_list)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    print("Running C-Safe Uniformity Analysis...")
    # Set seeds for reproducibility
    np.random.seed(ANALYSIS_SEED)
    torch.manual_seed(ANALYSIS_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ANALYSIS_SEED)

    parser = argparse.ArgumentParser(description="Run C-Safe Uniformity Analysis.")
    
    # Define default paths
    MAP_COND_DIR = os.path.join(PROJECT_ROOT, 'map_conditioning')
    # default_input_dir = os.path.join(MAP_COND_DIR, "dataset_polygon_uniformity_localized_gt")
    # default_input_dir = os.path.join(MAP_COND_DIR, "dataset_polygon_uniformity_localized_gt_dt0.1")
    # default_input_dir = os.path.join(MAP_COND_DIR, "dataset_polygon_uniformity_localized_gt_v_1.25_dt0.2")
    # default_input_dir = os.path.join(MAP_COND_DIR, "dataset_polygon_uniformity_localized_gt_v_1.25_dt0.1")
    default_input_dir = os.path.join(MAP_COND_DIR, "dataset_polygon_uniformity_localized_gt_v_1.25_dt0.1_t2.41")
    #NOTE: don't forget to change the open space gt path
    #NOTE: don't forget to change the unsupervided cuniform model path inside the experiment_config.yaml
    
    # Using the absolute path provided in the user's context
    # default_open_space_gt = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset_supervised/saved_pickles/C_Uniform_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_seed_2025_grid_0.100_0.100_6.000deg_t1.21_ts0.2_vrange_2.5_2.5_steer_range_-30.0_30.0_steering_31.pkl"
    # default_open_space_gt = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset_supervised/saved_pickles/C_Uniform_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_seed_2025_grid_0.100_0.100_6.000deg_t1.21_ts0.1_vrange_2.5_2.5_steer_range_-30.0_30.0_steering_31.pkl"
    # default_open_space_gt = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset_supervised/saved_pickles/C_Uniform_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_seed_2025_grid_0.100_0.100_6.000deg_t1.21_ts0.2_vrange_1.25_1.25_steer_range_-30.0_30.0_steering_31.pkl"
    # default_open_space_gt = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset_supervised/saved_pickles/C_Uniform_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_seed_2025_grid_0.100_0.100_6.000deg_t1.21_ts0.1_vrange_1.25_1.25_steer_range_-30.0_30.0_steering_31.pkl"
    default_open_space_gt = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset_supervised/saved_pickles/C_Uniform_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_seed_2025_grid_0.100_0.100_6.000deg_t2.41_ts0.1_vrange_1.25_1.25_steer_range_-30.0_30.0_steering_31.pkl"

    parser.add_argument('--input_dir', type=str, default=default_input_dir,
                        help='Directory containing the environment ground truth subfolders.')
    parser.add_argument('--open_space_gt', type=str, default=default_open_space_gt,
                        help='Path to the master open-space ground truth pickle file.')
    
    # Handle execution in environments where sys.argv might be complex
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Warning: Argument parsing failed or caused exit. Attempting to run with default arguments.")
        args = parser.parse_args([])

    main(args.input_dir, args.open_space_gt)