import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import pickle
import yaml
import time
from collections import defaultdict
import gc
import argparse
import copy
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tabulate import tabulate
from scipy.ndimage import binary_erosion

# =============================================================================
# 1. Setup, Path Handling, and Initialization
# =============================================================================

# Set up paths relative to the script location
# Script location: PROJECT_ROOT/map_conditioning/single_frame_analysis
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    print("Warning: Running in an environment where __file__ is not defined.")
    SCRIPT_DIR = os.getcwd()

# Navigate up to the PROJECT_ROOT (e.g., traj_sampling)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
# Navigate to navigation_experiments root (for importing controllers/utils)
NAV_EXP_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, 'map_conditioning/navigation_experiments'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if NAV_EXP_ROOT not in sys.path:
    sys.path.append(NAV_EXP_ROOT)

print(f"Project Root set to: {PROJECT_ROOT}")
print(f"Navigation Experiments Root set to: {NAV_EXP_ROOT}")

# Imports requiring initialization
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import Controllers and Base Classes
from controllers.nn_cuniform_controller import CUniformController
from controllers.mppi_pytorch_controller import MPPIPyTorchController
from controllers.torch_planner_base import TorchPlannerBase

from map_conditioning.navigation_experiments.post_experiment_analysis_scripts.run_c_safe_uniformity_analysis import (
    check_collision_interpolated_vectorized
)
INTERPOLATION_POINTS = 10

# =============================================================================
# 1.5. Visualization Utility (Adapted from run_c_safe_uniformity_analysis.py)
# =============================================================================

# --- Visualization Aesthetics (Matches Figure 1) ---
COLOR_OBSTACLE = '#000000'
COLOR_SAFE = '#1F77B4'
COLOR_COLLIDED = '#7F7F7F'
COLOR_SELECTED = '#FF7F0E'

def visualize_trajectories_minimal(
    state_rollouts, costmap, resolution, goal_state, goal_tolerance, sdf_np, 
    save_vis=False, vis_filepath=None
):
    """
    Minimal visualization with clean presentation style.
    Trajectories colored by status: blue (collision-free), grey (collided), orange (successful).
    Uses binary erosion to reduce obstacle inflation and match collision detection.
    """
    if costmap is None or costmap.ndim != 2:
        raise ValueError("Invalid costmap provided for visualization.")

    grid_size = costmap.shape[0]
    if costmap.shape[0] != costmap.shape[1]:
        raise ValueError("Visualization expects a square costmap.")

    center_index = (grid_size - 1) / 2.0
    x_coords = (np.arange(grid_size) - center_index) * resolution
    y_coords = (center_index - np.arange(grid_size)) * resolution

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    # 1. Apply binary erosion to reduce inflation by 1 pixel to match uninflated collision detection
    eroded_costmap = binary_erosion(costmap.astype(bool), structure=np.ones((3,3))).astype(float)
    
    # Pure black obstacles on white background using exact color codes
    cmap = ListedColormap(['white', COLOR_OBSTACLE])
    plt.imshow(eroded_costmap, origin='lower',
               extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
               cmap=cmap, alpha=1.0, zorder=1, vmin=0, vmax=1)
    
    # 2. Categorize trajectories
    if state_rollouts is not None and state_rollouts.shape[0] > 0:
        # Collision check
        trajectory_collided = check_collision_interpolated_vectorized(
            state_rollouts, sdf_np, resolution, num_interp_points=INTERPOLATION_POINTS
        )
        is_safe = ~trajectory_collided
        
        # Goal reach check
        positions = state_rollouts[:, :, :2]
        distances = np.linalg.norm(positions - goal_state[:2], axis=2)
        reached_goal = np.any(distances <= goal_tolerance, axis=1)
        
        # Categorize trajectories
        successful = is_safe & reached_goal  # Orange: collision-free AND reached goal
        collision_free = is_safe & ~reached_goal  # Blue: collision-free but no goal
        collided = trajectory_collided  # Grey: collided
        
        # Draw trajectories by category using exact color codes
        for k in range(state_rollouts.shape[0]):
            x_traj = state_rollouts[k, :, 0]
            y_traj = state_rollouts[k, :, 1]
            
            if successful[k]:
                plt.plot(x_traj, y_traj, color=COLOR_SELECTED, linewidth=0.8, alpha=0.7, zorder=100)
            elif collision_free[k]:
                plt.plot(x_traj, y_traj, color=COLOR_SAFE, linewidth=0.8, alpha=0.7, zorder=99)
            elif collided[k]:
                plt.plot(x_traj, y_traj, color=COLOR_COLLIDED, linewidth=0.8, alpha=0.5, zorder=98)

    # 3. Start and goal markers
    plt.scatter(0, 0, marker='s', color='green', s=80, zorder=1000, edgecolor='black', linewidth=1)
    if goal_state is not None:
        # Goal as circle with radius = goal_tolerance
        goal_circle = plt.Circle((goal_state[0], goal_state[1]), goal_tolerance, 
                                color='red', alpha=0.3, zorder=1000, linewidth=2, 
                                edgecolor='darkred', fill=True)
        plt.gca().add_patch(goal_circle)

    # 4. Clean formatting - no grid, no labels, minimal
    plt.axis('off')
    plt.gca().set_aspect('equal')
    
    if save_vis and vis_filepath:
        os.makedirs(os.path.dirname(vis_filepath), exist_ok=True)
        plt.savefig(vis_filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()

def create_legend_visualization(output_dir):
    """
    Creates a separate legend file explaining the visualization elements.
    """
    plt.figure(figsize=(6, 4))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
    
    # Create legend elements using exact color codes
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                   markersize=12, markeredgecolor='black', label='Start Position'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=12, markeredgecolor='darkred', alpha=0.3, label='Goal Region (tolerance radius)'),
        plt.Line2D([0], [0], color=COLOR_OBSTACLE, linewidth=3, label='Obstacles'),
        plt.Line2D([0], [0], color=COLOR_SAFE, linewidth=2, label='Collision-free trajectories'),
        plt.Line2D([0], [0], color=COLOR_COLLIDED, linewidth=2, label='Collided trajectories'),
        plt.Line2D([0], [0], color=COLOR_SELECTED, linewidth=2, label='Successful goal reaching (collision-free)')
    ]
    
    plt.legend(handles=legend_elements, loc='center', fontsize=12, frameon=True, 
               fancybox=True, shadow=True)
    plt.axis('off')
    plt.title('Trajectory Visualization Legend', fontsize=14, fontweight='bold', pad=20)
    
    legend_path = os.path.join(output_dir, "LEGEND.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(legend_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Legend saved to: {legend_path}")

# =============================================================================
# 2. Configuration Loading and Preparation
# =============================================================================
def load_yaml_config(config_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading YAML file {config_path}: {e}")
        raise e

def prepare_runtime_config(base_exp_config, env_gt_config):
    """
    Merges the ground truth dynamics configuration (from the dataset) into the
    base experiment configuration. Ensures controllers use the exact same dynamics
    as the GT generation. (Adapted from run_c_safe_uniformity_analysis.py)
    """
    runtime_config = copy.deepcopy(base_exp_config)
    
    try:
        # ASSUMPTION: env_gt_config (from dataset pickle) MUST contain 'dt', 'total_t', 
        # 'vrange', and either 'steering_angle_range' or 'wrange'.
        # 1. Update shared parameters (experiment_config level)
        runtime_config['dt'] = float(env_gt_config['dt'])
        # GT uses 'total_t', controllers use 'horizon_T'.
        runtime_config['horizon_T'] = round(float(env_gt_config['total_t']), 3)
        
        runtime_config['vrange'] = [float(v) for v in env_gt_config['vrange']]
        
        # GT config (steering_angle_range) often uses degrees, config (wrange) uses radians.
        if 'steering_angle_range' in env_gt_config:
            steering_range_deg = env_gt_config['steering_angle_range']
            runtime_config['wrange'] = [np.deg2rad(float(s)) for s in steering_range_deg]
        elif 'wrange' in env_gt_config:
             # Handle datasets where wrange might already be in radians
            runtime_config['wrange'] = [float(w) for w in env_gt_config['wrange']]

        # Update velocity mode based on the new vrange
        vrange = runtime_config['vrange']
        runtime_config['variable_velocity_mode'] = (len(vrange) == 2 and vrange[0] != vrange[1])

        # 2. Update C-Uniform specific parameters (controller_config level)
        if 'cuniform_controller' in runtime_config:
            cu_config = runtime_config['cuniform_controller']
            if 'arange' in env_gt_config:
                cu_config['arange'] = [float(a) for a in env_gt_config['arange']]
            if 'num_a' in env_gt_config:
                cu_config['num_a'] = int(env_gt_config['num_a'])
            if 'num_steering_angle' in env_gt_config:
                cu_config['num_steering_angle'] = int(env_gt_config['num_steering_angle'])

    except (KeyError, ValueError, TypeError) as e:
        raise RuntimeError(f"Error processing GT configuration during merge: {e}. Check keys in GT config.")

    return runtime_config

# =============================================================================
# 3. Controller Wrappers (Dependency Injection for Stateless Analysis)
# =============================================================================
class AnalysisTorchPlannerBase(TorchPlannerBase):
    """
    A wrapper for TorchPlannerBase that overrides the perception pipeline
    and allows runtime modification of the sampling budget.
    """
    # ASSUMPTION: TorchPlannerBase initialization is successful and sets up necessary attributes.
    def set_precomputed_perception(self, perception_data):
        self.precomputed_perception = perception_data
        # Update the internal state used by the cost calculation
        self.local_costmap_map = perception_data.get('inflated_costmap')
        # Clear tensor cache to force regeneration based on the new map
        if hasattr(self, 'local_costmap_tensor'):
            self.local_costmap_tensor = None
        if hasattr(self, 'convolved_costmap_tensor'):
            self.convolved_costmap_tensor = None

    def override_sampling_budget(self, new_budget):
        """Override the sampling budget (K) at runtime."""
        self.num_rollouts = new_budget
        
        # Update specific internal parameters for different controller types
        if hasattr(self, 'num_trajectories'): # C-Uniform
            self.num_trajectories = new_budget
        if hasattr(self, 'K'):  # MPPI
            self.K = new_budget
        
        # ensure visualization limit matches the sampling budget.
        # This forces the controller to return ALL samples in 'state_rollouts', 
        # allowing accurate evaluation of whether ANY sample reached the goal.
        if hasattr(self, 'num_vis_rollouts'):
            self.num_vis_rollouts = new_budget
        if hasattr(self, 'num_vis_trajectories'):
             self.num_vis_trajectories = new_budget

    def _prepare_perception_inputs(self, global_occupancy_grid, current_state):
        # ASSUMPTION: For this analysis harness, perception MUST be injected beforehand.
        # If this fails, the experimental setup is incorrect.
        if hasattr(self, 'precomputed_perception') and self.precomputed_perception:
            return self.precomputed_perception
        raise RuntimeError("Analysis Error: _prepare_perception_inputs called before perception data was injected.")


# Define specific controller wrappers using Multiple Inheritance
class AnalysisCUniformController(CUniformController, AnalysisTorchPlannerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precomputed_perception = None

class AnalysisMPPIPyTorchController(MPPIPyTorchController, AnalysisTorchPlannerBase):
     def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precomputed_perception = None

# Define the Controller Factory
CONTROLLERS_FACTORY = {
    "C-Free-Uniform": (
        AnalysisCUniformController, "cuniform_controller", {"type_override": 1}
    ),
    "C-Uniform": (
        AnalysisCUniformController, "cuniform_controller", {"type_override": 0}
    ),
    "MPPI": (
        AnalysisMPPIPyTorchController, "mppi_controller", {"type_override": 0}
    ),
    "Log-MPPI": (
        AnalysisMPPIPyTorchController, "mppi_controller", {"type_override": 1}
    ),
}

# =============================================================================
# 4. Helper Functions (Perception, Goal Selection, Evaluation)
# =============================================================================
def get_perception_input(scenario_path, mode='dataset'):
    """
    modular function to provide perception data (SDF, Costmap).
    """
    # ASSUMPTION: If mode='dataset', 'sdf.npy' and 'costmap.npy' MUST exist and be valid numpy files.
    if mode == 'dataset':
        try:
            sdf_path = os.path.join(scenario_path, "sdf.npy")
            costmap_path = os.path.join(scenario_path, "costmap.npy")

            if not os.path.exists(sdf_path) or not os.path.exists(costmap_path):
                raise FileNotFoundError(f"SDF or Costmap missing in {scenario_path}")

            local_sdf = np.load(sdf_path)
            # The dataset 'costmap.npy' is used as the 'inflated_costmap' for planning costs.
            local_costmap = np.load(costmap_path).astype(np.float32)
            
            # Convert SDF to the required tensor format (B, C, H, W)
            sdf_tensor = torch.from_numpy(local_sdf).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

            perception_data = {
                'binary_costmap': local_costmap,
                'inflated_costmap': local_costmap, 
                'sdf_tensor': sdf_tensor,
                'sdf_np': local_sdf # keep numpy version for collision checking
            }
            return perception_data

        except Exception as e:
            print(f"  Error loading perception data from {scenario_path}: {e}")
            # Re-raise the exception to stop processing this scenario if data is missing/corrupt.
            raise e
    
    elif mode == 'ground_truth':
        # Placeholder for future implementation
        raise NotImplementedError("Ground truth perception mode is not yet implemented.")
    else:
        raise ValueError(f"Unknown perception mode: {mode}")

def select_goal_state(scenario_path, rng):
    """
    loads GT data and selects a reachable goal from the final level set.
    """
    # ASSUMPTION: 'uniformity_gt.pkl' MUST exist and be a valid pickle file.
    try:
        gt_path = os.path.join(scenario_path, "uniformity_gt.pkl")
        with open(gt_path, 'rb') as f:
            env_gt = pickle.load(f)
    except Exception as e:
        print(f"  Error loading GT data from {gt_path}: {e}")
        raise e

    # Handle potential key variations in the dataset
    level_set_key = 'level_set_representatives'
    if level_set_key not in env_gt:
        level_set_key = 'pruned_level_set_representatives_across_LS' # Alternative key

    level_sets = env_gt.get(level_set_key)
    
    if not level_sets:
        raise ValueError(f"  Warning: GT data has no level sets in {scenario_path}.")

    # Find the last non-empty level set
    final_level_set = None
    for level_set in reversed(level_sets):
        if level_set is not None and len(level_set) > 0:
            final_level_set = level_set
            break
    
    if final_level_set is None:
        # Environment is completely unreachable
        raise ValueError(f"  Warning: GT data has no last level set in {scenario_path}.")

    # Convert set/list to numpy array
    if isinstance(final_level_set, set):
        final_states = np.array(list(final_level_set))
    else:
        # Handle potential tuples in the list
        final_states = np.array([list(s) if isinstance(s, tuple) else s for s in final_level_set])

    # Randomly sample one state (x, y, theta)
    goal_index = rng.choice(len(final_states))
    goal_state = final_states[goal_index]
    
    # Basic validation (Ensure at least x, y, theta)
    if len(goal_state) < 3:
         # Attempt recovery if only (x,y) present
         if len(goal_state) == 2:
             goal_state = np.append(goal_state, 0.0)
         else:
            raise ValueError(f"  Warning: GT state has invalid dimensions in {scenario_path}.")

    return goal_state[:3], env_gt

def is_successful(state_rollouts, goal_state, goal_tolerance, sdf, resolution):
    """ Evaluates Safe Reachability.
    Success if ANY trajectory is both collision-free AND reaches the goal at ANY point.
    """
    if state_rollouts is None or state_rollouts.shape[0] == 0:
        return False
    
    # ASSUMPTION: state_rollouts must be 3D (K, T+1, D) and D >= 2.
    assert state_rollouts.ndim == 3 and state_rollouts.shape[2] >= 2, f"Invalid state_rollouts shape: {state_rollouts.shape}"

    # 1. Collision Check (Swept Volume)
    # (K,) boolean array, True if the trajectory collides.
    trajectory_collided = check_collision_interpolated_vectorized(
        state_rollouts, sdf, resolution, num_interp_points=INTERPOLATION_POINTS
    )
    is_safe = ~trajectory_collided

    # 2. Goal Reach Check (Along the trajectory)
    positions = state_rollouts[:, :, :2] # (K, T+1, 2)
    distances = np.linalg.norm(positions - goal_state[:2], axis=2) # (K, T+1)
    # (K,) boolean array, True if the goal was reached at any point.
    reached_goal = np.any(distances <= goal_tolerance, axis=1)

    # 3. Combine Criteria: Safe AND Reached Goal, then check if ANY succeeded.
    success = np.any(is_safe & reached_goal)
    return success

# =============================================================================
# 5. Core Analysis Logic
# =============================================================================

def run_analysis(planning_config, base_exp_config, output_dir):
    """Main loop for the single-frame planning analysis."""
    
    # Initialization
    seed = planning_config['analysis_seed']
    # np.random.seed(seed) # We use specific RNGs below for determinism
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Resolve Dataset Path
    dataset_path_config = planning_config['dataset_path']
    if os.path.isabs(dataset_path_config):
        dataset_path = dataset_path_config
    else:
        # Assume path is relative to PROJECT_ROOT if not absolute
        dataset_path = os.path.join(PROJECT_ROOT, dataset_path_config)

    budgets = sorted(planning_config['sampling_budgets']) # Ensure sorted order
    samplers = planning_config['samplers_to_test']
    goal_tolerance = planning_config['goal_tolerance']
    num_mc_trials = planning_config['num_mc_trials'] # Get MC trials count

    # Discover Scenarios
    if not os.path.exists(dataset_path):
        print(f"Error: Input dataset directory not found: {dataset_path}"); return
    
    scenario_dirs = []
    # Use sorted list for deterministic processing order
    for item in sorted(os.listdir(dataset_path)):
        full_path = os.path.join(dataset_path, item)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "uniformity_gt.pkl")):
             scenario_dirs.append(full_path)

    if not scenario_dirs:
        print("No valid scenario directories found."); return

    print(f"\nFound {len(scenario_dirs)} scenarios to analyze.")
    print(f"Running {num_mc_trials} Monte Carlo trials per configuration.")
    
    # Results storage: results[sampler][budget][scenario_idx] = average success rate (float)
    results = defaultdict(lambda: defaultdict(list))

    enable_visualization = planning_config.get('visualization_debug', False)

    # --- Main Loop ---
    # Calculate total configurations (excluding MC trials) for progress tracking if needed
    total_configs = len(budgets) * len(samplers) * len(scenario_dirs)
    config_count = 0

    start_time = time.time()

    # Optimal Loop Structure: Scenario -> Sampler -> Budget -> MC Trial
    # This minimizes data loading and configuration switching overhead.
    for scenario_idx, scenario_path in enumerate(scenario_dirs):
        scenario_name = os.path.basename(scenario_path)
        print(f"\n--- Analyzing Scenario {scenario_idx+1}/{len(scenario_dirs)}: {scenario_name} ---")

        # 1. Load Scenario Data and Select Goal
        # CRITICAL: Use a deterministic RNG seeded by scenario index for goal selection.
        try:
            scenario_goal_rng = np.random.RandomState(seed + scenario_idx)
            goal_state, env_gt = select_goal_state(scenario_path, scenario_goal_rng)
        except Exception as e:
            print(f"  Skipping Scenario: Failed during goal selection/GT loading. Error: {e}"); continue

        if goal_state is None: 
            raise RuntimeError("Unreachable environment. This should never happen.")

        # 2. Prepare Perception Input
        try:
            perception_data = get_perception_input(scenario_path, mode='dataset')
            # Extract numpy SDF for collision checking
            local_sdf_np = perception_data['sdf_np']
        except Exception as e:
            print(f"  Skipping Scenario: Failed during perception loading. Error: {e}"); continue

        # 3. Prepare Runtime Configuration (Sync dynamics with GT)
        try:
            runtime_config = prepare_runtime_config(base_exp_config, env_gt['config'])
            # Extract resolution for visualization and collision checking
            local_resolution = runtime_config.get('local_costmap_resolution', 0.05)
        except RuntimeError as e:
            print(f"  Skipping: Configuration error - {e}"); continue

        # Define Start State (Origin in local frame)
        initial_velocity = env_gt['config']['vrange'][0]
        start_state = np.array([0.0, 0.0, 0.0, initial_velocity])
        planner_goal_xy = goal_state[:2] 

        # 4. Iterate over Samplers
        for sampler_name in samplers:
            if sampler_name not in CONTROLLERS_FACTORY:
                print(f"  Warning: Sampler '{sampler_name}' not found in factory. Skipping."); continue
            
            print(f"  Testing Sampler: {sampler_name}")
            ControllerClass, config_section, overrides = CONTROLLERS_FACTORY[sampler_name]

            # 5. Iterate over Budgets
            for budget in budgets:
                config_count += 1
                # --- Monte Carlo Loop ---
                mc_successes = []
                for mc_trial_idx in range(num_mc_trials):
                    # Prepare specific configuration for this trial
                    # We must ensure the configuration reflects the current budget for correct initialization logs.
                    trial_specific_config = copy.deepcopy(runtime_config)
                    trial_specific_config['num_rollouts'] = budget
                    # Ensure visualization budget matches the actual budget for evaluation
                    trial_specific_config['num_vis_rollouts'] = budget

                    # This ensures a fresh initialization (cold start) for every single run.
                    # Use a unique, deterministic seed for this specific run
                    # (Scenario, Sampler, Budget, MC_Trial) combination
                    # Use large multipliers to ensure unique seeds.
                    sampler_seed = (seed + scenario_idx * 100000 + 
                                    list(CONTROLLERS_FACTORY.keys()).index(sampler_name) * 10000 + 
                                    budgets.index(budget) * 100 + 
                                    mc_trial_idx)
                    
                    try:
                        controller = ControllerClass(
                            controller_config=trial_specific_config[config_section],
                            experiment_config=trial_specific_config,
                            **overrides,
                            seed=sampler_seed,
                        )
                        
                        # ASSUMPTION: Controller must have these methods.
                        if not hasattr(controller, 'reset') or not hasattr(controller, 'get_control_action'):
                            raise AttributeError(f"Controller {sampler_name} is missing required methods.")

                        # Force MPPI controllers to use exactly 1 iteration.
                        # This evaluates the single-step sampling performance, not iterative optimization.
                        if hasattr(controller, 'mppi_iterations'):
                            controller.mppi_iterations = 1

                        controller.reset()
                        # Inject the pre-computed perception data
                        controller.set_precomputed_perception(perception_data)

                    except Exception as e:
                        print(f"    Error initializing controller {sampler_name} @ K={budget}, MC={mc_trial_idx+1}: {e}")
                        mc_successes.append(False)
                        gc.collect(); 
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue

                    # Run One Planning Step
                    try:
                        # Pass dummy map (None) as perception is injected. Use runtime dt
                        # Note: get_control_action requires a map argument, even if None/dummy
                        dummy_map = np.zeros((10,10), dtype=np.int8)
                        _, info = controller.get_control_action(
                            start_state, planner_goal_xy, dummy_map, trial_specific_config['dt']
                        )
                        
                        # Extract rollouts. Because we correctly overrode the visualization budget
                        # this contains ALL sampled trajectories.
                        state_rollouts = info.get('state_rollouts')
                        
                        # ASSUMPTION: The controller MUST return 'state_rollouts' in the info dict
                        if state_rollouts is None:
                            raise RuntimeError("Controller did not return 'state_rollouts'.")
                        
                        # Evaluate Success, check for Safe Reachability
                        success = is_successful(
                            state_rollouts, goal_state, goal_tolerance, local_sdf_np, local_resolution
                        )

                    except Exception as e:
                        print(f"    Error during planning step for {sampler_name} @ K={budget}, MC={mc_trial_idx+1}: {e}")
                        success = False
                        state_rollouts = None # Ensure rollouts is None if planning failed
                    
                    mc_successes.append(success)

                    # Minimal Visualization
                    # Only visualize the FIRST Monte Carlo trial if enabled
                    if enable_visualization and state_rollouts is not None and mc_trial_idx == 0:
                        # Status and trial info in filename instead of title
                        status_str = "SUCCESS" if success else "FAILURE"
                        
                        # Sanitize names for filename
                        safe_sampler_name = sampler_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                        # Include all trial info in filename: scenario_sampler_K512_MC1_SUCCESS.png
                        vis_dir = os.path.join(output_dir, "visualizations")
                        vis_filename = f"{scenario_name}_{safe_sampler_name}_K{budget}_MC{mc_trial_idx+1}_{status_str}.png"
                        vis_filepath = os.path.join(vis_dir, vis_filename)

                        # Generate minimal visualization
                        visualize_trajectories_minimal(
                            state_rollouts=state_rollouts, 
                            costmap=perception_data['inflated_costmap'], 
                            resolution=local_resolution, 
                            goal_state=goal_state, 
                            goal_tolerance=goal_tolerance,
                            sdf_np=local_sdf_np,
                            save_vis=True, 
                            vis_filepath=vis_filepath
                        )
                    
                    # Cleanup controller instance immediately after use
                    del controller
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Aggregate Results (Requirement 3G - Updated)
                # Calculate the average success rate over the MC trials
                if mc_successes:
                    average_success_rate = np.mean(mc_successes)
                else:
                    average_success_rate = 0.0
                
                results[sampler_name][budget].append(average_success_rate)
                

    end_time = time.time()
    print(f"\nAnalysis complete. Total time: {end_time - start_time:.2f} seconds.")
    return results

# =============================================================================
# 6. Reporting
# =============================================================================
def generate_report(results, planning_config, output_dir):
    """aggregates results and generates the final LaTeX table."""
    print("\nGenerating final report...")
    
    samplers = planning_config['samplers_to_test']
    budgets = sorted(planning_config['sampling_budgets'])
    num_mc_trials = planning_config['num_mc_trials']
    
    # Calculate Success Rates
    summary = defaultdict(dict)
    
    # Determine the total number of scenarios processed
    max_len = 0
    for sampler_data in results.values():
        for budget_data in sampler_data.values():
            max_len = max(max_len, len(budget_data))
    num_scenarios = max_len

    if num_scenarios == 0:
        print("No results recorded. Cannot generate report."); return

    for sampler_name in samplers:
        if sampler_name not in results: continue
        
        for budget in budgets:
            if budget not in results[sampler_name]: continue
            
            # This list contains the average success rates (floats) from the MC trials for each scenario
            mc_avg_success_rates = results[sampler_name][budget]
            
            # Handle potential discrepancies if some trials failed initialization (pad with failures)
            if len(mc_avg_success_rates) < num_scenarios:
                raise RuntimeError(f"Number of MC trials ({len(mc_avg_success_rates)}) is less than the number of scenarios ({num_scenarios}).")
                failures_to_add = num_scenarios - len(mc_avg_success_rates)
                mc_avg_success_rates.extend([0.0] * failures_to_add)
            
            # Calculate the final success rate by averaging the MC-averaged rates over all scenarios
            final_success_rate = np.mean(mc_avg_success_rates)
            summary[sampler_name][budget] = final_success_rate

    # Generate LaTeX Table
    headers = ["Sampler"] + [f"K={b}" for b in budgets]
    table_data = []

    for sampler_name in samplers:
        # Show sampler in report even if it wasn't successfully run
        if sampler_name not in summary:
            row = [sampler_name] + ["-"] * len(budgets)
            table_data.append(row)
            continue
        
        row = [sampler_name]
        for budget in budgets:
            rate = summary[sampler_name].get(budget)
            if rate is not None:
                # Format as percentage for LaTeX (using \%)
                row.append(f"{rate*100:.1f}\%") 
            else:
                row.append("-")
        table_data.append(row)

    # Use the 'latex_raw' format from the tabulate library
    latex_table = tabulate(table_data, headers=headers, tablefmt="latex_raw")
    
    # Add preamble/title to the report
    report_header = f"""
=================================================================
          Single-Frame Planning Analysis: Final Report
=================================================================
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Total Scenarios Analyzed: {num_scenarios}
Monte Carlo Trials per Configuration: {num_mc_trials}
Goal Tolerance: {planning_config['goal_tolerance']}m
Seed: {planning_config['analysis_seed']}
Dataset: {planning_config['dataset_path']}

--- Success Rate vs. Sampling Budget (K) ---
--- Safe Reachability Success Rate vs. Sampling Budget (K) ---
(Success defined as: At least one trajectory is collision-free AND reaches the goal)
(Results averaged over {num_mc_trials} MC trials per scenario, then averaged over {num_scenarios} scenarios)
"""
    
    print(report_header)
    print(latex_table)
    
    # Save to file
    report_path = os.path.join(output_dir, "planning_success_report.txt")
    try:
        with open(report_path, 'w') as f:
            f.write(report_header)
            f.write("\n\n")
            f.write(latex_table)
        print(f"\nFinal report saved to: {report_path}")
    except Exception as e:
        print(f"Error saving final report: {e}")

# =============================================================================
# 7. Main Execution
# =============================================================================
def main(config_file_path):
    print("="*80)
    print("SINGLE-FRAME PLANNING ANALYSIS HARNESS")
    print("="*80)

    # 1. Load Planning Configuration
    # Resolve the path relative to the script directory if it's not absolute
    if not os.path.isabs(config_file_path):
         config_file_path = os.path.abspath(os.path.join(SCRIPT_DIR, config_file_path))

    planning_config = load_yaml_config(config_file_path)
    if not planning_config: return

    # 2. Load Base Experiment Configuration
    base_config_path_config = planning_config['base_experiment_config_path']
    
    if os.path.isabs(base_config_path_config):
        base_config_path = base_config_path_config
    else:
        # Assume path is relative to PROJECT_ROOT if not absolute
        base_config_path = os.path.join(PROJECT_ROOT, base_config_path_config)

    base_exp_config = load_yaml_config(base_config_path)
    if not base_exp_config: sys.exit(1)

    # 3. Define Output Directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(SCRIPT_DIR, f"results_planning_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Save copies of the configurations used
    try:
        with open(os.path.join(output_dir, 'planning_config_copy.yaml'), 'w') as f:
            yaml.dump(planning_config, f)
        with open(os.path.join(output_dir, 'base_experiment_config_copy.yaml'), 'w') as f:
            yaml.dump(base_exp_config, f)
    except Exception as e:
        print(f"Warning: Failed to save configuration copies: {e}")

    # 4. Create Legend First
    create_legend_visualization(output_dir)
    
    # 5. Run Analysis
    results = run_analysis(planning_config, base_exp_config, output_dir)

    # 6. Generate Report
    if results:
        generate_report(results, planning_config, output_dir)
    
    print("\nAnalysis script finished.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Single-Frame Planning Analysis.")
    
    # Default path assumes the standard structure: single_frame_analysis/configs/planning_config.yaml
    default_config_path = os.path.join("configs", "planning_config.yaml")
    
    parser.add_argument('--config', type=str, default=default_config_path,
                        help=f'Path to the planning configuration YAML file (default: {default_config_path}). Path is relative to the script directory.')
    
    # Robust argument parsing
    try:
        # Use parse_known_args to handle environments that might pass extra arguments
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"Warning: Unrecognized arguments ignored: {unknown}")
    except SystemExit:
        # Handle cases where argparse might exit (e.g., -h)
        print("Info: Argument parsing caused exit.")
        sys.exit(0)

    main(args.config)