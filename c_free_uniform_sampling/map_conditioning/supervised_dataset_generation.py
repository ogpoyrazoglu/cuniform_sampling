import os
import sys
import torch
import contextlib
import random
import numpy as np
import pickle
import time
import math
from flow_Cuniform.dynamics_helpers import (
    dynamics_KS_3d_steering_angle,
    dynamics_KS_4d_steering_angle_v,

    inverse_dynamics_KS_4d_steering_angle_v_vectorized,
    inverse_dynamics_KS_3d_steering_angle,

    vectorized_dynamics_KS_4d_steering_angle_v,
    vectorized_dynamics_KS_3d_steering_angle,

    cuda_dynamics_KS_3d_steering_angle_vectorized,
    cuda_dynamics_KS_4d_steering_angle_v_vectorized,
)
from flow_Cuniform.utility_helpers import (
    generate_actions,
    precompute_graph_structure_parallel,
    prune_graph,
    calculate_reachable_level_sets,
    calculate_reachable_level_sets_adaptive_uniformity,
    setup_single_transition_flow,
    flow_to_action_prob_approximation_all,
)
from flow_Cuniform.c_uniform_sampling import(
    parallelized_collision_checker_nodes,
)
from classes.grid import Grid, AdaptiveGrid
from utility_helper_map import (
    visualize_trajectories_background, 
    visualize_sdf, 
    save_level_set_visualization,
    create_sdf_from_obstacles,
    parallelized_trajectory_sampling_cuda
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# NOTE: make sure the configuration is same everywhere
model_configs = {
    "KS_3D_STEERING_ANGLE": {
        "model_name": "KS_3D_STEERING_ANGLE",
        "dynamics": dynamics_KS_3d_steering_angle,
        "inverse_dynamics": inverse_dynamics_KS_3d_steering_angle,
        "vectorized_dynamics": vectorized_dynamics_KS_3d_steering_angle,
        "perturbation_param": 2.01,
        "slack_parameter": 0.00,
        "vrange": (2.5, 2.5),
        "arange": (0.0, 0.0),
        "num_a": 1,
        "steering_angle_range": (-30.0, 30.0),
        "num_steering_angle": 31,
        "actions": None,
        # "thresholds": [0.05, 0.05, (2 * math.pi)/60],
        "thresholds": [0.10, 0.10, (2 * math.pi)/60],
        "state_dim": 3,
        "dt": 0.20,
        "total_t": 1.21,
        "initial_state_set": np.array([[0.0, 0.0, 0.0]]),  # Robot always starts at origin in robot-local frame
        "adaptive_uniformity": False,
        "sdf_inflation": 0.0,  # Safety margin for collision detection (meters)
    },
    "KS_4D_STEERING_ANGLE_V": {
        "model_name": "KS_4D_STEERING_ANGLE_V",
        "dynamics": dynamics_KS_4d_steering_angle_v,
        "inverse_dynamics": inverse_dynamics_KS_4d_steering_angle_v_vectorized, 
        "vectorized_dynamics": vectorized_dynamics_KS_4d_steering_angle_v,
        "perturbation_param": 2.01,          # offsets add to representative at level set t
        "vrange": (0.0, 4.0),                # Velocity range (min, max)
        "arange": (-6.0, 6.0),               # acceleration range (min, max)
        # "arange": (-6.5, 5.3),             # acceleration range (min, max)
        "num_a": 11,
        "steering_angle_range": (-30.0, 30.0),   # Steering range in degrees
        "num_steering_angle": 15,
        "actions": None,                     # placeholder, will define later
        "thresholds": [0.05, 0.05, (2 * math.pi)/80, 0.25],
        "state_dim": 4,                      # (x, y, theta - yaw angle, velocity)
        "dt" : 0.10,
        "total_t" : 1.51,
        "initial_state_set": np.array([[0.0, 0.0, 0.0, 0.0]]),  # Single initial state, will be modified dynamically
        "adaptive_uniformity": True,
        "sdf_inflation": 0.0,  # Safety margin for collision detection (meters)
    },
}

# System Configuration
config = model_configs["KS_3D_STEERING_ANGLE"]
# config = model_configs["KS_4D_STEERING_ANGLE_V"]

assert config["model_name"] == "KS_3D_STEERING_ANGLE", f"BARN dataset requires KS_3D_STEERING_ANGLE model, got {config['model_name']}"
print("="*80)
print("SYSTEM CONFIGURATION")
print("="*80)
print(f" Model: {config['model_name']}")
print(f" State Dimension: {config['state_dim']}")
print(f"  Time Step (dt): {config['dt']}")
print(f" Total Time: {config['total_t']}")
print(f"  Velocity Range: {config['vrange']}")
if config['state_dim'] == 4:
    print(f"⚡ Acceleration Range: {config['arange']}")
    print(f" Number of Accelerations: {config['num_a']}")
print(f" Steering Range: {config['steering_angle_range']} degrees")
print(f" Number of Steering Angles: {config['num_steering_angle']}")
print(f" Thresholds: {config['thresholds']}")
print(f" Adaptive Uniformity: {config['adaptive_uniformity']}")
print(f"  SDF Inflation: {config['sdf_inflation']}m")
print(f"📍 Initial State: {config['initial_state_set'][0]}")

config["actions"] = generate_actions(
    config["arange"], config["num_a"], config["steering_angle_range"], 
    config["num_steering_angle"], deg2rad_conversion=True
)

print(f" Total Actions Generated: {len(config['actions'])}")
print("="*80)
print()

# Global control for terminal output suppression
SUPPRESS_DETAILED_OUTPUT = True # Set to True to hide detailed computation logs

def conditional_stdout_suppress():
    """Context manager that suppresses stdout only if SUPPRESS_DETAILED_OUTPUT is True"""
    if SUPPRESS_DETAILED_OUTPUT:
        return contextlib.redirect_stdout(None)
    else:
        return contextlib.nullcontext()  # Does nothing, preserves output

def save_config_to_txt(config, env_folder, filename="config_info.txt"):
    """
    Save configuration information to a readable text file.
    
    Args:
        config (dict): Model configuration dictionary
        env_folder (str): Environment folder path
        filename (str): Name of the config file to save
    """
    config_file_path = os.path.join(env_folder, filename)
    
    try:
        with open(config_file_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATASET CONFIGURATION\n")
            f.write("="*80 + "\n\n")
            
            f.write(" MODEL CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            
            # Core model settings
            important_keys = [
                'model_name', 'state_dim', 'dt', 'total_t', 
                'vrange', 'arange', 'num_a', 'steering_angle_range', 
                'num_steering_angle', 'thresholds', 'perturbation_param',
                'slack_parameter', 'adaptive_uniformity', 'sdf_inflation'
            ]
            
            for key in important_keys:
                if key in config:
                    value = config[key]
                    f.write(f"{key:25}: {value}\n")
            
            # Special handling for arrays
            if 'actions' in config and config['actions'] is not None:
                actions = config['actions']
                f.write(f"{'actions':25}: ndarray with shape {actions.shape}\n")
                f.write(f"{' '*25}  Range: [{actions.min():.4f}, {actions.max():.4f}]\n")
                if len(actions) <= 10:
                    f.write(f"{' '*25}  All values:\n")
                    for i, action in enumerate(actions):
                        f.write(f"{' '*27}  [{i:2d}]: {action}\n")
                else:
                    f.write(f"{' '*25}  First 5:\n")
                    for i in range(5):
                        f.write(f"{' '*27}  [{i:2d}]: {actions[i]}\n")
                    f.write(f"{' '*25}  Last 5:\n")
                    for i in range(len(actions)-5, len(actions)):
                        f.write(f"{' '*27}  [{i:2d}]: {actions[i]}\n")
            
            if 'initial_state_set' in config and config['initial_state_set'] is not None:
                initial_states = config['initial_state_set']
                f.write(f"{'initial_state_set':25}: ndarray with shape {initial_states.shape}\n")
                f.write(f"{' '*25}  Values: {initial_states}\n")
            
            # Add function names for reference (without printing the actual function objects)
            f.write(f"\n FUNCTION REFERENCES:\n")
            f.write("-" * 40 + "\n")
            function_keys = ['dynamics', 'inverse_dynamics', 'vectorized_dynamics']
            for key in function_keys:
                if key in config:
                    func = config[key]
                    f.write(f"{key:25}: {func.__name__ if hasattr(func, '__name__') else str(type(func))}\n")
            
            # Additional derived information
            f.write(f"\n DERIVED INFORMATION:\n")
            f.write("-" * 40 + "\n")
            if 'dt' in config and 'total_t' in config:
                num_steps = int(config['total_t'] / config['dt'])
                f.write(f"{'estimated_time_steps':25}: {num_steps}\n")
            
            if 'actions' in config and config['actions'] is not None:
                f.write(f"{'total_actions':25}: {len(config['actions'])}\n")
            
            # Timestamp
            import datetime
            f.write(f"\n GENERATION INFO:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'generated_at':25}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'script_version':25}: supervised_dataset_generation.py\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"  ✓ Configuration saved to {filename}")
        
    except Exception as e:
        print(f"    Warning: Could not save config file: {e}")

def create_environment_folder(env_name, base_dir="dataset_supervised"):
    """
    Create an environment folder under base_dir within the map_conditioning directory.
    """
    # Get the directory where this script is located (map_conditioning)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create absolute path to dataset_supervised within map_conditioning
    absolute_base_dir = os.path.join(script_dir, base_dir)
    env_folder = os.path.join(absolute_base_dir, env_name)
    os.makedirs(env_folder, exist_ok=True)
    print(f"Created folder: {env_folder}")
    return env_folder

def compute_action_probabilities_for_environment(config, obstacles, sdf, resolution, initial_state=None):
    """
    Compute action probabilities for states in each level set for a given environment.
    This is the supervised learning version that focuses on storing action probabilities rather than level set representatives.
    Also computes and saves level set representatives and boundary states.
    
    Args:
        config (dict): Model configuration (dynamics, actions, etc.).
        obstacles (list): List of obstacles.
        sdf (np.ndarray): Signed distance function array.
        resolution (float): Grid resolution in meters.
        initial_state (np.ndarray, optional): Specific initial state to use. If None, uses config's initial_state_set.
    
    Returns:
        tuple: (action_prob_data dict, pruned_graphs list) or (None, None) if failed
    """
    start_time = time.time()
    print("Computing action probabilities for environment...")
    
    # Use specific initial state if provided, otherwise use config's initial state set
    config_for_computation = config.copy()
    if initial_state is not None:
        config_for_computation["initial_state_set"] = np.array([initial_state])
    
    # Step 1: Compute reachable level sets 
    level_sets_start = time.time()
    print("  Computing reachable level sets...")
    with conditional_stdout_suppress():
        # Use different function based on state dimension
        if config_for_computation['state_dim'] == 4:
            ReaBoxIndices_LSs, ReaBox_LSs, _, _ = calculate_reachable_level_sets_adaptive_uniformity(
                config_for_computation,
                disjoint_level_set=True,
                obstacles=obstacles,
                sdf=sdf,
                collision_checker_nodes=lambda *args, **kwargs: parallelized_collision_checker_nodes(
                    *args, resolution=resolution, sdf_inflation=config_for_computation['sdf_inflation'], **kwargs
                ),
            )
        else:
            ReaBoxIndices_LSs, ReaBox_LSs, _, _ = calculate_reachable_level_sets(
                config_for_computation,
                disjoint_level_set=True,
                adaptive_resolution=True,
                obstacles=obstacles,
                sdf=sdf,
                collision_checker_nodes=lambda *args, **kwargs: parallelized_collision_checker_nodes(
                    *args, resolution=resolution, sdf_inflation=config_for_computation['sdf_inflation'], **kwargs
                ),
            )
        if ReaBoxIndices_LSs is None and ReaBox_LSs is None:
            init_state = config_for_computation['initial_state_set'][0]
            print(f"   Level set failed: state={init_state[:3]}, v={config_for_computation['vrange']}, obstacles={len(obstacles) if obstacles else 0}")
            return None, None
    level_sets_time = time.time() - level_sets_start
    print(f"  ✓ Level sets computed ({level_sets_time:.2f}s)")
    
    # Step 2: Build graph structure
    graph_start = time.time()
    print("  Building graph structure...")
    with conditional_stdout_suppress():
        raw_graphs, _ = precompute_graph_structure_parallel(ReaBox_LSs, ReaBoxIndices_LSs, config_for_computation)
    graph_time = time.time() - graph_start
    print(f"  ✓ Graph structure built ({graph_time:.2f}s)")
    
    # Step 3: Prune graphs
    prune_start = time.time()
    print("  Pruning graphs...")
    with conditional_stdout_suppress():
        pruned_graphs = prune_graph(raw_graphs)
    prune_time = time.time() - prune_start
    print(f"  ✓ Graphs pruned ({prune_time:.2f}s)")
    
    # Extract pruned level set representatives 
    pruned_level_set_representatives = []
    for i in range(len(pruned_graphs)):
        representatives = {node.point for node in pruned_graphs[i]["nodes"].values() if node.level_set == i}
        pruned_level_set_representatives.append(list(representatives))
        if len(representatives) == 0:
            print(f"    Warning: Level {i} has no representatives after pruning")
    
    # Add the final level set
    last_level = len(pruned_graphs)
    last_level_set = {
        node.point 
        for node in pruned_graphs[-1]["nodes"].values()
        if node.level_set == last_level
    }
    pruned_level_set_representatives.append(list(last_level_set))
    
    # Step 4: Compute network flow 
    flow_start = time.time()
    print("  Computing network flow...")
    # setup_single_transition_flow modifies the graph in-place
    with conditional_stdout_suppress():
        for graph in pruned_graphs:
            _ = setup_single_transition_flow(graph)
    flow_time = time.time() - flow_start
    print(f"  ✓ Network flow computed ({flow_time:.2f}s)")
    
    # Step 5: Compute action probabilities from flow
    action_prob_start = time.time()
    print("  Computing action probabilities...")
    with conditional_stdout_suppress():
        probabilities_list = flow_to_action_prob_approximation_all(
            graphs=pruned_graphs, model_config=config_for_computation
        )
    action_prob_time = time.time() - action_prob_start
    print(f"  ✓ Action probabilities computed ({action_prob_time:.2f}s)")
    
    # Step 6: Identify boundary states using forward propagation
    boundary_start = time.time()
    print("  Computing boundary states...")
    boundary_states = identify_boundary_states(
        pruned_level_set_representatives, config_for_computation
    )
    
    # Step 7: Add action probabilities to boundary states
    boundary_states_with_probs = add_action_probabilities_to_boundary_states(
        boundary_states['boundary_states'],
        pruned_level_set_representatives,
        probabilities_list,
        config_for_computation
    )
    boundary_time = time.time() - boundary_start
    print(f"  ✓ Boundary states computed ({boundary_time:.2f}s)")
    
    # Step 8: Organize the data for supervised learning
    action_prob_data = {
        'level_sets': probabilities_list,
        'level_set_representatives': pruned_level_set_representatives,
        'boundary_states': boundary_states,
        'boundary_states_with_probs': boundary_states_with_probs,
        'config': config_for_computation,
    }
    total_time = time.time() - start_time
    
    print(f" Action probabilities completed ({total_time:.2f}s)")
    print(f"   └─ Levels: {level_sets_time:.1f}s, Graph: {graph_time:.1f}s, Flow: {flow_time:.1f}s, Boundary: {boundary_time:.1f}s")
    
    return action_prob_data, pruned_graphs

def add_action_probabilities_to_boundary_states(boundary_states, level_set_representatives, probabilities_list, config):
    """
    Add action probabilities to boundary states by finding the closest representative state
    and using its action probabilities. Uses batched processing to avoid memory overflow.
    
    Args:
        boundary_states: List of boundary states
        level_set_representatives: List of level sets with representatives
        probabilities_list: List of level sets with action probabilities
        config: Model configuration
        
    Returns:
        list: Boundary states with action probabilities in the same format as regular samples
    """
    if len(boundary_states) == 0:
        return []
    
    boundary_states_with_probs = []
    state_dim = config['state_dim']
    tolerance = 1e-5  # Tolerance for floating point comparison
    batch_size = 500 # Process boundary states in batches to control memory
    
    # STEP 1: Create mapping from probabilities_list for efficient lookup
    prob_state_to_probs = {}  # Maps state tuple to action probabilities
    
    for level_idx, level_probs in enumerate(probabilities_list):
        if level_probs is None or len(level_probs) == 0:
            continue
            
        # Each level_probs is a numpy array of shape (N, state_dim + num_actions)
        for sample in level_probs:
            state = sample[:state_dim]  # Extract state portion
            action_probs = sample[state_dim:]  # Extract action probabilities portion
            state_tuple = tuple(state)
            prob_state_to_probs[state_tuple] = action_probs
    
    # STEP 2: Create flat list of representatives
    all_reps_with_level = []
    for level_idx, level_reps in enumerate(level_set_representatives):
        for rep_state in level_reps:
            all_reps_with_level.append((tuple(rep_state), level_idx))
    
    # Convert representatives to array once
    reps_array = np.array([rep_tuple for rep_tuple, _ in all_reps_with_level])
    
    tolerance_matches = 0
    closest_matches = 0
    
    # STEP 3: Process boundary states in batches to avoid memory overflow
    for batch_start in range(0, len(boundary_states), batch_size):
        batch_end = min(batch_start + batch_size, len(boundary_states))
        batch_boundary_states = boundary_states[batch_start:batch_end]
        batch_boundary_array = np.array(batch_boundary_states)
        
        # Compute distances for this batch
        batch_distances = np.linalg.norm(batch_boundary_array[:, np.newaxis] - reps_array[np.newaxis, :], axis=2)
        
        # Process each state in the batch
        for i, boundary_state in enumerate(batch_boundary_states):
            boundary_distances = batch_distances[i]  # Shape: (num_reps,)
            
            # Find tolerance matches first
            tolerance_mask = boundary_distances < tolerance
            if np.any(tolerance_mask):
                rep_idx = np.where(tolerance_mask)[0][0]  # Use first tolerance match
                tolerance_matches += 1
            else:
                rep_idx = np.argmin(boundary_distances)  # Use closest match
                closest_matches += 1
            
            rep_match = all_reps_with_level[rep_idx][0]
            
            # Get action probabilities for the matched representative
            if rep_match in prob_state_to_probs:
                action_probs = prob_state_to_probs[rep_match]
            else:
                # Use first available action probabilities as fallback
                for level_probs in probabilities_list:
                    if level_probs is not None and len(level_probs) > 0:
                        action_probs = level_probs[0][state_dim:]  # Use first sample
                        break
            
            # Create sample in the same format as regular samples: [state, action_probs]
            boundary_sample = np.concatenate([boundary_state, action_probs])
            boundary_states_with_probs.append(boundary_sample)
        
        # Clean up batch memory
        del batch_boundary_array
        del batch_distances
        import gc
        gc.collect()
    
    return boundary_states_with_probs

def identify_boundary_states(level_set_representatives, config, threshold_distance=None, batch_size=64):
    """
    Identify boundary states by forward propagating states from all level sets except the last
    and checking if propagated states are safe (within threshold distance to representatives).
    Uses batched processing to avoid memory explosion for large level sets.
    
    Args:
        level_set_representatives: List of level sets, each containing list of representative states
        config: Model configuration containing actions, dt, velocity, and thresholds
        threshold_distance: Distance threshold to determine if propagated state is safe (if None, calculated from config)
        batch_size: Number of states to process at once to control memory usage (default: 200 for aggressive processing)
        
    Returns:
        dict: Dictionary containing boundary states and visualization info
    """
    # Calculate threshold distance from config if not provided
    if threshold_distance is None:
        if config['state_dim'] == 4 and config['adaptive_uniformity']:
            # For 4D adaptive uniformity, use AdaptiveGrid with dynamic thresholds per level
            adaptive_grid = AdaptiveGrid(
                base_thresholds=config["thresholds"],
                dt=config["dt"],
                max_velocity=config["vrange"][1],
                max_acceleration=max(abs(config["arange"][0]), abs(config["arange"][1])),
                max_steering_deg=max(abs(config["steering_angle_range"][0]), abs(config["steering_angle_range"][1])),
                wheelbase=0.324  # Vehicle wheelbase from dynamics
            )
        else:
            # Use fixed thresholds for 3D or non-adaptive systems
            device = torch.device('cuda')
            T = (torch.sqrt(torch.tensor(config["thresholds"][0]**2) + torch.tensor(config["thresholds"][1]**2))/1.0).float().to(device)
            threshold_distance = T.item()
    
    if config['state_dim'] == 3:
        actions = config['actions'][:, 0]  # Extract steering angles
        vrange = config['vrange'][0]  # Velocity
    elif config['state_dim'] == 4:
        actions = config['actions']
        vrange = config['vrange']
    else:
        raise ValueError(f"Unsupported state dimension: {config['state_dim']}")
    dt = config['dt']
    
    # Convert to PyTorch tensors and use CUDA for performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert torch.cuda.is_available(), "CUDA must be available for efficient boundary state computation"
    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
    
    boundary_states_all = []
    boundary_info = []
    
    # Process all level sets except the last one (don't propagate from final level set)
    for level_idx in range(len(level_set_representatives) - 1):
        # Calculate dynamic threshold for this specific level
        if config['state_dim'] == 4 and config['adaptive_uniformity']:
            # Use level-specific thresholds for adaptive grid
            level_thresholds = adaptive_grid.get_thresholds(level=level_idx, zero_index=True)
            current_threshold = float(np.sqrt(level_thresholds[0]**2 + level_thresholds[1]**2))
        else:
            # Use fixed threshold for non-adaptive systems
            current_threshold = threshold_distance
        current_level_reps = level_set_representatives[level_idx]
        next_level_reps = level_set_representatives[level_idx + 1]
        
        if len(current_level_reps) == 0 or len(next_level_reps) == 0:
            continue
        
        # Convert current level representatives to tensor
        current_states = torch.tensor(current_level_reps, dtype=torch.float32, device=device)
        
        # Convert next level representatives to tensor for safety checking
        next_reps_tensor = torch.tensor(next_level_reps, dtype=torch.float32, device=device)
        next_reps_xy = next_reps_tensor[:, :2]  # Only x, y for distance calculation
        
        num_states = current_states.shape[0]
        num_actions = actions_tensor.shape[0]
        
        # Process states in batches to avoid memory explosion
        boundary_mask_full = torch.zeros(num_states, dtype=torch.bool, device=device)
        
        for batch_start in range(0, num_states, batch_size):
            batch_end = min(batch_start + batch_size, num_states)
            batch_states = current_states[batch_start:batch_end]
            batch_size_actual = batch_states.shape[0]
            # Forward propagate batch states with all actions
            # Check if the system is 3D or 4D based on state dimension
            if config['state_dim'] == 3:
                next_states_batch = cuda_dynamics_KS_3d_steering_angle_vectorized(
                    batch_states, actions_tensor, dt, vrange
                )  # Returns (batch_size_actual * num_actions, 3)
            elif config['state_dim'] == 4:
                next_states_batch = cuda_dynamics_KS_4d_steering_angle_v_vectorized(
                    batch_states, actions_tensor, dt, vrange
                )  # Returns (batch_size_actual * num_actions, 4)
            else:
                raise ValueError(f"Unsupported state dimension: {config['state_dim']}")
            
            # Extract x, y coordinates of propagated states
            next_states_xy_batch = next_states_batch[:, :2]  # (batch_size_actual * num_actions, 2)
            
            # Compute minimum distance to safe representatives
            if next_reps_xy.shape[0] == 0:
                print(f"      Warning: No next level representatives for level {level_idx + 1}")
                continue
                
            # Distance matrix: (batch_size_actual * num_actions, num_next_reps)
            distance_matrix_batch = torch.cdist(next_states_xy_batch, next_reps_xy, p=2)
            min_distances_batch = distance_matrix_batch.min(dim=1)[0]  # (batch_size_actual * num_actions,)
            
            # Identify unsafe propagations (outside safe region) using dynamic threshold
            unsafe_mask_batch = min_distances_batch > current_threshold  # (batch_size_actual * num_actions,)
            
            # Reshape to (batch_size_actual, num_actions)
            unsafe_mask_reshaped_batch = unsafe_mask_batch.reshape(batch_size_actual, num_actions)
            
            # Find boundary states in this batch: states that have at least one unsafe action
            boundary_mask_batch = unsafe_mask_reshaped_batch.any(dim=1)  # (batch_size_actual,)
            
            # Update the full boundary mask
            boundary_mask_full[batch_start:batch_end] = boundary_mask_batch
        
        # Extract boundary states for this level
        boundary_states_level = current_states[boundary_mask_full].cpu().numpy()
        boundary_states_all.extend(boundary_states_level.tolist())
        
        # Store information for this level
        level_info = {
            'level_idx': level_idx,
            'total_states': len(current_level_reps),
            'boundary_states': len(boundary_states_level),
            'boundary_percentage': len(boundary_states_level) / len(current_level_reps) * 100
        }
        boundary_info.append(level_info)
        
    # Clean up tensors to free GPU/CPU memory
    del actions_tensor
    if 'current_states' in locals():
        del current_states
    if 'next_reps_tensor' in locals():
        del next_reps_tensor
    import gc
    gc.collect()
    
    return {
        'boundary_states': boundary_states_all,
        'boundary_info': boundary_info,
        'threshold_distance': threshold_distance
    }


def create_open_space_4d_dataset(
        env_name="open_space_4d",
        base_dir="dataset_supervised",
        grid_size=401,
        resolution=0.05,
        num_trajectories=10000,
        num_initial_velocities=5,
    ):
    """
    Create supervised dataset for 4D system in open space (no obstacles) with multiple initial velocities.
    
    Args:
        env_name (str): Name of the environment
        base_dir (str): Base directory for datasets
        grid_size (int): Grid size for SDF (401 for 4D system)
        resolution (float): Grid resolution in meters
        num_trajectories (int): Number of trajectories to sample per initial velocity
        num_initial_velocities (int): Number of different initial velocities (default: 5)
    """
    print("="*80)
    print("CREATING OPEN SPACE 4D DATASET")
    print(f"Environment: {env_name}")
    print(f"Grid size: {grid_size} (for 4D system)")
    print(f"Initial velocities: {num_initial_velocities} (0 to {num_initial_velocities-1})")
    print(f"Trajectories per velocity: {num_trajectories}")
    print("="*80)
    
    env_start_time = time.time()
    env_folder = create_environment_folder(env_name, base_dir)
    
    # Create empty obstacles (open space)
    obstacles = []
    
    # Create SDF for open space using existing function
    sdf_start = time.time()
    costmap, sdf = create_sdf_from_obstacles(obstacles, grid_size, resolution)
    
    # Save SDF and costmap
    np.save(os.path.join(env_folder, "costmap.npy"), costmap)
    np.save(os.path.join(env_folder, "sdf.npy"), sdf)
    sdf_time = time.time() - sdf_start
    print(f"  ✓ Open space SDF created and saved ({sdf_time:.2f}s)")
    
    # Visualize SDF
    vis_sdf_start = time.time()
    scene_path = os.path.join(env_folder, "scene.png")
    visualize_sdf(sdf, resolution, scene_path)
    vis_sdf_time = time.time() - vis_sdf_start
    print(f"  ✓ SDF visualized ({vis_sdf_time:.2f}s)")
    
    # Generate action probabilities for each initial velocity SEQUENTIALLY
    
    for velocity in range(num_initial_velocities):
        print(f"\n Processing initial velocity {velocity} (state {velocity+1}/{num_initial_velocities})")
        
        # Create single initial state for this velocity
        initial_state = np.array([0.0, 0.0, 0.0, float(velocity)])  # (x=0, y=0, theta=0, v=velocity)
        
        # Compute action probabilities for this initial velocity (for open space, no SDF needed)
        action_prob_data, pruned_graphs = compute_action_probabilities_for_environment(
            config, obstacles, None, resolution, initial_state=initial_state
        )
        
        if action_prob_data is None:
            print(f" Failed to generate action probabilities for velocity {velocity}")
            continue
        
        # Save action probabilities with velocity-specific name IMMEDIATELY
        print(f"  Saving files for velocity {velocity}...")
        action_prob_filename = f"action_probs_v{velocity}.pkl"
        with open(os.path.join(env_folder, action_prob_filename), "wb") as f:
            pickle.dump(action_prob_data, f)
        
        # Save configuration as readable text file (velocity-specific)
        config_filename = f"config_info_v{velocity}.txt"
        save_config_to_txt(action_prob_data['config'], env_folder, config_filename)
        
        # Generate trajectories for this velocity
        trajectories = parallelized_trajectory_sampling_cuda(action_prob_data, num_trajectories)
        if len(trajectories) == 0:
            print(f"   No trajectories generated for velocity {velocity}")
            continue
        
        # Save individual trajectory visualization for this velocity
        vis_filepath = os.path.join(env_folder, f"ground_truth_trajectories_v{velocity}.png")
        visualize_trajectories_background(
            trajectories=trajectories,
            costmap=costmap,
            resolution=resolution,
            show_vis=False,
            save_vis=True,
            vis_filepath=vis_filepath,
            alpha=0.2,
            marker_size=0.7,
            title=f"Ground Truth Trajectories - Initial Velocity {velocity} m/s"
        )
        
        # Save level set visualization for this velocity
        level_set_vis_path = os.path.join(env_folder, f"level_set_representatives_v{velocity}.png")
        save_level_set_visualization(
            action_prob_data['level_set_representatives'], 
            level_set_vis_path,
            config,
            action_prob_data['boundary_states']['boundary_states'],
            velocity=velocity
        )
        
        # Always save boundary states file for consistency (even if empty, use raw action probabilities)
        boundary_filename = f"boundary_states_with_probs_v{velocity}.pkl"
        if len(action_prob_data['boundary_states_with_probs']) > 0:
            # Use actual boundary states with action probabilities
            boundary_data_to_save = action_prob_data['boundary_states_with_probs']
        else:
            # Use raw action probabilities from level sets as fallback for consistent file structure
            boundary_data_to_save = []
            for level_set in action_prob_data['level_sets']:
                if level_set is not None and len(level_set) > 0:
                    boundary_data_to_save.extend(level_set.tolist())
        
        with open(os.path.join(env_folder, boundary_filename), "wb") as f:
            pickle.dump(boundary_data_to_save, f)
        
        print(f"   Files saved for velocity {velocity}")
        
        # CRITICAL: Clear large data structures to free memory before next iteration
        del action_prob_data
        del pruned_graphs
        del trajectories
        import gc
        gc.collect()
    
    env_total_time = time.time() - env_start_time
    print(f"\n🏁 {env_name} completed in {env_total_time:.2f}s")
    print("="*80)

def create_supervised_datasets(
        dataset_type,
        base_dir="dataset_supervised",
        # Circle obstacle parameters
        num_envs=10,
        obstacle_radius=0.7,
        # Real-world LiDAR parameters  
        input_dataset_dir=None,
        output_dataset_dir=None,
        # Common parameters
        grid_size=121,
        resolution=0.05,
        num_trajectories=10000,
        skip_existing=True,
    ):
    """
    Unified function to create supervised learning datasets for both circle obstacles and real-world LiDAR data.
    
    Args:
        dataset_type (str): Either "circle" for synthetic circle obstacles or "lidar" for real-world LiDAR data
        base_dir (str): Base directory for datasets (used for circle type)
        num_envs (int): Number of environments to create (circle type only)
        obstacle_radius (float): Radius of circle obstacles (circle type only)
        input_dataset_dir (str): Input directory for LiDAR data (lidar type only)
        output_dataset_dir (str): Output directory for LiDAR data (lidar type only)
        grid_size (int): Grid size for SDF
        resolution (float): Grid resolution in meters
        num_trajectories (int): Number of trajectories to sample per environment
        skip_existing (bool): Skip environments that already exist
    """
    
    if dataset_type == "circle":
        print("="*80)
        print("CREATING CIRCLE OBSTACLE SUPERVISED DATASETS")
        print(f"Number of environments: {num_envs}")
        print(f"Obstacle radius: {obstacle_radius}")
        print(f"Output directory: {base_dir}")
        print("="*80)
        
        # Create circle environments with linearly spaced obstacles
        # precompute linearly spaced centers
        grid_n = int(np.ceil(np.sqrt(num_envs)))
        xs = np.linspace(1.0, 3.0, grid_n)
        ys = np.linspace(-2.0, 2.0, grid_n)

        # build list of (x,y) pairs, then take exactly num_envs of them
        centers = [(x, y) for x in xs for y in ys][:num_envs]

        for idx, (x_center, y_center) in enumerate(centers, start=1):
            env_start_time = time.time()
            env_name = f"env_{idx:02d}"
            env_folder = create_environment_folder(env_name, base_dir)
            
            # Check file completeness FIRST to avoid unnecessary operations
            file_status = check_environment_completeness(env_folder)
            missing_files = [key for key, exists in file_status.items() if not exists]
            
            if len(missing_files) == 0:
                print(f"   All files exist for {env_name}, skipping")
                env_total_time = time.time() - env_start_time
                print(f"🏁 {env_name} completed in {env_total_time:.2f}s")
                print("-" * 80)
                continue
            
            # Only proceed with operations if some files are missing
            print(f"🔴 {env_name}: obstacle at (x={x_center:.2f}, y={y_center:.2f}, r={obstacle_radius})")
            print(f"  📋 Missing files: {', '.join(missing_files)}")
            
            # --- 1) sample one circle obstacle ---
            obstacles = [(x_center, y_center, obstacle_radius)]

            # --- 2) build costmap & SDF only if missing ---
            if not file_status['sdf'] or not file_status['costmap']:
                sdf_start = time.time()
                costmap, sdf = create_sdf_from_obstacles(obstacles, grid_size, resolution)
                if not file_status['costmap']:
                    np.save(os.path.join(env_folder, "costmap.npy"), costmap)
                if not file_status['sdf']:
                    np.save(os.path.join(env_folder, "sdf.npy"), sdf)
                sdf_time = time.time() - sdf_start
                print(f"  ✓ SDF created and saved ({sdf_time:.2f}s)")
            else:
                # Load existing files for processing
                costmap = np.load(os.path.join(env_folder, "costmap.npy"))
                sdf = np.load(os.path.join(env_folder, "sdf.npy"))

            # --- 3) visualize SDF only if missing ---
            if not file_status['scene']:
                vis_sdf_start = time.time()
                scene_path = os.path.join(env_folder, "scene.png")
                visualize_sdf(sdf, resolution, scene_path)
                vis_sdf_time = time.time() - vis_sdf_start
                print(f"  ✓ SDF visualized ({vis_sdf_time:.2f}s)")

            # Process the environment (common code)
            _process_single_environment(
                env_name, env_folder, obstacles, sdf, costmap, resolution, num_trajectories
            )
            
            env_total_time = time.time() - env_start_time
            print(f"🏁 {env_name} completed in {env_total_time:.2f}s")
            print("-" * 80)
            
    elif dataset_type == "lidar":
        start_time = time.time()
        
        # Get the directory where this script is located (map_conditioning)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use absolute paths to avoid path issues
        input_dataset_dir = os.path.join(script_dir, input_dataset_dir)
        output_dataset_dir = os.path.join(script_dir, output_dataset_dir)
        
        # Create output directory
        os.makedirs(output_dataset_dir, exist_ok=True)
        print("="*80)
        print("SUPERVISED REAL-WORLD LIDAR DATASET PROCESSING")
        print(f"Input:  {input_dataset_dir}")
        print(f"Output: {output_dataset_dir}")
        print("="*80)
        
        # Get all environment directories in the input dataset
        if not os.path.exists(input_dataset_dir):
            print(f" Input directory does not exist: {input_dataset_dir}")
            return
            
        env_dirs = sorted([d for d in os.listdir(input_dataset_dir) 
                          if os.path.isdir(os.path.join(input_dataset_dir, d)) and d.startswith("env_")])
        
        if not env_dirs:
            print(f" No environment directories found in {input_dataset_dir}")
            return
        
        print(f" Found {len(env_dirs)} environments in input dataset")
        
        # Process each environment
        valid_count = 0
        skipped_count = 0
        failed_count = 0
        
        for i, env_name in enumerate(env_dirs):
            env_start_time = time.time()
            print(f"\n Processing environment {i+1}/{len(env_dirs)}: {env_name}")
            
            # Create environment directory in output dataset
            output_env_dir = os.path.join(output_dataset_dir, env_name)
            os.makedirs(output_env_dir, exist_ok=True)
            
            # Check file completeness FIRST to avoid unnecessary operations
            file_status = check_environment_completeness(output_env_dir)
            missing_files = [key for key, exists in file_status.items() if not exists]
            
            if len(missing_files) == 0:
                print(f"   All files exist for {env_name}, skipping")
                skipped_count += 1
                env_total_time = time.time() - env_start_time
                print(f"🏁 {env_name} completed in {env_total_time:.2f}s")
                print("-" * 80)
                continue
            
            # Only proceed with file operations if some files are missing
            print(f"  📋 Status for {env_name}:")
            existing_files = [key for key, exists in file_status.items() if exists]
            print(f"     Existing: {', '.join(existing_files) if existing_files else 'none'}")
            print(f"     Missing: {', '.join(missing_files)}")
            
            # Load SDF and costmap from input dataset only if needed
            sdf_path = os.path.join(input_dataset_dir, env_name, "sdf.npy")
            costmap_path = os.path.join(input_dataset_dir, env_name, "costmap.npy")
            
            if not os.path.exists(sdf_path) or not os.path.exists(costmap_path):
                print(f" Missing SDF or costmap for {env_name}, skipping...")
                failed_count += 1
                continue
            
            sdf = np.load(sdf_path)
            costmap = np.load(costmap_path)
            
            # Copy SDF and costmap to output directory only if missing
            if not file_status['sdf']:
                np.save(os.path.join(output_env_dir, "sdf.npy"), sdf)
                print(f"  ✓ SDF copied")
            if not file_status['costmap']:
                np.save(os.path.join(output_env_dir, "costmap.npy"), costmap)
                print(f"  ✓ Costmap copied")
            
            # Generate and save SDF visualization only if missing
            if not file_status['scene']:
                scene_filepath = os.path.join(output_env_dir, "scene.png")
                visualize_sdf(sdf, resolution, scene_filepath)
                print(f"  ✓ Scene visualization saved")
            
            # For real-world data, we don't have explicit obstacles, so use an empty list
            obstacles = []
            
            # Process the environment (common code)
            success = _process_single_environment(
                env_name, output_env_dir, obstacles, sdf, costmap, resolution, num_trajectories
            )
            
            if success:
                valid_count += 1
            else:
                failed_count += 1
                continue
            
            env_total_time = time.time() - env_start_time
            print(f"🏁 {env_name} completed in {env_total_time:.2f}s")
            print("-" * 80)
        
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print(" PROCESSING SUMMARY")
        print(f"Total environments processed: {len(env_dirs)}")
        print(f" Valid environments: {valid_count}")
        print(f"⏭  Skipped (existing): {skipped_count}")
        print(f" Failed environments: {failed_count}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f" Valid environments saved to: {output_dataset_dir}")
        print("="*80)
        
    elif dataset_type == "open_space_vel2":
        print("="*80)
        print("CREATING OPEN SPACE VEL2 SUPERVISED DATASET")
        print(f"Output directory: {base_dir}")
        print(f"Grid size: {grid_size}")
        print(f"Resolution: {resolution}")
        print(f"Trajectories: {num_trajectories}")
        print("="*80)
        
        env_start_time = time.time()
        env_name = "env_001"
        env_folder = create_environment_folder(env_name, base_dir)
        
        # Check file completeness FIRST to avoid unnecessary operations
        file_status = check_environment_completeness(env_folder)
        missing_files = [key for key, exists in file_status.items() if not exists]
        
        if len(missing_files) == 0 and skip_existing:
            print(f"   All files exist for {env_name}, skipping")
            env_total_time = time.time() - env_start_time
            print(f"🏁 {env_name} completed in {env_total_time:.2f}s")
            print("="*80)
            return
        
        # Only proceed with operations if some files are missing
        print(f"🔴 {env_name}: open space environment (no obstacles)")
        if missing_files:
            print(f"  📋 Missing files: {', '.join(missing_files)}")
        
        # Create empty obstacles (open space)
        obstacles = []
        
        # Build costmap & SDF only if missing
        if not file_status['sdf'] or not file_status['costmap']:
            sdf_start = time.time()
            costmap, sdf = create_sdf_from_obstacles(obstacles, grid_size, resolution)
            if not file_status['costmap']:
                np.save(os.path.join(env_folder, "costmap.npy"), costmap)
            if not file_status['sdf']:
                np.save(os.path.join(env_folder, "sdf.npy"), sdf)
            sdf_time = time.time() - sdf_start
            print(f"  ✓ Open space SDF created and saved ({sdf_time:.2f}s)")
        else:
            # Load existing files for processing
            costmap = np.load(os.path.join(env_folder, "costmap.npy"))
            sdf = np.load(os.path.join(env_folder, "sdf.npy"))

        # Visualize SDF only if missing
        if not file_status['scene']:
            vis_sdf_start = time.time()
            scene_path = os.path.join(env_folder, "scene.png")
            visualize_sdf(sdf, resolution, scene_path)
            vis_sdf_time = time.time() - vis_sdf_start
            print(f"  ✓ SDF visualized ({vis_sdf_time:.2f}s)")

        # Process the environment (common code)
        success = _process_single_environment(
            env_name, env_folder, obstacles, sdf, costmap, resolution, num_trajectories
        )
        
        if success:
            env_total_time = time.time() - env_start_time
            print(f"🏁 {env_name} completed in {env_total_time:.2f}s")
            print("="*80)
        else:
            print(f" Failed to process {env_name}")
            print("="*80)
    
    elif dataset_type == "barn":
        print("="*80)
        print("CREATING BARN SUPERVISED DATASET")
        print(f"BARN dataset: {input_dataset_dir}")
        print(f"Output directory: {output_dataset_dir}")
        print(f"Grid size: {grid_size}×{grid_size}@{resolution}m (consistent)")
        print("="*80)
        
        # Get the parent directory for BARN dataset access
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_barn_dir = os.path.join(parent_dir, "BARN_dataset")
        
        actual_barn_dir = input_dataset_dir or default_barn_dir
        
        from barn_dataset_helpers import create_barn_supervised_dataset_clean
        create_barn_supervised_dataset_clean(
            barn_dataset_dir=actual_barn_dir,
            output_base_dir=output_dataset_dir or os.path.join(base_dir, "barn_dataset"),
            num_environments=300,      # number of BARN environments to process 
            num_positions_per_env=5,  # Number of positions per environment
            start_env_id=0,           # Start from environment 0
            resolution=resolution,    # Use consistent 0.05m resolution
            skip_existing=skip_existing,
            config=config  # Pass the main configuration directly
        )
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'circle', 'lidar', 'open_space_vel2', or 'barn'")

def _process_single_environment(env_name, env_folder, obstacles, sdf, costmap, resolution, num_trajectories):
    """
    Common processing logic for both circle and LiDAR environments.
    Supports partial re-processing - can generate missing files for existing environments.
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Check what files already exist (this check is now done at higher level, but kept for safety)
        file_status = check_environment_completeness(env_folder)
        
        # Load or generate action probabilities
        action_prob_data = None
        if file_status['action_probs']:
            # Load existing action probabilities
            print(f"  Loading existing action probabilities...")
            with open(os.path.join(env_folder, "action_probs.pkl"), 'rb') as f:
                action_prob_data = pickle.load(f)
            print(f"  ✓ Action probabilities loaded")
        else:
            # Generate action probabilities
            print(f"  Generating action probabilities...")
            action_prob_data, pruned_graphs = compute_action_probabilities_for_environment(
                config, obstacles, sdf, resolution
            )
            
            if action_prob_data is None:
                print(f" Failed to generate valid action probabilities for {env_name}")
                return False
                
            # Save action probability data
            save_start = time.time()    
            print(f"  Saving action probabilities...")
            with open(os.path.join(env_folder, "action_probs.pkl"), "wb") as f:
                pickle.dump(action_prob_data, f)
            print(f"  ✓ Action probabilities saved ({time.time() - save_start:.2f}s)")
            
            # Save configuration as readable text file
            save_config_to_txt(action_prob_data['config'], env_folder)
            
            # Save boundary states with probabilities separately for data augmentation
            if len(action_prob_data['boundary_states_with_probs']) > 0:
                with open(os.path.join(env_folder, "boundary_states_with_probs.pkl"), "wb") as f:
                    pickle.dump(action_prob_data['boundary_states_with_probs'], f)
                print(f"  ✓ Boundary states saved ({len(action_prob_data['boundary_states_with_probs'])} samples)")

        # Generate level set visualization if missing
        if not file_status['level_set_vis']:
            level_vis_start = time.time()
            print(f"  Generating level set visualization...")
            level_set_vis_path = os.path.join(env_folder, "level_set_representatives.png")
            save_level_set_visualization(
                action_prob_data['level_set_representatives'], 
                level_set_vis_path,
                config,
                action_prob_data['boundary_states']['boundary_states']
            )
            level_vis_time = time.time() - level_vis_start
            print(f"  ✓ Level set visualization saved ({level_vis_time:.2f}s)")
        
        # Print boundary state statistics if action probabilities were just generated
        if not file_status['action_probs']:
            boundary_info = action_prob_data['boundary_states']['boundary_info']
            total_boundary = len(action_prob_data['boundary_states']['boundary_states'])
            print(f"  Boundary States Summary: {total_boundary} total")
            for info in boundary_info:
                print(f"    Level {info['level_idx']}: {info['boundary_states']}/{info['total_states']} ({info['boundary_percentage']:.1f}%)")

        # Generate trajectories and visualization if missing
        if not file_status['trajectory_vis']:
            print(f"  Generating trajectories using action probabilities...")
            success = generate_trajectories_from_action_probs(
                env_name, env_folder, action_prob_data, costmap, resolution, num_trajectories
            )
            
            if not success:
                print(f"   Failed to generate trajectories for {env_name}")
                return False
            else:
                print(f"  ✓ Generated {num_trajectories} ground truth trajectories")
        
        return True
        
    except Exception as e:
        print(f"   Error processing {env_name}: {str(e)}")
        return False

def check_environment_completeness(env_folder):
    """
    Check what files exist in an environment folder.
    
    Returns:
        dict: Status of each required file
    """
    required_files = {
        'action_probs': 'action_probs.pkl',
        'config_info': 'config_info.txt',
        'trajectory_vis': 'ground_truth_trajectories.png',
        'level_set_vis': 'level_set_representatives.png',
        'scene': 'scene.png',
        'sdf': 'sdf.npy',
        'costmap': 'costmap.npy'
    }
    
    status = {}
    for key, filename in required_files.items():
        filepath = os.path.join(env_folder, filename)
        status[key] = os.path.exists(filepath)
    
    return status

def generate_trajectories_from_action_probs(
        env_name, env_folder, action_prob_data, costmap, resolution, num_trajectories
    ):
    """
    Generate trajectories using action probabilities and create visualization.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Sample trajectories using CUDA-accelerated parallel sampling
        sample_start = time.time()
        print(f"    Sampling {num_trajectories} trajectories using CUDA-accelerated parallel sampling...")
        trajectories = parallelized_trajectory_sampling_cuda(action_prob_data, num_trajectories)
        
        if len(trajectories) == 0:
            print(f"     No trajectories sampled")
            return False
            
        sample_time = time.time() - sample_start
        print(f"    Trajectories sampled efficiently ({sample_time:.2f}s)")
        
        # Visualize trajectories (no perturbation needed as trajectories are naturally sampled)
        vis_start = time.time()
        vis_filepath = os.path.join(env_folder, "ground_truth_trajectories.png")
        print(f"    Visualizing trajectories...")
        visualize_trajectories_background(
            trajectories=trajectories,
            costmap=costmap,
            resolution=resolution,
            show_vis=False,
            save_vis=True,
            vis_filepath=vis_filepath,
            alpha=0.2,
            marker_size=1,
        )
        vis_time = time.time() - vis_start
        print(f"    ✓ Trajectories visualized ({vis_time:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"     Error generating trajectories: {str(e)}")
        return False

# -------------------------------
# Main Module: Create Multiple Environment Datasets for Supervised Learning
# -------------------------------
def main():
    torch.manual_seed(2025)
    np.random.seed(2025)
    random.seed(2025)
    
    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "dataset_supervised")
    os.makedirs(base_dir, exist_ok=True)
    
    print("="*80)
    print("SUPERVISED DATASET GENERATION")
    print("Generating action probability datasets for supervised learning")
    print("Including ground truth trajectory generation and visualization")
    print(f"Working directory: {base_dir}")
    print(f"Detailed Output: {'Visible' if not SUPPRESS_DETAILED_OUTPUT else 'Suppressed'}")
    print("="*80)
    
    # Create circle obstacle datasets
    # print("\n--- Creating Circle Obstacle Environments ---")
    # create_supervised_datasets(
    #     dataset_type="circle",
    #     base_dir=os.path.join(base_dir, "circle_obstacles_ten"),
    #     num_envs=10,
    #     obstacle_radius=0.7,
    #     grid_size=121,
    #     resolution=0.05,
    #     num_trajectories=10000,
    #     skip_existing=True,
    # )
    
    # Process real-world LiDAR dataset for supervised learning
    # print("\n--- Processing Real-World LiDAR Dataset for Supervised Learning ---")
    # create_supervised_datasets(
    #     dataset_type="lidar",
    #     # input_dataset_dir="dataset/realworld_lidar_scan_dataset_raw",  # Relative to script directory
    #     input_dataset_dir="dataset_supervised/shepherd_dataset_supervised_cleaned_only_costmap_sdf",  # Relative to script directory
    #     output_dataset_dir="dataset_supervised/shepherd_dataset_supervised_cleaned_resolution_0.05_v1_sdfNoInflation",  # Relative to script directory
    #     grid_size=121,
    #     resolution=0.05,
    #     num_trajectories=1000,
    #     skip_existing=True,
    # )
    
    # print("\n--- Generating trajectories for open environment for 4D model---")
    # create_open_space_4d_dataset(
    #     env_name="open_space_4d",
    #     base_dir=base_dir,
    #     grid_size=401,
    #     resolution=0.05,
    #     num_trajectories=10000,
    #     num_initial_velocities=5,
    # )

    # Process open space vel2 dataset for supervised learning
    # print("\n--- Processing Open Space Vel2 Dataset for Supervised Learning ---")
    # create_supervised_datasets(
    #     dataset_type="open_space_vel2",
    #     base_dir=os.path.join(base_dir, "open_space_vel2"),
    #     grid_size=201,
    #     resolution=0.10,
    #     num_trajectories=10000,
    #     skip_existing=True,
    # )

    # Process BARN dataset for supervised learning
    print("\n--- Processing BARN Dataset for Supervised Learning ---")
    create_supervised_datasets(
        dataset_type="barn",
        input_dataset_dir=None,  # Use default absolute path construction
        output_dataset_dir=os.path.join(base_dir, "barn_dataset_vmax2.5"),
        grid_size=121,
        resolution=0.05,
        num_trajectories=10000,
        skip_existing=True,
    )
if __name__ == "__main__":
    main() 