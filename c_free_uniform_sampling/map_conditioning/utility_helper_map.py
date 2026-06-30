import pickle
import torch
import random
import contextlib
from classes.grid import *
import torch.nn.functional as F
import time
import numpy as np
import os
import math
from pprint import pprint
import torch.distributions as dist
from scipy.ndimage import distance_transform_edt # For computing the SDF
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from flow_Cuniform.c_uniform_sampling import(
    parallelized_collision_checker_nodes,
)
from flow_Cuniform.dynamics_helpers import (
    vectorized_dynamics_KS_3d_steering_angle,
)

torch.manual_seed(2025)
np.random.seed(2025)
random.seed(2025)

# =============================================================================
# DETERMINISTIC SAMPLING FUNCTIONS
# =============================================================================

def set_trajectory_sampling_seeds(seed):
    """
    Set all random seeds for deterministic trajectory sampling.
    Call this before any trajectory sampling operations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Set environment variables for CUDA determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def visualize_states(state_list, title, max_samples=20000):
    """
    Visualize state sets (e.g., representatives or trajectory states) in 2D by plotting their x and y coordinates.
    For each time step, if there are more than max_samples states, max_samples are sampled at random.
    
    Args:
        state_list (list): A list (per time step) of state vectors (either as numpy arrays or torch tensors).
        title (str): Title for the plot.
        max_samples (int): Maximum number of points to plot per time step.
    """
    plt.figure(figsize=(10, 6))
    for idx, states in enumerate(state_list):
        # Convert to numpy array if needed
        if isinstance(states[0], torch.Tensor):
            states_np = torch.stack(states).cpu().numpy()
        else:
            states_np = np.array(states)
        # Sample if necessary
        if states_np.shape[0] > max_samples:
            sample_indices = np.random.choice(states_np.shape[0], max_samples, replace=False)
            states_np = states_np[sample_indices]
        plt.scatter(states_np[:, 0], states_np[:, 1], s=10, alpha=0.6, label=f"Time Step {idx}")
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis('equal')
    plt.legend()
    plt.show()

def visualize_sdf(sdf, resolution, filepath=None):
    """
    Visualize the Signed Distance Function (SDF) in the robot frame.
    """
    assert resolution > 0, "Resolution must be a positive value."
    assert sdf.ndim == 2 and sdf.shape[0] == sdf.shape[1], \
        "SDF must be computed on a 2D square grid."

    grid_size = sdf.shape[0]
    center_index = (grid_size - 1) / 2.0
    
    # Create x and y coordinate arrays in the robot's frame (in meters)
    x_coords = (np.arange(grid_size) - center_index) * resolution
    y_coords = (center_index - np.arange(grid_size)) * resolution
    
    # Use seismic colormap with proper normalization to show white line at 0 boundary
    vmax = np.max(np.abs(sdf))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    plt.figure(figsize=(6, 6))
    plt.imshow(sdf, origin='lower', extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]], cmap='seismic', norm=norm)
    plt.colorbar(label='Signed Distance (m)')
    plt.title("Signed Distance Function (SDF) in Robot Frame")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.gca().invert_yaxis()
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SDF visualization saved to: {filepath}")
    else:
        plt.show()

def save_level_set_visualization(level_set_representatives, filepath, config, boundary_states=None, max_samples=5000, velocity=None):
    """
    Visualize the level set representatives in 2D (x, y) and save the plot.
    Adds small noise proportional to grid thresholds for better visualization of overlapping states.
    Uses adaptive thresholds if system supports adaptive uniformity.
    Optionally overlays boundary states as red cross markers.
    
    Args:
        level_set_representatives: List of level sets, each containing list of representative states
        filepath: Path to save visualization
        config: Model configuration containing thresholds
        boundary_states: Optional list of boundary states to overlay as red crosses
        max_samples: Maximum number of samples to plot per level set
        velocity: Optional initial velocity for title customization
    """
    plt.figure(figsize=(10, 8))
    # Use viridis colormap for level set progression (dark to bright)
    colors_list = plt.cm.viridis(np.linspace(0, 1, len(level_set_representatives)))
    
    # Initialize adaptive grid if needed
    if config['state_dim'] == 4 and config['adaptive_uniformity']:
        adaptive_grid = AdaptiveGrid(
            base_thresholds=config["thresholds"],
            dt=config["dt"],
            max_velocity=config["vrange"][1],
            max_acceleration=max(abs(config["arange"][0]), abs(config["arange"][1])),
            max_steering_deg=max(abs(config["steering_angle_range"][0]), abs(config["steering_angle_range"][1])),
            wheelbase=0.324  # Vehicle wheelbase from dynamics
        )
    else:
        adaptive_grid = None
    
    # Plot level set representatives
    for level_idx, level_reps in enumerate(level_set_representatives):
        if len(level_reps) > 0:
            # Convert to numpy array
            level_array = np.array(level_reps)
            
            # Subsample if too many points
            if level_array.shape[0] > max_samples:
                sample_indices = np.random.choice(level_array.shape[0], max_samples, replace=False)
                level_array = level_array[sample_indices]
            
            # Calculate level-specific noise 
            if adaptive_grid is not None:
                level_thresholds = adaptive_grid.get_thresholds(level=level_idx, zero_index=True)
                level_noise_x = level_thresholds[0] * 0.5
                level_noise_y = level_thresholds[1] * 0.5
            else:
                level_noise_x = config['thresholds'][0] * 0.5
                level_noise_y = config['thresholds'][1] * 0.5
                
            # Add noise for visualization (only to x, y coordinates)
            noise_x = np.random.uniform(-level_noise_x, level_noise_x, level_array.shape[0])
            noise_y = np.random.uniform(-level_noise_y, level_noise_y, level_array.shape[0])
            
            # Apply noise to x, y coordinates
            level_array_vis = level_array.copy()
            level_array_vis[:, 0] += noise_x  # x coordinate
            level_array_vis[:, 1] += noise_y  # y coordinate
            
            plt.scatter(level_array_vis[:, 0], level_array_vis[:, 1], 
                       c=[colors_list[level_idx]], alpha=0.6, s=5, 
                       label=f'Level {level_idx} ({len(level_reps)} states)')
    
    # Optionally plot boundary states as small red cross markers on top
    if boundary_states is not None and len(boundary_states) > 0:
        boundary_array = np.array(boundary_states)
        
        # Use base threshold noise for boundary states
        if adaptive_grid is not None:
            boundary_noise_x = config['thresholds'][0] * 0.5  # Use base threshold for boundary states
            boundary_noise_y = config['thresholds'][1] * 0.5
        else:
            boundary_noise_x = config['thresholds'][0] * 0.5
            boundary_noise_y = config['thresholds'][1] * 0.5
        
        noise_x_boundary = np.random.uniform(-boundary_noise_x, boundary_noise_x, boundary_array.shape[0])
        noise_y_boundary = np.random.uniform(-boundary_noise_y, boundary_noise_y, boundary_array.shape[0])
        
        # Apply noise to boundary states
        boundary_array_vis = boundary_array.copy()
        boundary_array_vis[:, 0] += noise_x_boundary  # x coordinate
        boundary_array_vis[:, 1] += noise_y_boundary  # y coordinate
        
        plt.scatter(boundary_array_vis[:, 0], boundary_array_vis[:, 1], 
                   c='red', marker='x', s=5, alpha=0.6, 
                   label=f'Boundary States ({len(boundary_states)})')
        
        if velocity is not None:
            title = f'Level Set Representatives with Boundary States - Initial Velocity {velocity} m/s (with visualization noise)'
        else:
            title = 'Level Set Representatives with Boundary States (with visualization noise)'
    else:
        if velocity is not None:
            title = f'Level Set Representatives - Initial Velocity {velocity} m/s (with visualization noise)'
        else:
            title = 'Level Set Representatives (with visualization noise)'
    
    plt.title(title)
    plt.xlabel('X coordinate (m)')
    plt.ylabel('Y coordinate (m)')
    plt.axis('equal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.close()

##########################################
# Data Preparation
##########################################
def farthest_point_sampling_torch(X_raw: torch.Tensor, k: int, seed_raw: torch.Tensor = None):
    """
    X_raw:    (N,3) float tensor on GPU.
    k:        number of points to keep.
    seed_raw: (m,3) tensor of seeds.
    Returns:  (k,3) tensor of selected points.
    """
    device = X_raw.device
    N, D = X_raw.shape
    if k >= N:
        return X_raw.clone()

    # 1) normalize to [0,1]
    mins = X_raw.min(dim=0).values
    maxs = X_raw.max(dim=0).values
    scale = (maxs - mins).clamp(min=1e-6)
    X = (X_raw - mins) / scale

    # 2) seeds → indices
    S = []
    if seed_raw is not None:
        seed_norm = (seed_raw - mins) / scale
        # compute all pairwise squared dists: (m, N)
        diffs = seed_norm.unsqueeze(1) - X.unsqueeze(0) # shape (m, N, 3)
        d2    = (diffs**2).sum(dim=2)                  # shape (m, N)
        idxs  = d2.argmin(dim=1).cpu().tolist()        # to Python list
        for i in idxs:
            if i not in S:
                S.append(i)
    if not S:
        S = [0]

    # 3) initialize min‑distances (squared)
    D2 = torch.full((N,), float('inf'), device=device)
    for i in S:
        d2 = ((X - X[i])**2).sum(dim=1)
        D2 = torch.minimum(D2, d2)

    # 4) greedy select
    while len(S) < k:
        next_i = int(D2.argmax().item())
        S.append(next_i)
        d2 = ((X - X[next_i])**2).sum(dim=1)
        D2 = torch.minimum(D2, d2)

    return X_raw[torch.tensor(S, device=device)]

def prepare_trajectory_data(
    data,
    config,
    visualization,
    num_samples_per_level: int = 10000,
    sdf: np.ndarray = None,
    obstacles: list = None,
    skip = False,
    openspace = False
):
    """ 
    Enhanced data preparation: for each level set, generate exactly num_samples_per_level states by
    forward-propagation, collision checking, and include extreme-turn trajectories.

    Args:
        data (dict): Must contain 'pruned_level_set_representatives_across_LS': list of sets of states.
        config (dict): Must contain:
            'actions' (np.ndarray), 'dt' (float), 'vrange' (tuple), 'vectorized_dynamics' (callable).
        num_samples_per_level (int): Desired number of samples per level.
        sdf (np.ndarray): Signed distance function for collision checking (optional).
        obstacles (list): List of obstacle definitions for fallback collision checking.
        skip (bool): Whether to skip the oversampled calculation.
        openspace (bool): Whether to use the openspace baseline.

    Returns:
        oversampled (List[np.ndarray]): Each entry has shape (num_samples_per_level, state_dim).
        representative_list (List[List[torch.Tensor]]): Original reps per level as torch tensors.
    """
    start_time = time.time()

    # Extract raw representatives for each level
    raw_reps = data["pruned_level_set_representatives_across_LS"]
    num_levels = len(raw_reps)

    # Convert raw representatives to torch tensors for downstream use
    representative_list = []
    for level in range(num_levels):
        level_reps = []
        for state in raw_reps[level]:
            tensor_state = torch.tensor(state, dtype=torch.float32)
            level_reps.append(tensor_state)
        representative_list.append(level_reps)

    if skip:
        oversampled = None
        return oversampled, representative_list
    elif openspace:
        print("Preprocessing data with openspace baseline...")
        training_samples = representative_list[:-1].copy() #NOTE: use the representative list as oversampled for baseline
        return training_samples, representative_list
    else:
        print("Preprocessing data with high fidelity samples assuming disjoint level sets...")

    # Dynamics(HARDCODED) and parameters
    vectorized_dynamics = vectorized_dynamics_KS_3d_steering_angle
    dt = config['dt']
    vrange = config['vrange']
    actions = config['actions']

    # Setup disjoint‐level bookkeeping
    uniformity_grid     = Grid(thresholds=np.array(config["thresholds"]))
    all_visited_indices = set()

    # Storage for oversampled states
    oversampled = []

    # === Level 0: duplicate the single representative evenly ===
    reps_level_0 = np.array(list(raw_reps[0]), dtype=np.float32)
    # initial level set must have exactly one representative state
    assert reps_level_0.shape[0] == 1, (
        f"Expected exactly one initial representative, got {reps_level_0.shape[0]}"
    )
    # duplicate that single state num_samples_per_level times
    samples_level_0 = np.repeat(
        reps_level_0,
        repeats=num_samples_per_level,
        axis=0
    )
    # mark level‑0 cell as visited
    idx0 = uniformity_grid.get_index(samples_level_0[0])
    all_visited_indices.add(tuple(idx0))
    oversampled.append(samples_level_0)

    # === Subsequent levels: propagate from previous samples ===
    for level in range(1, num_levels-1):
        level_start_time = time.time()

        # Previous-level states: shape (num_samples_per_level, state_dim)
        prev_states = oversampled[level - 1]

        # Forward propagation: apply dynamics in one batch call
        all_next_states = vectorized_dynamics(
            prev_states,
            actions,
            dt,
            vrange 
        )

        # Collision checking: only (x,y) matters
        positions_xy = all_next_states[:, :2]
        collision_mask = parallelized_collision_checker_nodes(
            positions_xy,
            obstacles=obstacles,
            sdf=sdf,
            resolution=0.05  # NOTE: using default resolution and sdf_inflation=0.15
        )
        free_states = all_next_states[~collision_mask]

        # Filter out any state whose uniform‐grid cell was already seen
        grid_idxs   = uniformity_grid.get_index_vectorized(free_states)
        mask_new    = np.array([tuple(g) not in all_visited_indices for g in grid_idxs])
        pruned_count = free_states.shape[0] - mask_new.sum()
        print(f"Level {level}: pruned {pruned_count} overlapping points in the uniformity grid")
        free_states = free_states[mask_new]
        if free_states.size == 0:
            raise RuntimeError(f"No collision-free states generated at level {level}, double check pruned_level_set_representatives_across_LS")

        # Compute action-based seeds: lowest, middle, highest steering ---
        low_action = actions[0]
        high_action = actions[-1]
        mid_idx = len(actions) // 2
        mid_action = actions[mid_idx]
        state_dim = prev_states.shape[1]

        # propagate origin with each seed action for 'level' steps
        #NOTE: assume initial configuration is all zeros
        origin = np.zeros(state_dim, dtype=np.float32)
        def propagate(state, act):
            s = state.copy()
            for _ in range(level):
                s = vectorized_dynamics(s[None, :], np.array([act]), dt, vrange)[0]
            return s

        seed_low   = propagate(origin, low_action)
        seed_mid   = propagate(origin, mid_action)
        seed_high  = propagate(origin, high_action)
        seeds = np.vstack((seed_low, seed_mid, seed_high))

        # pick k samples from the *filtered* candidates
        # call helper with seeds, to return exactly 'total' points
        free_t   = torch.tensor(free_states, device='cuda', dtype=torch.float32)
        seeds_t  = torch.tensor(seeds,       device='cuda', dtype=torch.float32)
        combined = farthest_point_sampling_torch(free_t, num_samples_per_level, seeds_t).cpu().numpy()

        # mark these new samples as visited
        new_idxs = uniformity_grid.get_index_vectorized(combined)
        for g in new_idxs:
            all_visited_indices.add(tuple(g))

        oversampled.append(combined)
        elapsed = time.time() - level_start_time
        print(f"  Level set {level}: generated {combined.shape[0]} samples in {elapsed:.2f}s")

    total_elapsed = time.time() - start_time
    print(f"Data preprocessing done in {total_elapsed:.2f}s")

    if visualization:
        visualize_states(oversampled, title="2D Visualization of Oversampled States")
        visualize_states(representative_list, title="2D Visualization of Level Sets Representatives")

    # with open("/home/mikasa/RSN/traj_sampling/map_conditioning/Kinematic_3D_trained_models/fixed_HD_10000_per_LS_oversampled_level_sets_disjoint.pkl", "wb") as f:
    #     pickle.dump(oversampled, f)
    #     print("fix 10000 samples oversampled saved")
    return oversampled, representative_list

def aggregate_environment_data(base_dir, skip=False, openspace=False):
    """
    Aggregate data from multiple environment folders.
    
    For each folder in base_dir starting with "env_", this function:
      1. Loads the level set data from "level_sets.pkl" using load_data().
      2. Loads the corresponding SDF map from "sdf.npy".
      3. Processes the level set data via prepare_trajectory_data() to generate oversampled states and 
         the representative list for each level set.
      4. Returns a list of dictionaries where each dictionary contains:
           - "env_folder": the path of the environment folder.
           - "config": the configuration dictionary from the level set data.
           - "oversampled_level_sets": list of oversampled states per level set.
           - "representative_list": list of representative states per level set.
           - "sdf": the loaded SDF numpy array.
    
    Args:
        base_dir (str): Base directory containing environment folders.
        skip (bool): Whether to skip the oversampled calculation.
        openspace (bool): Whether to use the openspace baseline.
    Returns:
        list: A list of dictionaries, each containing aggregated data for one environment.
    """
    print("Aggregate data from multiple environment folders...")
    env_data_list = []  # This list will store the aggregated data for each environment
    for folder_name in os.listdir(base_dir): # Iterate over each item in the base directory
        if not folder_name.startswith("env_"): # Consider only directories that start with "env_"
            continue
        
        env_folder = os.path.join(base_dir, folder_name)
        
        # Define the expected file paths for the level set data and SDF
        pkl_path = os.path.join(env_folder, "level_sets.pkl")
        sdf_path = os.path.join(env_folder, "sdf.npy")
        costmap_path = os.path.join(env_folder, "costmap.npy")
        
        if not (os.path.exists(pkl_path) and os.path.exists(sdf_path) and os.path.exists(costmap_path)):
            print(f"Skipping {env_folder}: required files not found.")
            continue
        
        # Load the level set data from the pickle file.
        # load_data() should return a dictionary that contains, among other items, a "config" key
        # and "pruned_level_set_representatives_across_LS".
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            data = load_data(pkl_path)
        config = data["config"]
        
        sdf = np.load(sdf_path)
        costmap = np.load(costmap_path)
        
        # Generate oversampled level sets and representative lists
        # with contextlib.redirect_stdout(open(os.devnull, 'w')):
        oversampled_level_sets, representative_list = prepare_trajectory_data(
            data, config, sdf=sdf, obstacles=None, visualization=False, skip=skip, openspace=openspace
        )
        assert len(representative_list) >= 2, "Level sets representatives list too short!"
        
        # Create a dictionary for this environment containing all the necessary data.
        env_data = {
            "env_folder": env_folder,
            "config": config,
            "oversampled_level_sets": oversampled_level_sets,
            "representative_list": representative_list,
            "sdf": sdf,
            "costmap": costmap,
            "map_embedding": None #NOTE: map_embedding is not used in the current implementation
        }
        
        # Append the environment's data to the overall list.
        env_data_list.append(env_data)
    
    # Return the list of aggregated environment data dictionaries.
    return env_data_list

def split_environments_data(env_data_list, train_ratio, val_ratio):
    """
    Split the list of environment data dictionaries into training, validation, and test sets.
    
    The split is performed on the environment level, meaning each sample corresponds to
    one full environment with its associated oversampled states, representative list, sdf, and map embedding.
    
    Args:
        env_data_list (list): List of dictionaries, each containing data for one environment.
        train_ratio (float): Proportion of environments to use for training.
        val_ratio (float): Proportion of environments to use for validation.
                          The test ratio is implicitly (1 - train_ratio - val_ratio).
    
    Returns:
        tuple: (train_envs, val_envs, test_envs)
            - train_envs: List of environment data dictionaries for training.
            - val_envs: List of environment data dictionaries for validation.
            - test_envs: List of environment data dictionaries for testing.
    """
    # Shuffle the environments to ensure a random split
    random.shuffle(env_data_list)
    
    total_envs = len(env_data_list)
    train_count = int(total_envs * train_ratio) # number of training environments
    val_count = int(total_envs * val_ratio) # number of validation environments
    test_count = total_envs - train_count - val_count # use remaining environments for testing

    # Debug prints to check splits
    print(f"Total environments: {total_envs}")
    print(f"  Training environments: {train_count}")
    print(f"  Validation environments: {val_count}")
    print(f"  Testing environments: {test_count}")

    # Create the splits using slicing
    train_envs = env_data_list[:train_count]
    val_envs = env_data_list[train_count:train_count + val_count]
    test_envs = env_data_list[train_count + val_count:]
    
    return train_envs, val_envs, test_envs

##############################
# Data Loading Function
##############################
def load_data(pkl_path):
    """ Load experiment data from a pickle file.  """
    print("Loading data...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        print("Data Overview:")
        print("-" * 20)
        print("Data Keys:")
        pprint(list(data.keys()), indent=4)  # List keys for better readability
        print("\nConfig Keys:")
        pprint(list(data["config"].keys()), indent=4)
        vrange = data["config"]['vrange']
        assert vrange[0] == vrange[1], "Should be constant velocity"
    print()
    return data

##############################
# Testing and Visualizations
##############################
def dynamics_KS_3d_steering_angle(state, action, dt, v = 1, vrange=None): #constant velocity
    x, y, theta = state
    steering_angle= action
    # v = np.clip(v, vrange[0], vrange[1])
    L_wb = 0.324 # wheelbase for F1Tenth
    theta_new = theta + v/L_wb * np.tan(steering_angle) * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi # wrap into [−π, π]:
    x_new = x + v * np.cos(theta_new) * dt 
    y_new = y + v * np.sin(theta_new) * dt
    return (x_new, y_new, theta_new) 

def cuda_dynamics_KS_3d_steering_angle_batched(states, actions, dt, v):
    """
    Vectorized dynamic propagation for a batch of state-action pairs.
    
    Instead of propagating each state with all possible actions,
    this function assumes that for each state in the batch a corresponding 
    action is provided. Thus, the function computes one propagated state 
    per input state-action pair.
    
    Args:
        states (torch.Tensor): Tensor of shape (batch_size, 3) representing
            the current states, where each state is [x, y, theta].
        actions (torch.Tensor): Tensor of shape (batch_size,) or (batch_size,1)
            representing the steering angle for each corresponding state.
        dt (float): Time step for propagation.
        v (float): Velocity, assumed constant across the batch.
        
    Returns:
        next_states (torch.Tensor): Tensor of shape (batch_size, 3) containing
            the propagated states for each state-action pair.
    """
    assert states.dim() == 2 and states.shape[1] == 3, "states must be of shape (batch_size, 3)"
    assert states.shape[0] == actions.shape[0], "Batch size of states and actions must match"
    assert actions.dim() in [1, 2] and (actions.dim() == 1 or actions.shape[1] == 1), \
        "actions must be of shape (batch_size,) or (batch_size, 1)"
    L_wb = 0.324  # Wheelbase for F1Tenth (distance between axles)
    
    # Extract state components: x, y positions and orientation theta.
    # Shape: (batch_size,)
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    
    # Compute the new orientation (theta_new) for each state-action pair.
    # each element in 'actions' corresponds to the desired steering angle for the respective state.
    theta_new = theta + (v / L_wb) * torch.tan(actions.squeeze()) * dt
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [−π, π]: 
    
    # Compute the new x position using the updated orientation.
    x_new = x + v * torch.cos(theta_new) * dt
    
    # Compute the new y position using the updated orientation.
    y_new = y + v * torch.sin(theta_new) * dt
    
    # Stack the propagated components to form the new states tensor.
    # Each state is now [x_new, y_new, theta_new].
    next_states = torch.stack([x_new, y_new, theta_new], dim=1)
    return next_states

def batch_states_to_grid(states, sdf_shape, resolution):
    """
    Converts robot states to normalized grid coordinates for F.grid_sample.

    Args:
        states (np.ndarray or torch.Tensor): Array of shape (N, D) where D>=2.
            Only the first two dimensions (x,y) are used.
        sdf_shape (tuple): The shape (H, W) of the SDF grid.
        resolution (float): The size of each grid cell (meters per cell).

    Returns:
        torch.Tensor: A grid of shape (1, N, 1, 2) with values in [-1, 1],
                      ready to be used as the grid input for F.grid_sample.
    """
    if isinstance(states, np.ndarray): # If input is numpy, convert to torch tensor.
        states = torch.from_numpy(states).float()
    xy = states[:, :2]  # shape: (N, 2)
    H, W = sdf_shape
    center_x = (W - 1) / 2.0
    center_y = (H - 1) / 2.0

    # Convert world coordinates (in meters) to pixel indices.
    # Column index: center_x + (x / resolution)
    # Row index: center_y - (y / resolution)  (assuming y increases upward in world frame)
    cols = center_x + xy[:, 0] / resolution
    rows = center_y - xy[:, 1] / resolution

    # Normalize to the interval [0, 1] relative to grid dimensions.
    cols_norm = cols / (W - 1)
    rows_norm = rows / (H - 1)

    # Convert to grid_sample coordinates in the interval [-1, 1].
    x_grid = 2 * cols_norm - 1
    y_grid = 2 * rows_norm - 1

    grid = torch.stack([x_grid, y_grid], axis=1) # Stack into (N, 2) 
    grid = grid.unsqueeze(0).unsqueeze(2) # reshape to (1, N, 1, 2) required by grid_sample.
    return grid

def batch_states_to_grid_batched(states, sdf_shape, resolutions):
    """
    Converts a batch of robot states to normalized grid coordinates for batched F.grid_sample.
    Handles different resolutions per batch item.

    Args:
        states (torch.Tensor): Tensor of shape (B, D) where D>=2.
        sdf_shape (tuple): The shape (H, W) of the SDF grid.
        resolutions (torch.Tensor): Tensor of shape (B,) with resolution for each item.

    Returns:
        torch.Tensor: A grid of shape (B, 1, 1, 2) with values in [-1, 1].
    """
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states).float()

    B = states.shape[0]
    xy = states[:, :2]  # (B, 2)
    H, W = sdf_shape
    center_x = (W - 1) / 2.0
    center_y = (H - 1) / 2.0

    # Ensure resolutions is a column vector for broadcasting: (B,) -> (B, 1)
    if resolutions.dim() == 1:
        resolutions = resolutions.unsqueeze(1)

    # Vectorized coordinate conversion using broadcasting
    # xy (B, 2) / resolutions (B, 1) -> (B, 2)
    scaled_xy = xy / resolutions

    # Calculate cols and rows indices. Note the Y-flip for rows.
    cols = center_x + scaled_xy[:, 0] # (B,)
    rows = center_y - scaled_xy[:, 1] # (B,)

    # Normalize to [0, 1]
    cols_norm = cols / (W - 1)
    rows_norm = rows / (H - 1)

    # Convert to grid_sample coordinates [-1, 1].
    x_grid = 2 * cols_norm - 1
    y_grid = 2 * rows_norm - 1

    # Stack into (B, 2). The order [x_grid, y_grid] is critical for F.grid_sample.
    grid = torch.stack([x_grid, y_grid], dim=1)

    # CRITICAL: Reshape to (B, 1, 1, 2) as required by F.grid_sample for batched input.
    grid = grid.view(B, 1, 1, 2)
    return grid

def bilinear_sample_sdf_features(sdf_features, states, resolution):
    """
    sdf_features: (B, C, H, W)
    states (np.ndarray or torch.Tensor): Robot states with shape (B, N, D)
            (only the first two coordinates are used).
    Returns: (B, N, C)
    """
    B, C, H, W = sdf_features.shape
    assert B == 1, "lets use multiple states to query the same feature map, batch operation not supported"
    grid = batch_states_to_grid(states, (H, W), resolution)   # → [1, N, 1, 2]

    # grid_sample expects (1, H_out, W_out, 2), so this is (1, N, 1, 2)
    sampled = F.grid_sample(sdf_features, grid, align_corners=False, mode='bilinear')  # returns (1, C, N, 1)
    sampled = sampled.squeeze(-1).squeeze(0)  # → [C, N]
    sampled = sampled.transpose(0, 1)         # → [N, C]
    return sampled

def bilinear_sample_sdf_features_batched(sdf_features, states, resolutions):
    """
    Performs bilinear sampling in a fully batched manner.

    Args:
        sdf_features (torch.Tensor): Shape (B, C, H, W).
        states (torch.Tensor): Shape (B, D).
        resolutions (torch.Tensor): Shape (B,).

    Returns:
        torch.Tensor: Shape (B, C).
    """
    B, C, H, W = sdf_features.shape
    
    grid = batch_states_to_grid_batched(states[:, :2], (H, W), resolutions)

    # Input: (B, C, H, W), Grid: (B, 1, 1, 2) -> Output: (B, C, 1, 1)
    # Use padding_mode='border' to handle states slightly outside the map boundaries safely.
    sampled = F.grid_sample(sdf_features, grid, align_corners=False, mode='bilinear', padding_mode='border')

    # Reshape output to (B, C)
    sampled = sampled.view(B, C)
    return sampled

def sample_trajectories_feasible(
        initial_state, actions, dynamics_cuda,
        num_trajectories, trajectory_length,
        model, feature_extractor, config, sdf_tensor, costmap_tensor, map_embedding,
        uniform_sampling=False,
    ):
    print("Sampling trajectories...")
    # Ensure deterministic trajectory sampling
    set_trajectory_sampling_seeds(2025)  # Use consistent seed for trajectory sampling
    
    start_time = time.time()
    t_step = config["dt"]
    v = config["vrange"][0]

    # Pre-convert actions to GPU tensor for efficient indexing
    actions_tensor = torch.tensor(actions, dtype=torch.float32, device="cuda")

    # Shape after repeat: (num_trajectories, state_dim)
    batch_current_states = (
        torch.tensor(initial_state, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(num_trajectories, 1)
        .cuda()
    )
    state_dim = batch_current_states.shape[1]

    # Preallocate a tensor to store states for each time step.
    # Shape: (trajectory_length+1, num_trajectories, state_dim)
    trajectory_states = torch.empty((trajectory_length + 1, num_trajectories, state_dim),
                                    dtype=torch.float32, device="cuda")
    trajectory_states[0] = batch_current_states

    # Preallocate a tensor to record the chosen action indices at each step.
    # Shape: (trajectory_length, num_trajectories)
    trajectory_actions = torch.empty((trajectory_length, num_trajectories),
                                     dtype=torch.int64, device="cuda")

    if hasattr(model, '_orig_mod') and model._orig_mod is not None:
        model_name = model._orig_mod.__class__.__name__
    else:
        model_name = model.__class__.__name__

    if model_name == "MapAct_PixelInterpolated":
        # For pixel-interpolated model, use the dense feature map and bilinear interpolation
        dense_features = feature_extractor(sdf_tensor)  # [1, feature_dim, H, W]
    for step in range(trajectory_length):
        theta = batch_current_states[:, 2]
        theta_sin = torch.sin(theta)
        theta_cos = torch.cos(theta)

        network_current_states = torch.hstack((
            batch_current_states[:, 0:2],
            theta_sin.unsqueeze(1),
            theta_cos.unsqueeze(1)
        ))

        # Predict action probabilities in batch using the model
        if model_name == "MapAct_PixelInterpolated":
            # dense_features = feature_extractor(costmap_tensor)  # [1, feature_dim, H, W]
            interpolated_features = bilinear_sample_sdf_features(
                sdf_features=dense_features, 
                states=network_current_states, 
                resolution=0.05 #NOTE: resolution is hard coded for now
            )
            current_probabilities = model(network_current_states, interpolated_features).detach()
            #TODO: check why runtime increase as num_trajectories increase, should be parallelized?
        elif model_name == "MapAct":
            if uniform_sampling:
                # Use uniform probabilities instead of model predictions
                batch_size = network_current_states.shape[0]
                num_actions = len(actions)
                current_probabilities = torch.ones((batch_size, num_actions), 
                                                 device=network_current_states.device, 
                                                 dtype=torch.float32) / num_actions
            else:
                current_probabilities = model(network_current_states).detach()
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        # if step == 0:
            # current_probabilities = torch.full_like(current_probabilities, 1.0 / current_probabilities.size(1))
        # Use torch.multinomial to efficiently sample one action index per trajectory.
        chosen_action_indices = torch.multinomial(current_probabilities, num_samples=1).squeeze(1)

        # Save the chosen action indices into the preallocated tensor.
        trajectory_actions[step] = chosen_action_indices.to("cuda", torch.int64)

        # Convert the chosen actions using efficient tensor indexing (OPTIMIZED & Verified)
        chosen_actions_tensor = actions_tensor[chosen_action_indices]

        # action jitter
        ACTION_PERTURBATION = False
        if ACTION_PERTURBATION:
            steer_min = float(np.min(actions))
            steer_max = float(np.max(actions))
            # small zero‑mean Gaussian noise
            noise = torch.randn_like(chosen_actions_tensor) * 0.05
            chosen_actions_tensor = chosen_actions_tensor + noise
            chosen_actions_tensor = torch.clamp(chosen_actions_tensor, min=steer_min, max=steer_max)

        # Propagate states in parallel.
        batch_current_states = dynamics_cuda(batch_current_states, chosen_actions_tensor, t_step, v)

        STATE_PERTURBATION = False
        if STATE_PERTURBATION:
            sigma_xy    = 0.05
            sigma_theta = 0.05
            noise_xy    = torch.randn_like(batch_current_states[:, 0:2]) * sigma_xy # xy noise
            noise_th    = torch.randn_like(batch_current_states[:, 2:3]) * sigma_theta # θ noise
            batch_current_states = torch.cat([
                batch_current_states[:, 0:2] + noise_xy,
                batch_current_states[:, 2:3] + noise_th,
            ], dim=1)

        # Record the newly updated states.
        trajectory_states[step + 1] = batch_current_states

    trajectory_states_cpu = trajectory_states.detach().cpu().numpy()  # Shape: (T+1, num_trajectories, state_dim)
    trajectory_actions_cpu = trajectory_actions.detach().cpu().numpy()  # Shape: (T, num_trajectories)
    
    # Create the final list of trajectories by combining states and actions.
    # This conversion is done once after the whole propagation loop.
    trajectories = []
    for i in range(num_trajectories):
        traj = []
        # For each time step, pair the state with the corresponding action.
        for t in range(trajectory_length):
            state = trajectory_states_cpu[t, i].tolist()
            action = actions[trajectory_actions_cpu[t, i]]
            traj.append((state, action))
        # Append the final state with a terminal marker (None for action).
        final_state = trajectory_states_cpu[trajectory_length, i].tolist()
        traj.append((final_state, None))
        trajectories.append(traj)
    print(f"    Trajectory sampling process took {time.time() - start_time} seconds")
    return trajectories

def visualize_trajectories_background(
    trajectories, costmap, resolution=0.05,
    show_vis=False, save_vis=False, vis_filepath=None,
    alpha=None, marker_size=None, title=None,
):
    """
    Visualizes the sampled trajectories on the costmap with colored markers for each state.

    Args:
        trajectories (list): List of trajectories, each a list of (state, action) tuples.
        costmap (np.ndarray): Costmap for visualization background.
        resolution (float): Grid resolution in meters per cell (default: 0.05).
        show_vis (bool): If True, display the visualization.
        save_vis (bool): If True, save the visualization to vis_filepath.
        vis_filepath (str, optional): Path to save the visualization if save_vis is True.
        title (str, optional): Custom title for the plot. If None, uses default title.
    """
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
    # Use a colormap that gives distinct colors for different time steps
    alpha = alpha if alpha is not None else 0.1 # default alpha value
    marker_size = marker_size if marker_size is not None else 1 # default marker size
    if trajectories and len(trajectories) > 0 and len(trajectories[0]) > 0:
        num_steps = len(trajectories[0])
        cmap_traj = plt.cm.get_cmap('viridis', num_steps)  # Renamed to avoid conflict
        
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
    if legend_handles:
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
            ncols = min(3, (len(legend_handles) + 2) // 3)  # Up to 3 columns
            plt.legend(handles=legend_handles, loc='upper right', 
                      title=legend_title, fontsize='x-small', 
                      title_fontsize='small', ncol=ncols, 
                      columnspacing=0.5, handletextpad=0.3)

    if save_vis and vis_filepath:
        plt.savefig(vis_filepath, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {vis_filepath}")
    elif show_vis:
        plt.show()
    else:
        plt.close()
    print(f"    Visualization took {time.time() - visual_start} seconds")

def traj_quantitative_analysis(env, trajectories, logging_filepath):
    quantitative_analysis_start_time = time.time()
    # === Input Assertions ===
    assert isinstance(env, dict), "env must be a dictionary."
    for key in ["config", "representative_list", "sdf"]:
        assert key in env, f"env must contain key '{key}'."
    config = env["config"]
    assert isinstance(config, dict), "env['config'] must be a dictionary."
    assert "thresholds" in config, "env['config'] must contain 'thresholds'."
    thresholds = config["thresholds"]
    assert isinstance(thresholds, (list, np.ndarray)), "'thresholds' must be a list or np.ndarray."
    
    rep_list_global = env["representative_list"]
    assert isinstance(rep_list_global, list), "env['representative_list'] must be a list."
    assert len(rep_list_global) > 0, "env['representative_list'] must not be empty."
    
    assert isinstance(trajectories, list), "trajectories must be a list."
    assert len(trajectories) > 0, "trajectories list must not be empty."
    expected_length = len(trajectories[0])
    for traj in trajectories:
        assert isinstance(traj, list), "Each trajectory must be a list."
        assert len(traj) == expected_length, "All trajectories must have the same length."
        for elem in traj:
            assert isinstance(elem, tuple), "Each element of a trajectory must be a tuple (state, action)."
            assert len(elem) == 2, "Each trajectory element must contain exactly 2 items: (state, action)."
    
    # === Setup Grid for Indexing ===
    uniformity_grid = Grid(thresholds=thresholds)
    
    # === Compute Ground Truth (Reachable) Indices per Level Set ===
    reachable_indices_per_t = []
    for t, rep_list in enumerate(rep_list_global):
        assert isinstance(rep_list, list), f"Representative list at time step {t} must be a list."
        rep_states = np.array([rep.numpy() if hasattr(rep, 'numpy') else rep for rep in rep_list])
        assert rep_states.ndim == 2, f"Representative states at time step {t} must be a 2D array."
        rep_indices = uniformity_grid.get_index_vectorized(rep_states)
        reachable_indices = set(tuple(idx) for idx in rep_indices)
        reachable_indices_per_t.append(reachable_indices)
    
    # === Coverage Analysis: Count Only States Falling in Reachable Cells ===
    overall_unique_indices = set()
    num_steps = expected_length  # number of time steps in each trajectory
    per_level_unique = [set() for _ in range(num_steps)]
    for traj in trajectories:
        for t, (state, _) in enumerate(traj):
            grid_index = uniformity_grid.get_index(state)
            if grid_index in reachable_indices_per_t[t]:
                overall_unique_indices.add(grid_index)
                per_level_unique[t].add(grid_index)
    overall_coverage = len(overall_unique_indices)
    
    # === Collision Analysis: Hard-Coded SDF Parameters (Temporary) ===
    def check_collision(env, state):
        sdf = env["sdf"]
        assert isinstance(sdf, np.ndarray), "env['sdf'] must be a numpy array."
        assert sdf.ndim == 2, "env['sdf'] must be a 2D array."
        H, W = sdf.shape
        resolution = 0.05  # Hard-coded temporary workaround
        center_index_x = (W - 1) / 2.0
        center_index_y = (H - 1) / 2.0
        x, y = state[0], state[1]
        col = center_index_x + x / resolution
        row = center_index_y - y / resolution
        iy = int(np.clip(round(col), 0, W - 1))
        ix = int(np.clip(round(row), 0, H - 1))
        return sdf[ix, iy] <= 0.0
    colliding_traj_count = sum(1 for traj in trajectories if any(check_collision(env, state) for state, _ in traj))
    collision_rate = colliding_traj_count / len(trajectories) if len(trajectories) > 0 else 0.0

    # === Build Compact Log String ===
    total_trajs = len(trajectories)
    log_lines = []
    log_lines.append("Trajectory Analysis:")
    log_lines.append(f"  Overall: Trajs={total_trajs} | UniqueHits={overall_coverage} | Collisions={colliding_traj_count} (Rate={collision_rate*100:.1f}%)")
    log_lines.append("  Per Level Set Coverage (UniqueHit/TotalReachable):")
    for t in range(num_steps):
        total_reachable = len(reachable_indices_per_t[t])
        hit_count = len(per_level_unique[t])
        perc = (hit_count / total_reachable * 100) if total_reachable > 0 else 0.0
        log_lines.append(f"    LS{t}: {hit_count}/{total_reachable} ({perc:.1f}%)")
    log_text = "\n".join(log_lines)
    
    # === Save Log to File ===
    with open(logging_filepath, "w") as f:
        f.write(log_text)
    print(f"Quantitative Analysis saved to {logging_filepath}")
    print(f"    Analysis took {time.time() - quantitative_analysis_start_time} seconds")
    return

def test_and_save_trajectories(
    env, model, feature_extractor, vis_folder,
    num_trajectories=1000, show_vis=False, save_vis=True, save_pickle=False
):
    """
    Tests the model on a given environment, samples trajectories, and optionally visualizes/saves them.

    Args:
        env (dict): Environment data including config, costmap, sdf, map_embedding, etc.
            - checkout function aggregate_environment_data() for specifications of env format
        model: Trained model for trajectory sampling.
        vis_folder (str): Directory to save visualizations.
        num_trajectories (int): Number of trajectories to sample.
        show_vis (bool): If True, display the visualization.
        save_vis (bool): If True, save the visualization.
    """
    env_name = os.path.basename(env["env_folder"])
    vis_filepath = os.path.join(vis_folder, f"{env_name}.png") if save_vis else None
    if vis_filepath:
        base_filepath, _ = os.path.splitext(vis_filepath)
        logging_filepath = base_filepath + "_analysis.txt"
    else:
        logging_filepath = os.path.join(vis_folder, f"{env_name}_analysis.txt")

    map_embedding = env["map_embedding"]
    if map_embedding is not None:
        map_embedding = map_embedding.to('cuda')

    sdf_t   = torch.tensor(env["sdf"],     dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
    cost_t  = torch.tensor(env["costmap"], dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)

    # Generate trajectories
    trajectories = sample_trajectories_feasible(
        initial_state=np.zeros(3).astype(np.float32),
        actions=env["config"]["actions"][:, 0],
        dynamics_cuda=cuda_dynamics_KS_3d_steering_angle_batched,
        num_trajectories=num_trajectories,
        trajectory_length=int(env["config"]["total_t"] / env["config"]["dt"]),
        model=model,
        feature_extractor=feature_extractor,
        config=env["config"],
        sdf_tensor=sdf_t,
        costmap_tensor=cost_t,
        map_embedding=map_embedding,
        uniform_sampling=False,
    )
    if save_pickle:
        fname = (
            f"{env_name}_{model.__class__.__name__}_"
            f"{num_trajectories}traj_dt{env['config']['dt']}_"
            f"T{int(env['config']['total_t'])}.pkl"
        )
        pkl_path = os.path.join(vis_folder, fname)
        with open(pkl_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"Saved trajectories to {pkl_path}")

    # -----------------------------------------------------------------
    # Quantitative Analysis:
    # Use the trajectory quantitative analysis function to compute:
    #   - Coverage: Only counting states that fall within the collision-free 
    #     (reachable) representative indices for each time step.
    #   - Collision Rate: Fraction of trajectories that hit an obstacle.
    # The analysis is then saved to a structured logging file.
    # -----------------------------------------------------------------
    traj_quantitative_analysis(env, trajectories, logging_filepath)

    # Visualize trajectories if requested
    if show_vis or save_vis:
        visualize_trajectories_background(
            trajectories=trajectories,
            costmap=env["costmap"],
            resolution=0.05,
            show_vis=show_vis,
            save_vis=save_vis,
            vis_filepath=vis_filepath
        )

    print(f"Processed {env_name}" + (f", visualization saved to {vis_filepath}" if save_vis else ""))

def plot_loss_curves(
        train_entropy_loss, train_obstacle_loss, 
        train_uniform_entropy_loss, train_uniform_obstacle_loss,
        val_entropy_loss, val_obstacle_loss,
        train_recon_loss, val_recon_loss,
        val_uniform_entropy_loss, val_uniform_obstacle_loss,
        vis_folder
    ):
    """
    Plots the loss curves for training and validation.
    The plot is saved as 'loss_curves.png' in vis_folder.
    If any input list is empty, its curve is skipped.
    """
    epochs = range(1, len(train_entropy_loss) + 1)
    plt.figure(figsize=(10, 6))

    # Helper function for safe plotting.
    def maybe_plot(epochs, data, label, marker, linestyle, color):
        if data and len(data) > 0:
            plt.plot(epochs, data, label=label, marker=marker, linestyle=linestyle, color=color)

    ### Entropy Loss (red pair)
    maybe_plot(epochs, train_entropy_loss, 'Train Entropy Loss', 'o', '-', 'lightcoral')
    maybe_plot(epochs, val_entropy_loss, 'Val Entropy Loss', 'x', '--', 'red')
    
    ### Uniform (Baseline) Losses (dotted gray pair)
    maybe_plot(epochs, train_uniform_entropy_loss, 'Train Uniform (Baseline) Entropy Loss', 's', ':', '#cccccc')
    maybe_plot(epochs, val_uniform_entropy_loss, 'Val Uniform (Baseline) Entropy Loss', 's', ':', '#999999')
    maybe_plot(epochs, train_uniform_obstacle_loss, 'Train Uniform (Baseline) Obstacle Loss', 's', ':', '#bbbbbb')
    maybe_plot(epochs, val_uniform_obstacle_loss, 'Val Uniform (Baseline) Obstacle Loss', 's', ':', '#777777')

    ### Obstacle Loss (green pair)
    maybe_plot(epochs, train_obstacle_loss, 'Train Obstacle Loss', 'o', '-', 'lightgreen')
    maybe_plot(epochs, val_obstacle_loss, 'Val Obstacle Loss', 'x', '--', 'green')
    
    ### Reconstruction Loss (orange pair)
    maybe_plot(epochs, train_recon_loss, 'Train Reconstruction Loss', 'o', '-', 'khaki')
    maybe_plot(epochs, val_recon_loss, 'Val Reconstruction Loss', 'x', '--', 'darkorange')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filepath = os.path.join(vis_folder, 'loss_curves.png')
    plt.savefig(plot_filepath, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {plot_filepath}")

def parallelized_trajectory_sampling_cuda(action_prob_data: dict, num_trajectories: int = 10000, device='cuda'):
    """
    CUDA-accelerated parallel trajectory sampling using efficient nearest neighbor search.
    Pre-builds data structures for fast action probability lookup.
    
    Args:
        action_prob_data: Dictionary containing level sets and action probabilities
        num_trajectories: Number of trajectories to generate
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        list: Generated trajectories
    """
    # Ensure deterministic trajectory sampling
    set_trajectory_sampling_seeds(2025)  # Use consistent seed for trajectory sampling
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    config = action_prob_data['config']
    level_sets = action_prob_data['level_sets']
    
    # Import dynamics helpers
    if config['state_dim'] == 3:
        from flow_Cuniform.dynamics_helpers import dynamics_KS_3d_steering_angle as dynamics_func
    elif config['state_dim'] == 4:
        from flow_Cuniform.dynamics_helpers import dynamics_KS_4d_steering_angle_v as dynamics_func
    else:
        raise ValueError(f"Unsupported state dimension: {config['state_dim']}")
    
    # Generate actions
    from flow_Cuniform.utility_helpers import generate_actions
    actions = generate_actions(
        config["arange"], config["num_a"], 
        config["steering_angle_range"], config["num_steering_angle"], 
        True
    )
    
    state_dim = config['state_dim']
    dt = config['dt']
    max_steps = int(config['total_t'] / dt)
    
    print(f"  Building efficient lookup structures for {len(level_sets)} level sets...")
    
    # Build efficient lookup structures using CUDA
    device = torch.device(device)
    all_states = []
    all_probs = []
    level_boundaries = [0]  # Track where each level starts in the flattened arrays
    
    for level_set in level_sets:
        if level_set is not None and len(level_set) > 0:
            states_tensor = torch.tensor(level_set[:, :state_dim], dtype=torch.float32, device=device)
            probs_tensor = torch.tensor(level_set[:, state_dim:], dtype=torch.float32, device=device)
            all_states.append(states_tensor)
            all_probs.append(probs_tensor)
            level_boundaries.append(level_boundaries[-1] + len(level_set))
    
    if not all_states:
        return []
    
    # Concatenate all states and probabilities for efficient batch processing
    all_states_tensor = torch.cat(all_states, dim=0)  # (total_states, state_dim)
    all_probs_tensor = torch.cat(all_probs, dim=0)    # (total_states, num_actions)
    level_boundaries_tensor = torch.tensor(level_boundaries, device=device)
    
    print(f"  Total states: {all_states_tensor.shape[0]}, Processing {num_trajectories} trajectories...")
    
    # Get initial states from first level set
    initial_states = all_states[:1] if all_states else []
    if not initial_states or len(initial_states[0]) == 0:
        return []
    
    initial_states_tensor = initial_states[0]  # (num_initial_states, state_dim)
    trajectories = []
    
    # Process trajectories in batches for memory efficiency
    batch_size = min(1000, num_trajectories)
    num_batches = (num_trajectories + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_trajectories)
        current_batch_size = batch_end - batch_start
        
        batch_trajectories = []
        
        # Random initial states for this batch using deterministic generator
        generator = torch.Generator(device=device).manual_seed(2025 + batch_idx)
        init_indices = torch.randint(len(initial_states_tensor), (current_batch_size,), device=device, generator=generator)
        current_states = initial_states_tensor[init_indices].clone()  # (batch_size, state_dim)
        
        for step in range(max_steps):
            #TODO: consider only look for nearest neighbors on the current level set
            # Find nearest neighbors for all states in batch using efficient vectorized search
            batch_action_probs = find_nearest_action_probs_cuda(
                current_states, all_states_tensor, all_probs_tensor
            )  # (batch_size, num_actions)
            
            action_indices = torch.multinomial(batch_action_probs, num_samples=1).squeeze(-1)  # (batch_size,)
            # Convert to numpy first to avoid slow tensor creation warning
            action_indices_np = action_indices.cpu().numpy()
            selected_actions_np = actions[action_indices_np]  # Efficient numpy indexing
            selected_actions = torch.tensor(selected_actions_np, dtype=torch.float32, device=device)  # (batch_size, action_dim)
            
            # Store trajectory step for each trajectory in batch
            for i in range(current_batch_size):
                if step == 0:
                    batch_trajectories.append([])
                
                state_tuple = tuple(current_states[i].cpu().numpy())
                action_tuple = tuple(selected_actions[i].cpu().numpy())
                batch_trajectories[i].append((state_tuple, action_tuple))
            
            # Apply dynamics to entire batch (vectorized)
            try:
                if config['state_dim'] == 3:
                    # For 3D, apply dynamics one by one (since dynamics_func expects single state)
                    next_states = []
                    for i in range(current_batch_size):
                        next_state = dynamics_func(current_states[i].cpu().numpy(), 
                                                 selected_actions[i].cpu().numpy(), 
                                                 dt, config['vrange'])
                        next_states.append(next_state)
                    current_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                elif config['state_dim'] == 4:
                    # For 4D, similar approach (can be optimized further with proper vectorized dynamics)
                    next_states = []
                    for i in range(current_batch_size):
                        next_state = dynamics_func(current_states[i].cpu().numpy(), 
                                                 selected_actions[i].cpu().numpy(), 
                                                 dt, config['vrange'])
                        next_states.append(next_state)
                    current_states = torch.tensor(next_states, dtype=torch.float32, device=device)
            except Exception as e:
                # Fallback: use uniform action probabilities
                print(f"  Warning: Dynamics failed at step {step}, using fallback")
                break
        
        # Add final states to trajectories
        for i in range(current_batch_size):
            state_tuple = tuple(current_states[i].cpu().numpy())
            batch_trajectories[i].append((state_tuple, None))
        
        trajectories.extend(batch_trajectories)
        
        # Clean up GPU memory
        del current_states, batch_action_probs, action_indices, selected_actions
        torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"    Processed {batch_idx + 1}/{num_batches} batches...")
    
    # Clean up
    del all_states_tensor, all_probs_tensor, level_boundaries_tensor
    torch.cuda.empty_cache()
    
    print(f"   Generated {len(trajectories)} trajectories efficiently")
    return trajectories

def find_nearest_action_probs_cuda(query_states, all_states, all_probs):
    """
    Find nearest neighbors for query states using CUDA-accelerated computation.
    Matches the exact behavior of find_closest_action_probs but with CUDA acceleration.
    
    Args:
        query_states: (batch_size, state_dim) tensor of query states
        all_states: (total_states, state_dim) tensor of all available states  
        all_probs: (total_states, num_actions) tensor of corresponding action probabilities
        
    Returns:
        (batch_size, num_actions) tensor of action probabilities
    """
    # Compute distances efficiently using broadcasting
    distances = torch.cdist(query_states, all_states, p=2)  # (batch_size, total_states)
    
    # Find nearest neighbor for each query state
    _, nearest_indices = torch.min(distances, dim=1)  # (batch_size,)
    
    # Get action probabilities for nearest neighbors
    action_probs = all_probs[nearest_indices]  # (batch_size, num_actions)
    
    # Normalize probabilities (matching find_closest_action_probs behavior)
    row_sums = action_probs.sum(dim=1, keepdim=True)
    action_probs = action_probs / row_sums
    
    return action_probs

def create_sdf_from_obstacles(obstacles, grid_size, resolution):
    """
    Create a costmap and compute the signed distance function (SDF) from a list of obstacles.
    The world (robot) frame is assumed to have (0,0) at the center.
    
    Args:
        obstacles: list of obstacles (circular or rectangular).
        grid_size: size of the square grid.
        resolution: cell size in meters.
    
    Returns:
        costmap: binary numpy array (True for obstacle).
        sdf: numpy array of shape (grid_size, grid_size).
    """
    costmap = np.zeros((grid_size, grid_size), dtype=bool)
    center_index = (grid_size - 1) / 2.0  # center of grid in index coordinates
    rows, cols = np.indices((grid_size, grid_size))
    
    for obs in obstacles:
        if len(obs) == 3:
            # Circular obstacle: (x, y, radius)
            x_world, y_world, r = obs
            col_center = x_world / resolution + center_index
            row_center = center_index - y_world / resolution
            dist_sq = (cols - col_center) ** 2 + (rows - row_center) ** 2
            circle_radius_cells = r / resolution
            costmap[dist_sq <= circle_radius_cells ** 2] = True
        elif len(obs) == 4:
            # Rectangular obstacle: (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = obs
            col_min = int(np.floor(x_min / resolution + center_index))
            col_max = int(np.ceil(x_max / resolution + center_index))
            row_min = int(np.floor(center_index - y_max / resolution))
            row_max = int(np.ceil(center_index - y_min / resolution))
            costmap[row_min:row_max, col_min:col_max] = True
        else:
            print("Warning: Unrecognized obstacle format", obs)
    
    # Compute SDF: distance from free cells to obstacles (and vice versa)
    dist_out = distance_transform_edt(~costmap) * resolution 
    dist_in = distance_transform_edt(costmap) * resolution
    sdf = dist_out.copy()
    sdf[costmap] = -dist_in[costmap]
    return costmap, sdf
