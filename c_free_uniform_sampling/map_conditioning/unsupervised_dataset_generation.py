'''
This script is used to generate a dataset of level set representatives for different environments..
Not used anymore since we switch to supervised approach that scales better.
'''
import os
import time
import copy
import math
import torch
import contextlib
import random
import numpy as np
import pickle
from pprint import pprint
from scipy.ndimage import distance_transform_edt
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from flow_Cuniform.dynamics_helpers import (
    dynamics_KS_3d_steering_angle,
    inverse_dynamics_KS_3d_steering_angle,
    vectorized_dynamics_KS_3d_steering_angle,
)
from flow_Cuniform.utility_helpers import (
    generate_actions,
    precompute_graph_structure_parallel,
    prune_graph,
    calculate_reachable_level_sets,
)
from flow_Cuniform.c_uniform_sampling import(
    parallelized_collision_checker_nodes,
)

from map_conditioning.utility_helper_map import(
    visualize_sdf,
)

torch.manual_seed(2025)
np.random.seed(2025)
random.seed(2025)

# NOTE: make sure the configuration is same everywhere
model_configs = {
    "KS_3D_STEERING_ANGLE": {
        "model_name": "KS_3D_STEERING_ANGLE",
        "dynamics": dynamics_KS_3d_steering_angle,
        "inverse_dynamics": inverse_dynamics_KS_3d_steering_angle,
        "vectorized_dynamics": vectorized_dynamics_KS_3d_steering_angle,
        "perturbation_param": 2.01,
        "slack_parameter": 0.00,
        "vrange": (1.25, 1.25),
        "arange": (0.0, 0.0),
        "num_a": 1,
        "steering_angle_range": (-30.0, 30.0),
        "num_steering_angle": 31,
        "actions": None,
        # "thresholds": [0.05, 0.05, (2 * np.pi)/60],
        "thresholds": [0.10, 0.10, (2 * np.pi)/60],
        # "thresholds": [0.025, 0.025, (2 * np.pi)/60],
        "state_dim": 3,
        "dt": 0.10,
        "total_t": 2.41,
        "initial_state_set": np.array([[0.0, 0.0, 0.0]]),  # Fixed format as np.array with shape (1, 3)
        "adaptive_uniformity": False,
    }
}
config = model_configs["KS_3D_STEERING_ANGLE"]
config["actions"] = generate_actions(
    config["arange"], config["num_a"], config["steering_angle_range"], 
    config["num_steering_angle"], deg2rad_conversion=True
)

# -------------------------------
# Module: Obstacle and SDF Generation
# -------------------------------
def generate_random_obstacles(num_obstacles):
    """
    Generate a fixed list of 2 obstacles in the robot (world) frame.
    
    Returns:
        obstacles: list of obstacles.
            Circular obstacle format: (x, y, radius)
            Rectangular obstacle format: (x_min, y_min, x_max, y_max)
    """
    obstacles = []
    while len(obstacles) < num_obstacles:
        # obs_type = random.choice([0, 1])
        obs_type = 0
        if obs_type == 0:
            # Circular obstacle
            x = random.uniform(-1, 3)
            y = random.uniform(-3, 3)
            r = random.uniform(0.1, 1.0)
            # Check if the origin (0,0) lies within the circle.
            distance_from_origin = math.sqrt(x**2 + y**2)
            if distance_from_origin <= r+0.1:
                continue  # Resample if the circle covers the origin.
            obstacles.append((x, y, r))
        else:
            # Rectangular obstacle
            x_min = random.uniform(-3.0, 2.9)
            x_max = random.uniform(x_min + 0.1, 3)
            y_min = random.uniform(-3, 2.9)
            y_max = random.uniform(y_min + 0.1, 3)
            # Check if the origin (0,0) is inside the rectangle.
            if x_min <= 0 <= x_max and y_min <= 0 <= y_max:
                continue  # Resample if the rectangle contains the origin.
            obstacles.append((x_min, y_min, x_max, y_max))
    return obstacles

def create_sdf_from_obstacles(obstacles, grid_size=121, resolution=0.05):
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
    
    if np.sum(costmap) == 0:
        costmap[0, :] = True
        costmap[-1, :] = True
        costmap[:, 0] = True
        costmap[:, -1] = True
    
    # Compute SDF: distance from free cells to obstacles (and vice versa)
    dist_out = distance_transform_edt(~costmap) * resolution 
    dist_in = distance_transform_edt(costmap) * resolution
    sdf = dist_out.copy()
    sdf[costmap] = -dist_in[costmap]
    return costmap, sdf

# -------------------------------
# Module: Level Set Generation
# -------------------------------
def compute_level_set_representatives(
        config, obstacles, sdf, resolution,
        use_fixed_oversampled_states
    ):
    """
    Compute pruned level set representatives for a given configuration and obstacles using SDF-based collision checking.
    
    Args:
        config (dict): Model configuration (dynamics, actions, etc.).
        obstacles (list): List of obstacles from generate_random_obstacles().
        sdf (np.ndarray): Signed distance function array.
        resolution (float): Grid resolution in meters.
    
    Returns:
        list: List of sets, where each set contains pruned representative points for a level set.
    """
    # Step 1: Compute reachable level sets
    with contextlib.redirect_stdout(None):
        if use_fixed_oversampled_states: # generate representatives in open space environment
            ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, _ = calculate_reachable_level_sets(
                config,
                disjoint_level_set=True,
                adaptive_resolution=True,
                obstacles=None,
                sdf=None,
                collision_checker_nodes=None
            )
        else:
            ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, _ = calculate_reachable_level_sets(
                config,
                disjoint_level_set=True,
                adaptive_resolution=True,
                obstacles=obstacles,
                sdf=sdf,
                collision_checker_nodes=lambda *args, **kwargs: parallelized_collision_checker_nodes(
                    *args, resolution=resolution, **kwargs
                ),
            )
            if ReaBoxIndices_LSs is None and ReaBox_LSs is None:
                return None, None
    
    # Step 2: Build graph structure(Suppress print statements)
    with contextlib.redirect_stdout(None):
        raw_graphs, _ = precompute_graph_structure_parallel(ReaBox_LSs, ReaBoxIndices_LSs, config)
    
    # Step 3: Prune the graph
    with contextlib.redirect_stdout(None):
        pruned_graphs = prune_graph(raw_graphs)
    
    # Step 4: Extract pruned level set representatives
    pruned_level_set_representatives_across_LS = []
    for i in range(len(pruned_graphs)):
        representatives = {node.point for node in pruned_graphs[i]["nodes"].values() if node.level_set == i}
        pruned_level_set_representatives_across_LS.append(representatives)
        if len(representatives) == 0:
            return None, None # meaning after pruning, no representative left.
    last_level = len(pruned_graphs)
    last_level_set = {
        node.point 
        for node in pruned_graphs[-1]["nodes"].values()
        if node.level_set == last_level
    }
    pruned_level_set_representatives_across_LS.append(last_level_set)
    return ReaBoxIndices_LSs, pruned_level_set_representatives_across_LS

def generate_level_set_representatives(config, obstacles, sdf, resolution):
    """
    Generate a level set representation dictionary with configuration and pruned representatives.
    
    Args:
        config (dict): Model configuration.
        obstacles (list): List of obstacles.
        sdf (np.ndarray): Signed distance function array.
        resolution (float): Grid resolution in meters.
    
    Returns:
        dict: Dictionary containing config, obstacles, representatives, and info.
    """
    # Compute the pruned level set representatives
    (
        ReaBoxIndices_LSs, pruned_level_set_representatives_across_LS
    ) = compute_level_set_representatives(
        config, obstacles, sdf, resolution, use_fixed_oversampled_states=False
    )
    if ReaBoxIndices_LSs is None and pruned_level_set_representatives_across_LS is None:
        return None
    pruned_level_set_representatives_across_LS = [
        list(level_set) 
        for level_set in pruned_level_set_representatives_across_LS
    ]
    
    # Prepare a serializable config copy
    config_copy = copy.deepcopy(config)
    keys_to_remove = ["dynamics", "inverse_dynamics", "vectorized_dynamics"]
    for key in keys_to_remove:
        config_copy.pop(key, None)
    
    # Create the level set dictionary
    level_set_info = {
        "config": config_copy,
        "environment": obstacles,
        "reachable_indicex_across_LS": ReaBoxIndices_LSs,
        "pruned_level_set_representatives_across_LS": pruned_level_set_representatives_across_LS,
        #TODO: also save the oversampled states, make them balance on level set basis
        "info": "Pruned level set representatives for each level set"
    }
    
    return level_set_info

def save_level_set_visualization(level_set_info, filepath, max_samples=5000, title="Level Sets Visualization"):
    """
    Visualize the level set representatives in 2D (x, y) and save the plot.
    This function is based on the provided visualize_states function.
    """
    plt.figure(figsize=(10, 6))
    for idx, states in enumerate(level_set_info):
        # Convert states to numpy array if necessary
        if isinstance(states[0], torch.Tensor):
            states_np = torch.stack(states).cpu().numpy()
        else:
            states_np = np.array(states)
        if states_np.shape[0] > max_samples:
            sample_indices = np.random.choice(states_np.shape[0], max_samples, replace=False)
            states_np = states_np[sample_indices]
        plt.scatter(states_np[:, 0], states_np[:, 1], s=10, alpha=0.6, label=f"Time Step {idx}")
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis('equal')
    plt.legend()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved level set visualization to {filepath}")

# -------------------------------
# Module: Environment Folder Creation & Dataset Assembly
# -------------------------------
def create_environment_folder(env_name, base_dir="dataset"):
    """
    Create an environment folder under base_dir.
    """
    env_folder = os.path.join(base_dir, env_name)
    os.makedirs(env_folder, exist_ok=True)
    print(f"Created folder: {env_folder}")
    return env_folder

def create_environment_dataset(env_name, base_dir="dataset", grid_size=121, resolution=0.05, openspace=False):
    """
    For each environment, randomly generate obstacles.
    Save 5 files inside the environment folder:
      - sdf.npy
      - costmap.npy
      - scene.png
      - level_sets.pkl
      - level_sets_vis.png
    """
    env_folder = create_environment_folder(env_name, base_dir)
    
    max_attempts = 100  # Prevent infinite loops
    for attempt in range(max_attempts):
        # Generate random obstacles
        num_obstacles = random.randint(5, 10)
        obstacles = [] if openspace else  generate_random_obstacles(num_obstacles)
        num_obstacles = len(obstacles)
        # obstacles = [
        #     (1.0, 0.0, 0.3) # circle x, y, radius
        # ]
        print(f"Attempt {attempt+1}: Generated {num_obstacles} obstacles for {env_name}")
    
        # Create SDF from obstacles
        costmap, sdf = create_sdf_from_obstacles(obstacles, grid_size, resolution)

        # Generate and save level set representation
        level_set_info = generate_level_set_representatives(config, obstacles, sdf, resolution)
        if level_set_info is not None:
            print(f"Valid environment found after {attempt+1} attempts.")
            break
        else:
            print(f"Attempt {attempt+1}: Environment invalid (some level sets empty). Resampling obstacles...")

    costmap_filepath = os.path.join(env_folder, "costmap.npy")
    sdf_filepath = os.path.join(env_folder, "sdf.npy")
    np.save(sdf_filepath, sdf)
    np.save(costmap_filepath, costmap)
    print(f"Saved Costmap to {costmap_filepath}")
    print(f"Saved SDF to {sdf_filepath}")

    # Generate and save SDF visualization as image
    scene_filepath = os.path.join(env_folder, "sdf.png")
    visualize_sdf(sdf, resolution, scene_filepath)
        
    level_set_filepath = os.path.join(env_folder, "level_sets.pkl")
    with open(level_set_filepath, "wb") as f:
        pickle.dump(level_set_info, f)
    print(f"Saved level set representation to {level_set_filepath}")

    level_set_vis_filepath = os.path.join(env_folder, "level_set_vis.png")
    save_level_set_visualization(
        level_set_info["pruned_level_set_representatives_across_LS"], level_set_vis_filepath
    )

# -------------------------------
# Main Module: Create Multiple Environment Datasets
# -------------------------------
def main():
    base_dir = "dataset"
    os.makedirs(base_dir, exist_ok=True)
    
    random_dataset_base_dir = "dataset/open_space_env_v_1.25_dt0.1_t2.41"
    os.makedirs(random_dataset_base_dir, exist_ok=True)

    env_names = ["env_01"]
    # env_names = ["env_01", "env_02", "env_03"]
    # env_names = [f"env_{i:02d}" for i in range(1, 1001)] # List of environment names to generate.
    total_env = len(env_names)
    for i, env_name in enumerate(env_names, start=1):
        print(f"-----------Generating dataset {i}/{total_env}: {env_name}-----------")
        create_environment_dataset(env_name, random_dataset_base_dir, openspace=True)
    print("All environment datasets have been generated.")

if __name__ == "__main__":
    main()
