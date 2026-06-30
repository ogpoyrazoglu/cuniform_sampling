import numpy as np
import os
import torch
import torch.nn.functional as F
import pickle
import time
import math
import copy
from collections import defaultdict
from classes.grid import Grid, AdaptiveGrid
from classes.graph_structure import Node
from flow_Cuniform.dynamics_helpers import (
    dynamics_dubins_4d,
    vectorized_dynamics_dubins_4d,

    dynamics_dubins, 
    inverse_dynamics_dubins, 
    vectorized_dynamics_dubins, 

    dynamics_KS_3d_steering_angle,
    inverse_dynamics_KS_3d_steering_angle,
    vectorized_dynamics_KS_3d_steering_angle,

    dynamics_KS_4d_steering_angle_v,
    inverse_dynamics_KS_4d_steering_angle_v_vectorized,
    vectorized_dynamics_KS_4d_steering_angle_v,

    dynamics_KS_3d_v_cmd,
    inverse_dynamics_KS_3d_v_cmd_vectorized,
    vectorized_dynamics_KS_3d_v_cmd,

    dynamics_2d_random_walk,
    inverse_dynamics_2d_random_walk,
    vectorized_dynamics_2D_Walk,
)
from flow_Cuniform.utility_helpers import (
    obstacles_to_str,
    flow_to_action_prob_approximation_all,
    plot_total_runtime_vs_horizon,
    sample_trajectories_network_flow,
    visualize_trajectories,
    visualize_graph_flow_and_low_arcs_nodes,
    get_state_u_distribution_across_LS,
    generate_uniform_trajectories,
    analyze_trajectory_distribution,
    generate_actions,
    setup_single_transition_flow,
    precompute_graph_structure_parallel,
    prune_graph,
    calculate_reachable_level_sets_adaptive_uniformity,
    calculate_reachable_level_sets,
)
from pprint import pprint

##################################################### Helper Functions #####################################################
def parallelized_collision_checker_nodes(points, obstacles=None, sdf=None, resolution=0.05, sdf_inflation=0.0):
    """
    Vectorized collision checker for multiple points using SDF or obstacles.
    
    Args:
        points (np.ndarray): Array of shape (N_points, 2) with (x, y) coordinates in world frame.
        obstacles (list, optional): List of obstacles. Rectangles: (x_min, y_min, x_max, y_max),
                                    Circles: (cx, cy, r). Ignored if sdf is provided.
        sdf (np.ndarray, optional): 2D array representing the SDF.
        resolution (float): SDF grid cell size in meters.
        sdf_inflation (float): Safety margin for collision detection when using SDF. Points are 
                              considered in collision if their SDF value is <= sdf_inflation. 
                              Higher values create a larger safety buffer around obstacles.
                              Default is 0.0 meters.

    Returns:
        np.ndarray: Boolean array of shape (N_points,) where True indicates collision.
    """
    assert points.ndim == 2 and points.shape[1] == 2, \
        f"points must be of shape (N_points, 2), got {points.shape}"

    if sdf is not None:
        H, W = sdf.shape # sdf is a 2D numpy array [H, W]
        sdf_tensor = torch.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

        # Extract point coordinates
        x = points[:, 0]  # N_points
        y = points[:, 1]  # N_points

        # Normalize coordinates to grid indices
        center_index_x = (W - 1) / 2.0
        center_index_y = (H - 1) / 2.0
        col = center_index_x + x / resolution
        row = center_index_y - y / resolution

        # Normalize to [0, 1]
        col_normalized = col / (W - 1)
        row_normalized = row / (H - 1)

        # Convert to [-1, 1] for grid_sample
        x_grid = 2 * col_normalized - 1
        y_grid = 2 * row_normalized - 1

        # Create grid with shape [1, N_points, 1, 2]
        grid_np = np.stack([x_grid, y_grid], axis=1)  # Shape: [N_points, 2]
        grid = grid_np[None, :, None, :]  # Shape: [1, N_points, 1, 2]
        grid = torch.from_numpy(grid).float()

        # Sample SDF values
        values = F.grid_sample(
            sdf_tensor,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # Output: [1, 1, N_points, 1]

        if values.shape != (1, 1, values.shape[2], 1):
            print(f"values shape not expected in parallelized_collision_checker: {values.shape}, input: {points}")
            print("values.squeeze shape: ", values.squeeze().shape)
            values = values.reshape(-1)  # results in shape (N_points,)

        # Squeeze to get per-point values
        values = values.squeeze()  # Shape: [N_points]

        # Determine collisions using the configurable inflation parameter
        collision = values <= sdf_inflation
        return collision.numpy()
    else:
        # Obstacle-based collision checking
        if obstacles is None:
            return np.zeros(points.shape[0], dtype=bool)
        # Initialize collision array
        collision = np.zeros(points.shape[0], dtype=bool)
        # Separate rectangles and circles
        rectangles = [obs for obs in obstacles if len(obs) == 4]
        circles = [obs for obs in obstacles if len(obs) == 3]
        # Check rectangles
        if rectangles:
            rect_array = np.array(rectangles)  # (N_rect, 4)
            inside = (
                (points[:, 0, None] >= rect_array[None, :, 0]) &  # x >= x_min
                (points[:, 0, None] <= rect_array[None, :, 2]) &  # x <= x_max
                (points[:, 1, None] >= rect_array[None, :, 1]) &  # y >= y_min
                (points[:, 1, None] <= rect_array[None, :, 3])    # y <= y_max
            )
            collision |= np.any(inside, axis=1)
        # Check circles
        if circles:
            circ_array = np.array(circles)  # (N_circ, 3)
            dist = np.sqrt(
                (points[:, 0, None] - circ_array[None, :, 0])**2 +
                (points[:, 1, None] - circ_array[None, :, 1])**2
            )
            inside = dist <= circ_array[None, :, 2]
            collision |= np.any(inside, axis=1)
        return collision

def _backpropagate_one_level(graph, next_level_probs, incoming_counts):
    """
    Performs one step of probability back-propagation for a single graph transition.

    It calculates the probability for each node in the current level set (t) based on the
    probabilities of the nodes they connect to in the next level set (t+1). It also
    updates the flow value on each arc in the graph.

    Args:
        graph (dict): The graph dictionary for the transition t -> t+1.
        next_level_probs (dict): A dictionary mapping node_id -> probability for nodes in level t+1.
        incoming_counts (defaultdict): A dictionary mapping node_id -> incoming arc count.

    Returns:
        dict: A dictionary mapping node_id -> probability for the nodes in the current level set (t).
    """
    current_level_idx = -1
    all_levels = {node.level_set for node in graph["nodes"].values()}
    if not all_levels:
        print("Warning: No nodes found in the current level set. Cannot compute flow.")
        return {}  # Handle empty graph

    current_level_idx = min(all_levels)

    nodes_in_current_level = [
        node for node in graph["nodes"].values() if node.level_set == current_level_idx
    ]

    current_level_probs = {}
    for u in nodes_in_current_level:
        node_prob = 0.0
        updated_outgoing_arcs = []
        for to_node_id, _ in u.outgoing_arcs:
            if to_node_id not in next_level_probs:
                continue

            prob_v = next_level_probs.get(to_node_id, 0.0)
            count_v = incoming_counts.get(to_node_id, 0)

            # Proof relies on Backward Connectivity. If a node (not in L0) has
            # no parents, it means prune_graph failed. We must crash if this happens.
            assert count_v > 0, f"Node {to_node_id} has an in-degree of 0, violating core assumptions"
            arc_flow = prob_v / count_v
            node_prob += arc_flow
            updated_outgoing_arcs.append((to_node_id, arc_flow))

        u.outgoing_arcs = updated_outgoing_arcs
        current_level_probs[u.id] = node_prob

    # Normalize the probabilities for the current level to sum to 1
    total_prob_t = sum(current_level_probs.values())
    assert np.isclose(total_prob_t, 1.0), f"Total probability at level {current_level_idx} is {total_prob_t}, not 1."

    for node_id in current_level_probs:
        current_level_probs[node_id] /= total_prob_t
    
    return current_level_probs

def compute_network_flow(graphs):
    """
    Distributes flows across the graphs for all level sets.
    Args:
        graphs (list): List of graph dictionaries, one per level set transition.
    Returns:
        graphs_with_flow (list): A list of graph dictionaries with flow distributed among arcs.
    """
    start_time_all = time.time()
    print("=" * 80)
    print("Solving Network Flow for Level Set Transition Graphs...")
    print("=" * 80)
    max_flow_runtime = []
    for t, graph in enumerate(graphs):
        start_time_t = time.time()
        flow_ratio = setup_single_transition_flow(graph) # Solve network flow for the graph
        elapsed_time = time.time() - start_time_t
        max_flow_runtime.append(elapsed_time)
    end_time_all = time.time()
    elapsed_time_all = end_time_all - start_time_all
    print(f"Compute network flow for all graphs takes {elapsed_time_all:.4f} seconds.\n")
    return graphs, max_flow_runtime

def load_or_compute_reachable_set(filename, config, disjoint_level_set, obstacles, sdf):
    # NOTE: the level set cannot be simply obtained by the masking because 
    #       L_i(the level set for the obstacle free case) is not a subset of L'i (the level set for the obstacle case)
    #       Because even if a configure x is collision free, it may take longer to reach it
    if sdf is None and os.path.exists(filename):
        print(f"File {filename} exists. Reading graphs from file.\n")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return  data['reachable_indicex_across_LS'], \
                    data['reachable_representative_across_LS'], \
                    data['finer_repre_LSs'], \
                    [],
    else:
        print("\nReachable cells data not found. Computing from scratch...")
        if config["adaptive_uniformity"]:
            ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtime = calculate_reachable_level_sets_adaptive_uniformity(
                config, disjoint_level_set=disjoint_level_set, 
                obstacles=obstacles, sdf=sdf, collision_checker_nodes=parallelized_collision_checker_nodes
            )
        else:
            ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtime = calculate_reachable_level_sets(
                config, disjoint_level_set=disjoint_level_set, adaptive_resolution=True,
                obstacles=obstacles, sdf=sdf, collision_checker_nodes=parallelized_collision_checker_nodes
            )

        # instaeed of pop the "config", make a deep copy of it and then modify on the copied config and save it
        config_copy = copy.deepcopy(config)
        keys_to_remove = ["dynamics", "inverse_dynamics", "vectorized_dynamics"] # remove keys that require extra dependencies
        for key in keys_to_remove:
            config_copy.pop(key, None)  # Using pop to avoid KeyError if the key doesn't exist
        with open(filename, 'wb') as f:
            pickle.dump({
                'reachable_indicex_across_LS': ReaBoxIndices_LSs, # under uniformity resolution
                'reachable_representative_across_LS': ReaBox_LSs, # under uniformity resolution
                'finer_repre_LSs': finer_repre_LSs,
                'config': config_copy,
            }, f)
        print(f"Reachable cells data saved to {filename}.\n")
        return ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtime

def load_or_compute_final_data(
        filename, 
        ReaBoxIndices_LSs, ReaBox_LSs, 
        config, obstacles, sdf, parallelized_collision_checker_nodes
    ):
    """ Load final processed data from a file, or compute and save it if unavailable. """
    MODEL = config["model_name"]
    if sdf is None and os.path.exists(filename):
        print(f"File {filename} exists. Reading final processed data from file.\n")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data['graphs'], data['action_prob_list'], data, [], []
    else:
        ################# Build Graphs Structure #################
        raw_graphs, graph_runtime = precompute_graph_structure_parallel(ReaBox_LSs, ReaBoxIndices_LSs, config)
        print("Final processed data not found. Computing from scratch...")

        ############## Prune Graphs ##############
        pruned_graph = prune_graph(raw_graphs)

        ############## Copmute Network Flow ##############
        pruned_graph_with_flow, max_flow_runtime = compute_network_flow(pruned_graph)

        # NOTE: can use the following function to approximate action probability
        action_prob_list = flow_to_action_prob_approximation_all(graphs=pruned_graph_with_flow, model_config=config)
        print()
        assert type(action_prob_list[0]) == np.ndarray, "action_prob_list should be a list of numpy arrays"
        for i, action_prob in enumerate(action_prob_list):
            print(f"  action_prob_list[{i}]: ", action_prob.shape)
        print()

        keys_to_remove = ["dynamics", "inverse_dynamics", "vectorized_dynamics"] # remove keys that require extra dependencies
        for key in keys_to_remove:
            config.pop(key, None)  # Using pop to avoid KeyError if the key doesn't exist

        # convert the pruned graph to pruned level set representatives
        conversion_start = time.time()
        pruned_level_set_representatives_across_LS = []

        print("    Converting pruned graph into level set representatives...")
        for i, graph in enumerate(pruned_graph_with_flow):
            pruned_level_set_representatives = {node.point for node in graph["nodes"].values() if node.level_set == i}
            pruned_level_set_representatives_across_LS.append(pruned_level_set_representatives)
        last_level = len(pruned_graph_with_flow)
        last_level_set = {node.point for node in pruned_graph_with_flow[-1]["nodes"].values() if node.level_set == last_level}
        pruned_level_set_representatives_across_LS.append(last_level_set)
        print("    Number of elements in each pruned level set:")
        for idx, level_set in enumerate(pruned_level_set_representatives_across_LS):
            print(f"      Level set {idx}: {len(level_set)} elements")
        print(f"    Conversion done... Take {time.time() - conversion_start}")

        final_data = {
            'config': config,
            'graphs': pruned_graph_with_flow,
            'action_prob_list': action_prob_list,
            'reachable_indicex_across_LS': ReaBoxIndices_LSs,
            'reachable_representative_across_LS': ReaBox_LSs,
            'pruned_level_set_representatives_across_LS': pruned_level_set_representatives_across_LS,
            'model': MODEL,
            'environment': obstacles
        }
        with open(filename, 'wb') as f:
            pickle.dump(final_data, f, protocol=5)
        print(f"Final processed data saved to {filename}.\n")
        return pruned_graph_with_flow, action_prob_list, final_data, graph_runtime, max_flow_runtime

def main():
    seed = 2025
    np.random.seed(seed)
    UNIFORM_ACTION_TRAJ = False
    DISJOINT_LEVEL_SET = True
    MULTIPLE_INITIAL_CONFIG = False
    # MODEL = "DUBINS_4D"
    # MODEL = "DUBINS"
    # MODEL = "2D_RANDOM_WALK"
    # MODEL = "KS_3D_STEERING_ANGLE"
    MODEL = "KS_3D_V_CMD"
    # MODEL = "DUBINS_4D"
    # MODEL = "KS_4D_STEERING_ANGLE_V"

    ############################### System Dynamic Configuration Parameters ###############################
    model_configs = {
        "DUBINS_4D": { 
        #NOTE: this model is not currently supported, 
            # missing 'arange' 'num_a' 'steering_angle_range' 'num_steering_angle' field
            "model_name": "DUBINS_4D",
            "dynamics": dynamics_dubins_4d,
            "inverse_dynamics": inverse_dynamics_dubins, 
            "vectorized_dynamics": vectorized_dynamics_dubins_4d,
            "perturbation_param": 2.01,         # offsets add to representative at level set t
            "vrange": (0.0, 1.0),               # Velocity range (min, max)
            "arange": (-3.0, 3.0),              # acceleration range (min, max)
            "num_a": 5,
            "steering_angle_range": (-45, 45),  # Steering range in degrees
            "num_steering_angle": 11,
            "actions": None,                    # placeholder, will define later
            "thresholds": [0.05, 0.05, (2 * math.pi)/40, 0.2],
            "state_dim": 4,
            "dt" : 0.2,
            "total_t" : 1.01,
            "adaptive_uniformity": False,
            "sdf_inflation": 0.15,  # Safety margin for collision detection (meters)
        },
        "DUBINS": {
            "model_name": "DUBINS",
            "dynamics": dynamics_dubins,
            "inverse_dynamics": inverse_dynamics_dubins, 
            "vectorized_dynamics": vectorized_dynamics_dubins,
            "perturbation_param": 2.01,         # offsets add to representative at level set t
            "vrange": (1.0, 1.0),               # Velocity range (min, max)
            "arange": (0.0, 0.0),               # acceleration range (min, max)
            "num_a": 1,
            "steering_angle_range": (-45, 45),  # Steering range in degrees
            "num_steering_angle": 45,
            "actions": None,                    # placeholder, will define later
            "thresholds": [0.05, 0.05, (2 * math.pi) / 80],
            "state_dim": 3,
            "v": 1,
            "dt" : 0.2,
            "total_t" : 2.01,
            "adaptive_uniformity": False,
            "sdf_inflation": 0.15,  # Safety margin for collision detection (meters)
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
            "total_t" : 0.51,
            "adaptive_uniformity": True,
            "sdf_inflation": 0.15,  # Safety margin for collision detection (meters)
        },
        "KS_3D_V_CMD": {
            # 3D Kinematic Single-Track model with DIRECT velocity command input.
            # State:   (x, y, theta)                   - 3D
            # Action:  (steering_angle [rad], v_cmd [m/s]) - 2D
            # v_cmd is the direct longitudinal velocity command (no acceleration state).
            # Actions are discretized as steering_angle × v_cmd using vrange and num_v.
            "model_name": "KS_3D_V_CMD",
            "dynamics": dynamics_KS_3d_v_cmd,
            "inverse_dynamics": inverse_dynamics_KS_3d_v_cmd_vectorized,
            "vectorized_dynamics": vectorized_dynamics_KS_3d_v_cmd,
            "perturbation_param": 2.01,
            "vrange": (-1.0, 1.0),               # v_cmd range (m/s), also used for clipping
            "num_v": 11,                          # Number of v_cmd bins (discretizes vrange)
            "steering_angle_range": (-30.0, 30.0),  # Steering range in degrees
            "num_steering_angle": 15,
            "actions": None,                     # placeholder, will define later
            "thresholds": [0.05, 0.05, (2 * math.pi)/40],  # x, y, theta
            "state_dim": 3,                      # (x, y, theta)
            "wheelbase": 0.324,                  # F1Tenth wheelbase (m)
            "dt" : 0.20,
            "total_t" : 4.01,
            "adaptive_uniformity": False,
            "sdf_inflation": 0.15,
        },
        "KS_3D_STEERING_ANGLE": { 
            "model_name": "KS_3D_STEERING_ANGLE",
            "dynamics": dynamics_KS_3d_steering_angle,
            "inverse_dynamics": inverse_dynamics_KS_3d_steering_angle, 
            "vectorized_dynamics": vectorized_dynamics_KS_3d_steering_angle,
            "perturbation_param": 2.01,
            "vrange": (2.5, 2.5),               # Velocity range (min, max)
            "arange": (0.0, 0.0),               # acceleration range (min, max)
            "num_a": 1,
            "steering_angle_range": (-30.0, 30.0),  # Steering range in degrees
            "num_steering_angle": 31,
            "actions": None,                    # placeholder, will define later
            "thresholds": [0.10, 0.10, (2 * math.pi)/60], 
            "state_dim": 3,                     # (x, y, theta - yaw angle)
            "dt" : 0.20,
            "total_t" : 1.21,
            "adaptive_uniformity": False,
            "sdf_inflation": 0.0,  # Safety margin for collision detection (meters)
        },
        "2D_RANDOM_WALK": {
            "model_name": "2D_RANDOM_WALK",
            "dynamics": dynamics_2d_random_walk,
            "inverse_dynamics": inverse_dynamics_2d_random_walk, 
            "vectorized_dynamics": vectorized_dynamics_2D_Walk,
            "perturbation_param": 0.0,          # 0 means no perturbation
            "vrange": (1.0, 1.0),               # Velocity range (min, max)
            "arange": (0.0, 0.0),               # acceleration range (min, max)
            "num_a": 1,
            "steering_angle_range": (-1, 1),    # Steering range in degrees
            "num_steering_angle": 101,
            "actions": None,                    # placeholder, will define later
            "thresholds": [0.02, 0.02],
            "state_dim": 2,
            "dt" : 1.0,
            "total_t" : 100.01,
            "adaptive_uniformity": False,
            "sdf_inflation": 0.15,  # Safety margin for collision detection (meters)
        }
    }
    if MODEL not in model_configs:
        raise ValueError(f"Invalid MODEL type: {MODEL}. Please choose from {list(model_configs.keys())}.")

    config = model_configs[MODEL]
    if MODEL == "KS_3D_V_CMD":
        # KS_3D_V_CMD uses vrange + num_v instead of arange + num_a
        config["actions"] = generate_actions(
            config["vrange"], config["num_v"], config["steering_angle_range"], config["num_steering_angle"], MODEL!='2D_RANDOM_WALK'
        )
    else:
        config["actions"] = generate_actions(
            config["arange"], config["num_a"], config["steering_angle_range"], config["num_steering_angle"], MODEL!='2D_RANDOM_WALK'
        )
    dynamics = config["dynamics"]
    actions = config["actions"]
    thresholds = config["thresholds"]
    t_step = config["dt"]
    total_t = config["total_t"] 
    perturbation = config["perturbation_param"]
    vrange = config["vrange"]
    num_steering_angle = config["num_steering_angle"]

    if MULTIPLE_INITIAL_CONFIG:
        # assume we have different initial velocities as the level set 0
        v_min, v_max      = config["vrange"]
        dv                = config["thresholds"][-1]   # e.g. 1.0 m/s
        # build a 1D array [0.0,1.0,2.0,…,v_max]
        vel_bins          = np.arange(v_min, v_max + dv, dv, dtype=np.float32)
        # now make one 4-D state per v-bin: (x=0,y=0,θ=0,v=bin)
        initial_state_set     = np.zeros((len(vel_bins), config["state_dim"]), dtype=np.float32)
        initial_state_set[:,3] = vel_bins  # fill in each row's v-column
    else:
        # assume the robot always start from origin in the configuration space,
        # not necessarilly true for manipulator type of dynamics or higher dimensional systems
        initial_state_set = np.zeros((1, config["state_dim"]), dtype=np.float32)  # Shape: (1, state_dim)
    config["initial_state_set"] = initial_state_set
    # config["initial_state_set"] = np.array([[0.0, 0.0, 0.0, 3.0]])

    if MODEL in {"DUBINS", "KS_3D_STEERING_ANGLE", "DUBINS_4D", "KS_3D_V_CMD"}:
        thresholds_copy = thresholds[:]
        thresholds_copy[2] = round(math.degrees(thresholds[2]), 1)  # Convert radians to degrees and round to 1 digit
        grid_resolution_str = '_'.join([f"{threshold:.3f}" for threshold in thresholds_copy]) + "deg_"
    elif MODEL in {"DUBINS_4D", "KS_4D_STEERING_ANGLE_V"}:
        thresholds_copy = thresholds[:]
        thresholds_copy[2] = round(math.degrees(thresholds[2]), 1)  # Convert radians to degrees and round to 1 digit
        grid_resolution_str = '_'.join([f"{threshold:.3f}" for threshold in thresholds_copy]) + "_"
    elif MODEL in {"2D_RANDOM_WALK"}:
        grid_resolution_str = '_'.join([f"{threshold:.3f}" for threshold in thresholds]) + "_"
    
    print("=" * 80)
    print("Configuration Parameters")
    print("=" * 80)
    print(f"Seed: {seed}")
    print(f"Disjoint Level Set: {DISJOINT_LEVEL_SET}")
    print(f"Model: {MODEL}")
    print(f"Actions: \n{actions}")
    print(f"Grid Resolution: {thresholds}")
    print(f"Time Step: {t_step}")
    print(f"Initial State Set: \n{initial_state_set}")
    print(f"Total Time Length: {total_t}")
    print(f"Perturbation parameter: {perturbation}")
    print(f"Velocity Range: {config.get('vrange', 'N/A')}")
    if 'arange' in config:
        print(f"Acceleration Range: {config['arange']}")
        print(f"Number of Acceleration Values: {config['num_a']}")
    if 'num_v' in config:
        print(f"V_CMD Range (from vrange): {config['vrange']}")
        print(f"Number of V_CMD Values: {config['num_v']}")
    print(f"Steering Angle Range: {config.get('steering_angle_range', 'N/A')}")
    print(f"Number of Steering Angles: {config.get('num_steering_angle', 'N/A')}")

    if config["adaptive_uniformity"]:
        g = AdaptiveGrid(
            base_thresholds=thresholds,
            dt=config["dt"],
            max_velocity=config["vrange"][1],
            max_acceleration=max(abs(config["arange"][0]), abs(config["arange"][1])),
            max_steering_deg=max(abs(config["steering_angle_range"][0]), abs(config["steering_angle_range"][1])),
            wheelbase=0.324  # Vehicle wheelbase from dynamics
        )
        g.print_upper_bounds()
    else:
        g = Grid(thresholds=thresholds)
    # num_trajectories_list = [100]
    num_trajectories_list = [50000]
    # num_trajectories_list = [250, 500, 1000, 2500, 5000, 10000]

    # obstacles = [
    #     # (1.0, -1.0, 1.5, 1.0)
    #     # (1.0, -1.0, 1.5, 0.0)
    #     # (1.0, 0.0, 1.5, 1.0)
    #     # (0.25, -0.10, 0.35, -0.025)
    #     (1.0, 0.0, 0.3) # circle x, y, radius
    # ]
    # sdf = np.load("/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/env_01/sdf.npy")
    obstacles = None
    sdf = None
    def generate_filename(
            prefix, disjoint_LS, init, MODEL, seed, grid_resolution_str,
            total_t, t_step,
            perturb, 
            vrange, num_steering_angle, steering_angle_range,
            obstacles_str,
        ):
        return (
            f"{prefix}_"
            f"{disjoint_LS}"
            f"{init}"
            f"{MODEL}_"
            f"obs_{obstacles_str}_"
            f"perturb_{perturb}_"
            f"seed_{seed}_"
            f"grid_{grid_resolution_str}"
            f"t{total_t}_"
            f"ts{t_step}_"
            f"vrange_{vrange[0]}_{vrange[1]}_"
            f"steer_range_{steering_angle_range[0]}_{steering_angle_range[1]}_"
            f"steering_{num_steering_angle}.pkl"
        )
    disjoint_LS = "disjoint_" if DISJOINT_LEVEL_SET else "overlapping_"
    init = "multi_init_" if MULTIPLE_INITIAL_CONFIG else "1_init_"
    obstacles_str = obstacles_to_str(obstacles)

    level_sets_filename = generate_filename(
        "C_Uniform_reachability", disjoint_LS, init, MODEL, seed, grid_resolution_str, 
        total_t, t_step, perturbation, vrange, num_steering_angle, config["steering_angle_range"],
        obstacles_str,
    )
    final_data_filename = generate_filename(
        "C_Uniform_processed", disjoint_LS, init, MODEL, seed, grid_resolution_str,
        total_t, t_step, perturbation, vrange, num_steering_angle, config["steering_angle_range"],
        obstacles_str,
    )

    if UNIFORM_ACTION_TRAJ:
        num_trajectories_uniform_action = 10000
        generate_uniform_trajectories(
            config=config,
            num_trajectories=num_trajectories_uniform_action,
            trajectory_length=int(total_t/t_step),
            output_file=f'uniform_sampled_actions_trajectories_{num_trajectories_uniform_action}.pickle'
        ) 

    ############################### Main C_Uniform Logic ###############################
    ################### Exploration Phase ###################
    ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtime = load_or_compute_reachable_set(
        level_sets_filename, config, DISJOINT_LEVEL_SET, obstacles, sdf
    )

    ################### Load or compute final processed graph data ###################
    pruned_graph_with_flow, action_prob_list, final_data_to_save, graph_runtime, max_flow_runtime = load_or_compute_final_data(
        final_data_filename, ReaBoxIndices_LSs, ReaBox_LSs, config,
        obstacles, sdf, parallelized_collision_checker_nodes=parallelized_collision_checker_nodes
    )
    # visualize_graph_flow_and_low_arcs_nodes(pruned_graph_with_flow[0])
    # plot_total_runtime_vs_horizon(level_set_runtime, graph_runtime, max_flow_runtime, dt=config["dt"])

    # Loop through each number of trajectories
    print("Sampling trajectories...")
    for num_trajectories in num_trajectories_list:
        traj_filename = generate_filename(
            f"C_Uniform_{num_trajectories}_trajectories", 
            disjoint_LS, init, MODEL, seed, grid_resolution_str, total_t,
            t_step, perturbation, vrange, num_steering_angle, config["steering_angle_range"],
            obstacles_str,
        )
        sampled_trajectories_graph = sample_trajectories_network_flow(pruned_graph_with_flow, num_trajectories, seed)
        visualize_trajectories(sampled_trajectories_graph, xy_only=True, obstacles=obstacles) # only x, y
        # if MODEL not in {"2D_RANDOM_WALK"}:
        #     visualize_trajectories(sampled_trajectories_graph, xy_only=False) # visualize in the x,y,theta space
        with open(traj_filename, 'wb') as file:
            pickle.dump(sampled_trajectories_graph, file, protocol=5)
        print(f"Sampled {num_trajectories} trajectories(graph) and saved to {traj_filename}.\n")
        analyze_trajectory_distribution(ReaBoxIndices_LSs, traj_filename, g, level_set_range=(0, int(total_t/t_step)), nf_assertion=True)

if __name__ == "__main__":
    main()