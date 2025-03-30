import numpy as np
import random
import os
import pickle
import time
import math
import copy
from ortools.graph.python import max_flow
from classes.grid import Grid
from classes.graph_structure import Node
from dynamics_helpers import (
    dynamics_dubins, 
    inverse_dynamics_dubins, 
    vectorized_dynamics_dubins, 

    dynamics_2d_random_walk,
    inverse_dynamics_2d_random_walk,
    vectorized_dynamics_2D_Walk,
)
from utility_helpers import (
    perturb_state,
    flow_to_action_prob_approximation_all,
    plot_total_runtime_vs_horizon,
    sample_trajectories_using_action_probability,
    sample_trajectories_network_flow,
    visualize_trajectories,
    visualize_graph_connection,
    visualize_graph_flow_and_low_arcs_nodes,
    generate_uniform_trajectories,
    analyze_trajectory_distribution,
    generate_actions,
    setup_single_transition_flow,
    calculate_reachable_level_sets,
    precompute_graph_structure,
    collision_checker_node,
    prune_graph,
)

##################################################### Helper Functions #####################################################
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

def load_or_compute_reachable_set(filename, config, disjoint_level_set, obstacles):
    # NOTE: the level set cannot be simply obtained by the masking because 
    #       L_i(the level set for the obstacle free case) is not a subset of L’_i (the level set for the obstacle case)
    #       Because even if a configure x is collision free, it may take longer to reach it
    if os.path.exists(filename):
        print(f"File {filename} exists. Reading graphs from file.\n")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return  data['reachable_indicex_across_LS'], \
                    data['reachable_representative_across_LS'], \
                    data['finer_repre_LSs'], \
                    [],
    else:
        print("\nReachable cells data not found. Computing from scratch...")
        ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtime = calculate_reachable_level_sets(config)

        # instead of pop the "config", make a deep copy of it and then modify on the copied config and save it
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
        config, obstacles, collision_checker_node
    ):
    """ Load final processed data from a file, or compute and save it if unavailable. """
    MODEL = config["model_name"]
    if os.path.exists(filename):
        print(f"File {filename} exists. Reading final processed data from file.\n")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data['graphs'], data['action_prob_list'], data, [], []
    else:
        ################# Build Graphs Structure #################
        raw_graphs, graph_runtime = precompute_graph_structure(ReaBox_LSs, ReaBoxIndices_LSs, config)
        print("Final processed data not found. Computing from scratch...")
        ############## Prune Graphs ##############
        pruned_graph = prune_graph(raw_graphs, obstacles=obstacles, collision_checker_node=collision_checker_node)
        ############## Copmute Network Flow ##############
        pruned_graph_with_flow, max_flow_runtime = compute_network_flow(pruned_graph)

        action_prob_list = None
        # NOTE: can use the following function to approximate action probability
        # action_prob_list = flow_to_action_prob_approximation_all(graphs=pruned_graph_with_flow, model_config=config)

        keys_to_remove = ["dynamics", "inverse_dynamics", "vectorized_dynamics"] # remove keys that require extra dependencies
        for key in keys_to_remove:
            config.pop(key, None)  # Using pop to avoid KeyError if the key doesn't exist

        # convert the pruned graph to pruned level set representatives
        pruned_level_set_representatives_across_LS = []
        if obstacles is not None:
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
            print("    Conversion done...")

        final_data = {
            'config': config,
            'graphs': pruned_graph_with_flow,
            'action_prob_list': action_prob_list,
            'reachable_indicex_across_LS': ReaBoxIndices_LSs,
            'reachable_representative_across_LS': ReaBox_LSs,
            'pruned_level_set_representatives_across_LS': pruned_level_set_representatives_across_LS,
            'model': MODEL,
        }
        with open(filename, 'wb') as f:
            pickle.dump(final_data, f, protocol=5)
        print(f"Final processed data saved to {filename}.\n")
        return pruned_graph_with_flow, action_prob_list, final_data, graph_runtime, max_flow_runtime

def main():
    MODEL = "DUBINS"
    # MODEL = "2D_RANDOM_WALK"

    seed = 2024
    # seed = 2025
    np.random.seed(seed)
    random.seed(seed)
    UNIFORM_ACTION_TRAJ = False
    DISJOINT_LEVEL_SET = False
    ############################### System Dynamic Configuration Parameters ###############################
    model_configs = {
        "DUBINS": {
            "model_name": "DUBINS",
            "dynamics": dynamics_dubins,
            "inverse_dynamics": inverse_dynamics_dubins, 
            "vectorized_dynamics": vectorized_dynamics_dubins,
            "perturbation_param": 2.01, # offsets add to representative at level set t
            "slack_parameter": 0.00, # offsets add to propagated samples at level set t+1
            # "actions": np.deg2rad(np.linspace(-45, 45, 45)),
            "vrange": (1.0, 1.0),               # Velocity range (min, max)
            "arange": (0.0, 0.0),               # acceleration range (min, max)
            "num_a": 1,
            "steering_angle_range": (-45, 45),  # Steering range in degrees
            "num_steering_angle": 5,
            "actions": None, # placeholder, will define later after defninig model_configs
            "thresholds": [0.05, 0.05, (2 * math.pi) / 40],
            "state_dim": 3,
            "v": 1,
            "dt" : 0.2,
            "total_t" : 2.01,
        },
        "2D_RANDOM_WALK": {
            "model_name": "2D_RANDOM_WALK",
            "dynamics": dynamics_2d_random_walk,
            "inverse_dynamics": inverse_dynamics_2d_random_walk, 
            "vectorized_dynamics": vectorized_dynamics_2D_Walk,
            "perturbation_param": 0.0, # 0 means no perturbation
            "slack_parameter": 0.0,    # 0 means no slack
            "vrange": (1.0, 1.0),      # Velocity range (min, max)
            "arange": (0.0, 0.0),      # acceleration range (min, max)
            "num_a": 1,
            "steering_angle_range": (-1, 1),  # Steering range in degrees
            "num_steering_angle": 101,
            "actions": None, # placeholder, will define later after defninig model_configs
            "thresholds": [0.02, 0.02],
            "state_dim": 2,
            "dt" : 1.0,
            "total_t" : 10.01,
        }
    }
    if MODEL not in model_configs:
        raise ValueError(f"Invalid MODEL type: {MODEL}. Please choose from {list(model_configs.keys())}.")

    config = model_configs[MODEL]
    config["actions"] = generate_actions(
        config["arange"], config["num_a"], config["steering_angle_range"], config["num_steering_angle"], MODEL!='2D_RANDOM_WALK'
    )
    actions = config["actions"]
    thresholds = config["thresholds"]
    t_step = config["dt"]
    total_t = config["total_t"] 
    slack = config["slack_parameter"]
    perturbation = config["perturbation_param"]
    vrange = config["vrange"]
    num_steering_angle = config["num_steering_angle"]

    # assume the robot always start from origin in the configuration space,
    # not necessarilly true for manipulator type of dynamics or higher dimensional systems
    initial_state = np.zeros(config["state_dim"]).astype(np.float32) 

    if MODEL in {"DUBINS", "KS_3D_STEERING_ANGLE", "DUBINS_4D"}:
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
    print(f"Actions: {actions}")
    print(f"Grid Resolution: {thresholds}")
    print(f"Time Step: {t_step}")
    print(f"Initial State: {initial_state}")
    print(f"Total Time Length: {total_t}")
    print(f"Perturbation parameter: {perturbation}")
    print(f"Slack parameter: {slack}")
    print(f"Velocity Range: {config.get('vrange', 'N/A')}")
    print(f"Acceleration Range: {config.get('arange', 'N/A')}")
    print(f"Number of Acceleration Values: {config.get('num_a', 'N/A')}")
    print(f"Steering Angle Range: {config.get('steering_angle_range', 'N/A')}")
    print(f"Number of Steering Angles: {config.get('num_steering_angle', 'N/A')}")

    g = Grid(thresholds=thresholds)
    # num_trajectories_list = [10000]
    num_trajectories_list = [250, 500, 1000, 2500, 5000, 10000]

    def generate_filename(
            prefix, disjoint_LS, MODEL, seed, grid_resolution_str,
            total_t, t_step,
            perturb, slack,
            vrange, num_steering_angle,
        ):
        return (
            f"{prefix}_"
            f"{disjoint_LS}"
            f"{MODEL}_"
            f"perturb_{perturb}_"
            f"slack_{slack}_"
            f"seed_{seed}_"
            f"grid_{grid_resolution_str}"
            f"t{total_t}_"
            f"ts{t_step}_"
            f"vrange_{vrange[0]}_{vrange[1]}_"
            f"steering_{num_steering_angle}.pkl"
        )
    disjoint_LS = "disjoint_" if DISJOINT_LEVEL_SET else "overlapping_"
    level_sets_filename = generate_filename(
        "C_Uniform_reachability_", disjoint_LS, MODEL, seed, grid_resolution_str, 
        total_t, t_step, perturbation, slack, vrange, num_steering_angle,

    )
    final_data_filename = generate_filename(
        "C_Uniform_processed", disjoint_LS, MODEL, seed, grid_resolution_str,
        total_t, t_step, perturbation, slack, vrange, num_steering_angle
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
    obstacles = None
    ################### Exploration Phase ###################
    ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtime = load_or_compute_reachable_set(
        level_sets_filename, config, DISJOINT_LEVEL_SET, obstacles
    )

    ################### Load or compute final processed graph data ###################
    pruned_graph_with_flow, action_prob_list, final_data_to_save, graph_runtime, max_flow_runtime = load_or_compute_final_data(
        final_data_filename, ReaBoxIndices_LSs, ReaBox_LSs, config,
        obstacles, collision_checker_node=collision_checker_node
    )
    plot_total_runtime_vs_horizon(level_set_runtime, graph_runtime, max_flow_runtime, dt=config["dt"])

    # Loop through each number of trajectories
    print("Sampling trajectories...")
    for num_trajectories in num_trajectories_list:
        traj_filename = generate_filename(
            f"C_Uniform_{num_trajectories}_trajectories", 
            disjoint_LS, MODEL, seed, grid_resolution_str, total_t,
            t_step, perturbation, slack, vrange, num_steering_angle
        )
        sampled_trajectories_graph = sample_trajectories_network_flow(pruned_graph_with_flow, num_trajectories, seed)
        visualize_trajectories(sampled_trajectories_graph, xy_only=True, obstacles=obstacles) # only x, y
        with open(traj_filename, 'wb') as file:
            pickle.dump(sampled_trajectories_graph, file, protocol=5)
        print(f"Sampled {num_trajectories} trajectories(graph) and saved to {traj_filename}.\n")

if __name__ == "__main__":
    main()