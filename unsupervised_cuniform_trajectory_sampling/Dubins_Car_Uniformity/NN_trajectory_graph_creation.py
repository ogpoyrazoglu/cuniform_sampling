from classes.graph_structure import *
import pickle 
from classes.grid import *
import numpy as np
import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time 
from Model import *

## Getting the config file
with open("/home/isleri/mahes092/C_Uniform/Datasets/C_Uniform_processed_disjoint_KS_3D_STEERING_ANGLE_perturb_2.01_slack_0.0_seed_2025_grid_0.03_0.03_6.00deg_t3.01_ts0.2vrange_1.0_1.0_steering_13.pkl", "rb") as f:
    data = pickle.load(f)

## Creating the trained instance of the model
model = torch.load("/home/isleri/mahes092/C_Uniform/Dubins_Car_Uniformity/Models/Kinematic_Unsupervised_All_Weight_Perturbation_Latest/best_model.pt")
model.eval()

## Creating the grid object of resolution
thresholds = data["config"]["thresholds"]
g = Grid(thresholds=thresholds)


def vectorized_dynamics_KS_3d_steering_angle(states, actions, dt=0.2, vrange=[1,1]): # constant velocity
    '''
    F1tenth dynamics update vectorized
    '''
    # Ensure inputs are in float32
    states = states.astype(np.float32)
    actions = actions.astype(np.float32)
    dt = np.float32(dt)
    vrange = np.array(vrange, dtype=np.float32)
    # L_wb = a + b, a and b is distance from spring mass center of gravity to front axle and rear axle respectively
    L_wb = 0.324
    n = states.shape[0]   # Number of states
    nu = actions.shape[0] # Number of actions
    states_repeated = np.repeat(states, nu, axis=0)  # Shape: (n * nu, 3)
    # Tile actions n times along axis 0 so that each state will execute every action
    actions_tiled = np.tile(actions, (n, 1))         # Shape: (n * nu, 1)
    x, y, theta = (
        states_repeated[:, 0],
        states_repeated[:, 1],
        states_repeated[:, 2],
    )
    steering_angles, accelerations = actions_tiled[:, 0], actions_tiled[:, 1] # acceleration should be 0 here
    assert np.all(np.abs(accelerations) < 1e-8), "Accelerations are not all near zero (threshold 1e-8)"
    assert vrange[0] == vrange[1], "For constant velocity, vrange[0] must equal vrange[1]"
    v = np.clip(accelerations, vrange[0], vrange[1])
    ## Updating X, Y
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    ## Updating theta
    theta_new = theta + (v / L_wb) * np.tan(steering_angles) * dt
    # Stack new states into an array with shape (n * nu, 3)
    new_states = np.stack((x_new, y_new, theta_new), axis=1).astype(np.float32)
    return new_states

## Getting the reachable index and representatives across level sets
ReaBox_LSs = data["reachable_representative_across_LS"]
ReaBoxIndices_LSs = data["reachable_indicex_across_LS"]

    
def compute_reachability_nn(ReaBox_LSs, ReaBoxIndices_LSs, grid, model, actions):
    ## Creating graph instances
    graphs = []
    id_offset = 0

    # Iterate over each level set
    for t in range(len(ReaBox_LSs) - 1):
        start_time = time.time()
        graph = {
            "nodes": {},
            "node_lookup": {},  # Secondary dictionary to map (level_set, point) to node_id
        }

        current_level_set = ReaBox_LSs[t]
        next_level_set = ReaBox_LSs[t+1]
        next_level_indices = set(ReaBoxIndices_LSs[t + 1])
        # Add nodes to the graph for current and next level sets
        for point in current_level_set:
            node = Node(t, point)
            node.id = node.id - id_offset
            graph["nodes"][node.id] = node
            graph["node_lookup"][(t, point)] = node.id  # Store (level_set, point) to node_id mapping

        for point in next_level_set:
            node = Node(t + 1, point)
            node.id = node.id - id_offset
            graph["nodes"][node.id] = node
            graph["node_lookup"][(t + 1, point)] = node.id  # Store (level_set, point) to node_id mapping

        id_offset += len(next_level_set)
        total_hits = 0
        total_misses = 0
        total_actions = 0

        maximum_slack_allowed = 2.01
        # Compute reachability by sampling actions
        for point in current_level_set:
            # Use the lookup dictionary to find the node_id, then access the node
            node1_id = graph["node_lookup"][(t, point)]
            node1 = graph["nodes"][node1_id]
            # Get the corresponding grid cell index and its samples # NOTE: 27 samples when dimension = 3
            points_to_sample = grid.perturb_state_deterministic_vectorized(points=np.array([point]), division_factor=maximum_slack_allowed)
            points_to_sample = np.vstack((points_to_sample, point)) 
            ## Getting Next states of the perturbed samples
            new_states = vectorized_dynamics_KS_3d_steering_angle(states=points_to_sample,
                                            actions=actions)
            ## Converting to model input
            perturbed_states_theta_x = torch.sin(torch.tensor(points_to_sample[:,2]))
            perturbed_states_theta_y = torch.cos(torch.tensor(points_to_sample[:,2]))
            network_current_state = torch.hstack((torch.tensor(points_to_sample[:,0:2]), perturbed_states_theta_x.unsqueeze(1), perturbed_states_theta_y.unsqueeze(1))).to(torch.float32).cuda()
            ## Getting action probabilities
            current_probability_distribution = model(network_current_state).detach().cpu().numpy()
            if t == 0:
                ## Uniform probability for 0th level set
                current_probability_distribution = np.ones(current_probability_distribution.shape)/13
            box_index_entire_LS = grid.get_index_vectorized(new_states) # type np.daary int64
            box_index_set = set(map(tuple, box_index_entire_LS))
            unique_arc_keys = set() # used to avoid adding duplicate keys
            ## Creating graph from the action probabilities
            for each_action_probability in range(points_to_sample.shape[0]):
                current_next_states = new_states[13*each_action_probability:13*each_action_probability + 13, :]
                current_next_states_index = box_index_entire_LS[13*each_action_probability:13*each_action_probability + 13]
                current_probabability = current_probability_distribution[each_action_probability]
                for each_action in range(actions.shape[0]):
                    total_actions += 1
                    ## Check if next state in the next level set
                    if tuple(current_next_states_index[each_action]) in next_level_indices:
                        total_hits += 1
                        new_center_point = grid.get_grid_center(current_next_states_index[each_action])
                        node2_id = graph["node_lookup"][(t + 1, tuple(new_center_point))]
                        node2 = graph["nodes"][node2_id]
                        arc_key = (node1.id, node2.id)
                        ## if graph arc exists, add it and have the weight as the probability
                        if arc_key not in unique_arc_keys:
                            unique_arc_keys.add(arc_key)
                            node1.outgoing_arcs.append((node2.id, current_probabability[each_action]))
                        ## If arc exists, update the weights by adding the probability
                        else:
                            indices = [i for i, arc in enumerate(node1.outgoing_arcs) if arc[0] == node2.id]
                            node1.outgoing_arcs[indices[0]] = tuple(list([node2.id, node1.outgoing_arcs[indices[0]][1] + current_probabability[each_action]]))
                    else:
                        ## If next state not in reachable level set
                        total_misses += 1
        miss_ratio = total_misses / total_actions if total_actions > 0 else 0
        # Print the number of hits, misses, and the miss ratio for the current level set, with deterministic perturbation should see 0 miss
        print(f"Level set {t} -> {t+1}: Hits = {total_hits}, Misses = {total_misses}, Miss Ratio = {miss_ratio:.2f}")
        graphs.append(graph)
        end_time = time.time() # Calculate elapsed time
        elapsed_time = end_time - start_time
        print(f"    Elapsed time: {elapsed_time:.6f} seconds")
    return graphs


def prune_graph(raw_graphs):
    """
    prunes the graph by removing dead nodes and invalid arcs until no further changes occur.
    
    Args:
        raw_graphs (list): List of raw graph dictionaries. A "raw graph" is an unprocessed graph.
        obstacles (list): List of obstacle definitions (e.g., rectangles or circles).
        collision_checker_node (callable): Function to check if a node collides with an obstacle.

    Returns:
        pruned_graphs (list): List of pruned graph dictionaries.

    High-level pruning logic:
    1. Backward pass: Prune nodes that either collide with obstacles or have no outgoing arcs.
       This ensures that no invalid transitions are considered in the final graph.
    2. Forward pass: Identify and prune nodes that have no incoming arcs, ensuring all nodes are reachable
       from the initial state in level set 0.

    Assumptions:
        Static obstacles
        No arcs span non-consecutive level sets (i to i + 2 or more).
    """
    total_start_time = time.time()
    pruned_graphs = [{} for _ in raw_graphs]
    dead_nodes = set() # track dead nodes
    total_removed_nodes = 0  # cumulative count of removed nodes
    global_incoming_arc_count = {}

    print("=" * 80)
    print("Pruning Stage: Backward and Forward Passes ")
    print("=" * 80)
    print("Backward Pass (Removing Nodes with No Outgoing Arcs or Collisions): ")
    # Iterate backward through the list of graphs, where each graph contain 2 level sets, representing transition
      # or "window" covering transitions between two level sets: t -> t+1
    for i in range(len(raw_graphs) - 1, -1, -1):  # Start from the last graph and move backward
        # The "operating window" here corresponds to level sets i and i+1
        # For example, when i = len(raw_graphs) - 1, the window operates on level sets 4 and 5
        start_time = time.time()
        current_graph = raw_graphs[i]
        incoming_arc_count = {node_id: 0 for node_id in current_graph["nodes"] if current_graph["nodes"][node_id].level_set == i+1}
        pruned_graph = {"nodes": {}, "node_lookup": current_graph["node_lookup"]}
        dead_nodes_size_before = len(dead_nodes)

        # Step 2: Prune arcs lead to dead nodes and nodes with no valid outgoing arcs
        for node_id, node in current_graph["nodes"].items(): 
            if node_id in dead_nodes:  # Skip dead nodes
                continue

            node.outgoing_arcs = [ # remove arcs leading to dead nodes
                arc for arc in node.outgoing_arcs if arc[0] not in dead_nodes
            ]
            if node.level_set == i and len(node.outgoing_arcs) == 0: # mark dead if node at L_t has no outgoing arcs
                dead_nodes.add(node_id)
                continue 
            # Update incoming arc count for the target nodes
            for arc in node.outgoing_arcs:
                target_node_id = arc[0]
                incoming_arc_count[target_node_id] += 1
            pruned_graph["nodes"][node_id] = node # add to pruned graph if it still has valid outgoing arcs

        global_incoming_arc_count.update(incoming_arc_count)
        pruned_graphs[i] = pruned_graph
        dead_nodes_size_after = len(dead_nodes)
        nodes_removed = dead_nodes_size_after - dead_nodes_size_before  # Nodes removed in this iteration
        total_removed_nodes += nodes_removed
        elapsed_time = time.time() - start_time # elapsed time for this level set
        nodes_in_level_set_i = sum(1 for node in pruned_graph["nodes"].values() if node.level_set == i)
        print(f"  Pruned transition Level set {i:>2} -> {i+1:>2}: removed: {nodes_removed:<5} | time: {elapsed_time:>7.4f}s | nodes in {i:<2}: {nodes_in_level_set_i:<5}")

    print("\nForward Pass (Removing Nodes with No Incoming Arcs):")
    for i in range(len(raw_graphs)):
        current_graph = pruned_graphs[i]
        start_time = time.time()
        dead_nodes_size_before = len(dead_nodes)

        # Remove nodes with no incoming arcs
        for node_id, node in list(current_graph["nodes"].items()):
            if node.level_set == 0: # skip initial level set because they don't have incoming arcs
                continue
            if global_incoming_arc_count.get(node_id, 0) == 0:
                for arc in node.outgoing_arcs:
                    target_node_id = arc[0]
                    if target_node_id in global_incoming_arc_count:
                        global_incoming_arc_count[target_node_id] -= 1
                del current_graph["nodes"][node_id]
                dead_nodes.add(node_id)

        dead_nodes_size_after = len(dead_nodes)  # Capture size after pruning
        nodes_removed = dead_nodes_size_after - dead_nodes_size_before
        total_removed_nodes += nodes_removed
        elapsed_time = time.time() - start_time
        print(f"  Forward pass Level set {i:>2} -> {i+1:>2}: removed: {nodes_removed:<5} | time: {elapsed_time:>7.4f}s")

    total_elapsed_time = time.time() - total_start_time # total elapsed time
    print("Summary:")
    print(f"  Total nodes removed (all passes): {total_removed_nodes}")
    #TODO: also output total removed arcs
    print(f"  Total pruning time: {total_elapsed_time:.4f}s\n")
    return pruned_graphs 

## Creating graph for trajectory sampling
graphs = compute_reachability_nn(ReaBox_LSs, ReaBoxIndices_LSs, g, model, data["config"]["actions"])
data_to_save = {
            'grid' : g,
            'v' : 1.0,
            'actions' : data["config"]["actions"],
            'total_t' : 3.01,
            't_step' : 0.20,
            'graphs' : graphs,
            'action_prob_list' : None,
            'reachable_indicex_across_LS': ReaBoxIndices_LSs,
            'reachable_representative_across_LS': ReaBox_LSs,
        }
    
## Pruing the graph for removing empty nodes
pruned_graph = prune_graph(data_to_save["graphs"])
data_to_save["graphs"] = pruned_graph
## Saving the graph
with open("NN_Graph.pkl", 'wb') as file:
    pickle.dump(data_to_save, file)