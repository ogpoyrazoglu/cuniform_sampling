'''
This file contains the utility helper functions for C-Uniform sampling codebase, functionalities below:
    1) Trajectory generation 
    2) Visualization
    3) Analysis code
'''
import matplotlib.pyplot as plt
import os
from classes.grid import Grid
from classes.graph_structure import Node
import numpy as np
from ortools.graph.python import max_flow
import pickle
import time
import random
from pprint import pprint

################################################ Trajectory Generation Code ################################################
def sample_trajectories_network_flow(graphs, num_trajectories, seed):
    '''
    Directly use the network flow result to get trajectories, each graph consist of the following attributes
        graph = {
            "nodes": {},
            "node_lookup": {},  # Secondary dictionary to map (level_set, point) to node_id
        }
    Args:
        graphs (list): List of graph dictionaries for each level set.
        num_trajectories (int): Number of trajectories to sample.
        seed (int): Random seed for reproducibility.
    
    Returns:
        list: List of sampled trajectories, where each trajectory is a list of points [(point, action), ...].
        actions are None because they are unknown
    '''
    np.random.seed(seed)
    traj_len = len(graphs)
    sampled_trajectories = []

    # Build a lookup dictionary that maps node_id to 'point'
    node_to_point = {}
    for graph in graphs:
        for node_id, node in graph["nodes"].items():
            node_to_point[node_id] = node.point
    
    for _ in range(num_trajectories):
        trajectory = []
        current_node = None

        for level in range(traj_len):
            if level == 0:
                initial_nodes = [node for node in graphs[level]["nodes"].values() if node.level_set == 0]
                assert len(initial_nodes) == 1, "There should be a single initial node in level set 0."
                current_node = initial_nodes[0]
                next_point = current_node.point
                trajectory.append((next_point, None))
            # Extract outgoing node ids and flows
            outgoing_arcs = [(to_node_id, flow) for to_node_id, flow in current_node.outgoing_arcs]
            assert outgoing_arcs, f"No outgoing arcs found at level {level} for node {current_node.id}."

            to_node_ids = [to_node_id for to_node_id, flow in outgoing_arcs]
            flows = np.array([flow for _, flow in outgoing_arcs])

            if flows.sum() == 0: # If there is no flow, pick a random connected node at random
                # print(f"No flow detected at level {level} for state: {current_node.point}. Using uniform probabilities.")
                probabilities = np.ones(len(to_node_ids)) / len(to_node_ids)
            else:
                probabilities = flows / flows.sum() # normalize flows to get probabilities

            to_node_id = np.random.choice(to_node_ids, p=probabilities) # pick the next node based on the probabilities

            # Lookup the 'point' of the chosen node and append to the trajectory
            next_point = node_to_point[to_node_id]
            trajectory.append((next_point, None))
            level_set_index = level + 1 if level + 1 < traj_len else level
            current_node = graphs[level_set_index]["nodes"][to_node_id] # move to the next node
        sampled_trajectories.append(trajectory)
    return sampled_trajectories

def sample_trajectories_using_action_probability(
        grid, initial_state, action_probabilities_list, actions, dynamics,
        t_step, num_trajectories, trajectory_length, v, perturbation=False
    ):
    """
    Sample multiple trajectories, using different action probabilities for each time step.

    Args:
        grid (Grid): The grid object used to discretize the state space.
        initial_state (tuple): The starting state of the system (x, y, theta).
        action_probabilities_list (list): List of dictionaries mapping grid indices to action probability distributions for each time step.
        actions (list): List of discretized actions available for the system.
        dynamics (function): The function defining the robot's dynamics.
        t_step (float): The time step for the robot dynamics.
        num_trajectories (int): The number of trajectories to generate.
        trajectory_length (int): The maximum number of steps for each trajectory.
        perturbation(bool): if perturbation set to True, the current state will be perturbed randomly within the cell

    Returns:
        list: A list of sampled trajectories, where each trajectory is a list of (state, action) tuples.
    """
    all_trajectories = []  # To store all generated trajectories

    for _ in range(num_trajectories):
        trajectory = []  # Initialize an empty list for the current trajectory
        current_state = initial_state # Start from the initial state
        if perturbation:
            current_state_perturbation = grid.perturb_state_deterministic(current_state, division_factor=2.01)
            random_row_index = np.random.randint(0, current_state_perturbation.shape[0])
            current_state = current_state_perturbation[random_row_index]

        for step in range(trajectory_length):
            action_probabilities = action_probabilities_list[step]
            grid_index = grid.get_index(current_state)
            # Lookup the action probabilities for the current grid index
            if grid_index not in action_probabilities:
                # If no probabilities are available for this grid index, print an error and terminate the trajectory
                print(f"Error: No action probabilities available for grid index {grid_index} at time step {step}.")
                break
            action_pdf = action_probabilities[grid_index]
            # Select an action based on the probability distribution
            chosen_action_index = np.random.choice(range(len(actions)), p=action_pdf)
            chosen_action = actions[chosen_action_index]

            trajectory.append((current_state, chosen_action))
            new_state = dynamics(current_state, chosen_action, t_step, v)
            new_corresponding_center = grid.get_grid_center(grid.get_index(new_state))

            current_state = new_corresponding_center
            # perturb the points
            if perturbation:
                current_state_perturbation = grid.perturb_state_deterministic(current_state, division_factor=2.01)
                random_row_index = np.random.randint(0, current_state_perturbation.shape[0])
                current_state = current_state_perturbation[random_row_index]
        trajectory.append((current_state, None))
        all_trajectories.append(trajectory)
    return all_trajectories

################################################ Visualization Code Below #################################################
def plot_total_runtime_vs_horizon(level_set_runtime, graph_runtime, nf_runtime, dt, filename=None):
    """
    Plots the sum of two runtime lists as a function of the level-set index (i.e., planning horizon).
    """
    if len(graph_runtime) != len(nf_runtime):
        print("Length of runtimes is not equal! Exiting plot_total_runtime_vs_horizon()...") 
        return
    length = len(graph_runtime)
    if length < 1:
        print("No runtime analysis")
        return
    level_set_runtime = [0.0] + level_set_runtime
    graph_runtime = [0.0] + graph_runtime
    nf_runtime = [0.0] + nf_runtime
    
    # Initialize cumulative runtime list
    total_times = []
    cumulative_sum = 0.0  # Initialize cumulative sum to zero

    for i in range(length):
        # Add the current graph runtime and nf runtime to the cumulative sum
        cumulative_sum += graph_runtime[i] + nf_runtime[i]
        # Append the cumulative sum to the total_times list
        total_times.append(cumulative_sum)
    horizons = [i * dt for i in range(len(total_times))]

    if filename is None:
        max_horizon = horizons[-1]
        filename = f"runtime_plot_dt_{dt}_max_horizon_{max_horizon:.2f}.pdf"

    plt.figure(figsize=(8, 5))
    plt.plot(horizons, total_times, marker='o', label="Runtime")
    plt.xlabel("Planning Horizon (seconds)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Flow-based C-Uniform Runtime vs. Planning Horizon")
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(os.getcwd(), filename)
    plt.savefig(save_path, format=save_path.split('.')[-1], bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def visualize_reachable_cells(filename, lt=None, visualize=True):
    """
    Visualize reachable representatives at level sets from `lt` to the last level set. If `lt` is not specified, start from 0.
    
    Args:
        filename (str): Path to a pickle file containing a dictionary with a key 'reachable_representative_across_LS'.
        lt (int, optional): The starting level set index. Defaults to None (start from 0).
    """
    print("Starting the visualization process...")
    
    # Step 1: Check if the file is a pickle file
    if not filename.endswith('.pkl'):
        raise ValueError(f"Expected a pickle file (.pkl), but got: {filename}")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    print(f"Reading data from the pickle file: {filename}")

    # Step 2: Read from the pickle file and type-check
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("The file content should be a dictionary.")
    if 'reachable_representative_across_LS' not in data:
        raise KeyError("'reachable_representative_across_LS' key is missing in the dictionary.")
    print("Successfully read and validated the pickle file.")
    reachable_representatives = data['reachable_representative_across_LS']
    if not isinstance(reachable_representatives, list):
        raise ValueError("'reachable_representative_across_LS' should be a list of sets.")

    # Step 3: Determine starting level set
    if lt is None:
        lt = 0
    print(f"Starting visualization from level set {lt}...")

    # Step 4: Loop through level sets until the last one
    while lt < len(reachable_representatives) - 1:
        print(f"Extracting reachable representatives for level sets {lt} and {lt+1}...")

        reachable_lt = reachable_representatives[lt]
        reachable_lt_plus_1 = reachable_representatives[lt + 1]
        if not (isinstance(reachable_lt, set) and isinstance(reachable_lt_plus_1, set)):
            raise ValueError("Each level set should contain a set of reachable representatives.")

        # Find overlapping representatives
        overlapping_representatives = reachable_lt.intersection(reachable_lt_plus_1)
        print(f"    Identified {len(reachable_lt)} representatives at level set {lt}.")
        print(f"    Identified {len(reachable_lt_plus_1)} representatives at level set {lt+1}.")
        print(f"    Found {len(overlapping_representatives)} overlapping representatives.")

        if visualize:
            # Visualize using matplotlib
            print("Drawing the visualization...")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot representatives for lt
            for cell in reachable_lt:
                x, y, theta = cell  # Extract x, y, and theta
                ax.scatter(x, y, theta, color='blue', alpha=0.3)

            # Plot representatives for lt+1
            for cell in reachable_lt_plus_1:
                x, y, theta = cell  # Extract x, y, and theta
                ax.scatter(x, y, theta, color='green', alpha=0.3)

            # Highlight overlapping representatives
            for cell in overlapping_representatives:
                x, y, theta = cell  # Extract x, y, and theta
                ax.scatter(x, y, theta, color='red', alpha=0.5)

            # Set axis labels
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Theta (Orientation)')
            ax.set_title(f'Reachable Cells Visualization at Level Sets {lt} and {lt+1}')

            # Add a global legend
            blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Level Set lt', markerfacecolor='blue', markersize=10, alpha=0.3)
            green_patch = plt.Line2D([0], [0], marker='o', color='w', label='Level Set lt+1', markerfacecolor='green', markersize=10, alpha=0.3)
            red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Overlap', markerfacecolor='red', markersize=10, alpha=0.5)
            ax.legend(handles=[blue_patch, green_patch, red_patch], loc='upper right')

            plt.show()
            print(f"Visualization for level sets {lt} and {lt+1} completed successfully.")
        # Move to the next level set
        lt += 1


def visualize_trajectories(all_trajectories, xy_only=True, obstacles=None):
    """
    Args: all_trajectories (np.ndarray): A NumPy array containing all trajectories.
    """
    if xy_only:
        plt.figure(figsize=(10, 6))
        num_trajectories = len(all_trajectories)

        for i in range(num_trajectories): # Loop through each trajectory
            trajectory = all_trajectories[i]

            x_coords = []
            y_coords = []
            for entry in trajectory: # Extract x, y coordinates for each state in the trajectory
                if isinstance(entry, tuple) and len(entry) == 2:
                    state, action = entry  # unpack state and action
                    x, y = state[0], state[1]  # Extract x and y coordinates
                    x_coords.append(x)
                    y_coords.append(y)
                else:
                    print(f"Unexpected format in trajectory {i+1}: {entry}")
                    continue
            plt.plot(x_coords, y_coords, marker='o', linewidth=0.9, markersize=4)

        if obstacles: # plot obstacles if provided
            for rect in obstacles: #NOTE: right now assume all obstacles are rectangular
                if len(rect) == 4:  # Ensure the rectangle is defined properly
                    x_min, y_min, x_max, y_max = rect
                    rect_patch = plt.Rectangle(
                        (x_min, y_min),  # Bottom-left corner
                        x_max - x_min,  # Width
                        y_max - y_min,  # Height
                        color='red',
                        alpha=0.5,
                        label='Obstacle'
                    )
                    plt.gca().add_patch(rect_patch)
                else:
                    print(f"Invalid rectangle format: {rect}")

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axis('equal')
        plt.title(f'Trajectories Visualization ({num_trajectories} Trajectories)')
        plt.grid(True)
        plt.show()
    else: # visualize in 3d space
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        num_trajectories = len(all_trajectories)
        for i in range(num_trajectories):  # Loop through each trajectory
            trajectory = all_trajectories[i]
            x_coords = []
            y_coords = []
            theta_coords = [] 
            for entry in trajectory:  # Extract x, y, theta coordinates for each state in the trajectory
                if isinstance(entry, tuple) and len(entry) == 2:
                    state, action = entry  # unpack state and action
                    x, y, theta = state[0], state[1], state[2]  # Extract x, y, and z (or theta) coordinates
                    x_coords.append(x)
                    y_coords.append(y)
                    theta_coords.append(theta)
                else:
                    print(f"Unexpected format in trajectory {i+1}: {entry}")
                    continue
            ax.plot(x_coords, y_coords, theta_coords, marker='o', linewidth=0.9, markersize=4)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Theta Position')
        ax.set_title(f'3D Trajectories Visualization ({num_trajectories} Trajectories)')
        ax.grid(True)
        plt.show()

def visualize_graph_connection(graphs, obstacles=None, differentiate_occurrence=False, label_nodes=False, draw_2D=True):
    """
    Visualizes the connections in a list of graphs, optionally displaying obstacles.

    Args:
        graphs (list): List of graph dictionaries, where each graph contains nodes and their arcs.
        obstacles (list, optional): List of obstacles, where each obstacle is defined as a rectangle [x_min, y_min, x_max, y_max].
        differentiate_occurrence (bool, optional): If True, use different colors based on node appearance count. Default is True.
        label_nodes (bool, optional): If True, label each node with its ID. Default is False.
        draw_2D: if True, only visualize the graph connection in 2D
    """
    print("Start visualizing graph connections...")
    assert draw_2D, "the 3D graph connenction is not yet implemented"
    print("Step 1: Initialized figure and axes.")
    plt.figure(figsize=(10, 10)) # Initialize a figure for visualization
    ax = plt.gca()

    # Track node appearances and arcs
    node_appearance_count = {}  # Tracks how many times a node appears (by ID)
    all_nodes = {}              # Stores unique nodes by their id
    all_arcs = []               # Stores arcs as (from_node_id, to_node_id)

    print("Step 2: Processing graphs...")
    for graph in graphs:
        print("    Processing a graph with {} nodes.".format(len(graph["nodes"])))
        for node_id, node in graph["nodes"].items():
            # Count node appearances
            if node_id not in node_appearance_count:
                node_appearance_count[node_id] = 0
            node_appearance_count[node_id] += 1
            # Add unique nodes to all_nodes dictionary
            if node_id not in all_nodes:
                all_nodes[node_id] = node
            # Add arcs to the all_arcs list
            for arc in node.outgoing_arcs:
                to_node_id = arc[0]
                if to_node_id in graph["nodes"]:
                    all_arcs.append((node_id, to_node_id))
    print("Finished processing graphs: {} unique nodes, {} arcs.".format(len(all_nodes), len(all_arcs)))

    print("Step 3: Aggregating nodes based on (x,y) coordinates...")
    # Aggregate nodes based on (x,y) coordinates regardless of additional state dimensions.
    xy_nodes = {}     # Key: (x, y), Value: dict with 'occurrence' and a set of node IDs (for labeling)
    node_to_xy = {}   # Mapping from node id to its (x, y) coordinate
    for node_id, node in all_nodes.items():
        x, y = node.point[0], node.point[1]  # Use only x and y regardless of extra dimensions.
        coord = (x, y)
        node_to_xy[node_id] = coord
        if coord not in xy_nodes:
            xy_nodes[coord] = {"occurrence": 0, "labels": set()}
        xy_nodes[coord]["occurrence"] += 1
        xy_nodes[coord]["labels"].add(node_id)
    print("Aggregated {} unique (x,y) coordinates.".format(len(xy_nodes)))
    
    print("Step 4: Deduplicating arcs using (x,y) projection...")
    # Deduplicate arcs by converting node IDs to their (x, y) projection.
    arcs_set = set()
    for from_node_id, to_node_id in all_arcs:
        if from_node_id in node_to_xy and to_node_id in node_to_xy:
            arcs_set.add((node_to_xy[from_node_id], node_to_xy[to_node_id]))
    print("Deduplicated arcs count: {}.".format(len(arcs_set)))

    print("Step 5: Plotting nodes...")
    # Plot nodes (aggregated by (x, y)) with optional color differentiation.
    for coord, data in xy_nodes.items():
        x, y = coord
        occ = data["occurrence"]
        if differentiate_occurrence:
            if occ == 1:
                color = 'lightblue'
                label = 'Node (Single Occurrence)'
            elif occ == 2:
                color = 'darkblue'
                label = 'Node (Double Occurrence)'
            else:
                color = 'purple'
                label = 'Node (Multiple Occurrences)'
            # Avoid duplicate legend entries.
            existing_labels = ax.get_legend_handles_labels()[1]
            if label in existing_labels:
                label = None
            ax.scatter(x, y, color=color, zorder=100, s=20, label=label)
        else:
            label = 'Nodes'
            existing_labels = ax.get_legend_handles_labels()[1]
            if label in existing_labels:
                label = None
            ax.scatter(x, y, color='blue', zorder=100, s=20, label=label)
        
        if label_nodes:
            # If multiple node IDs share the same coordinate, join them with commas.
            ax.text(x, y + 0.02, ",".join(str(nid) for nid in data["labels"]),
                    fontsize=7, ha='center', color='black', zorder=101)
    print("Finished plotting nodes.")

    # skipped plotting arcs since there are too many arcs   
    # # Plot arcs based on the deduplicated (x, y) coordinates.
    # print("Step 6: Plotting arcs...")
    # for arc in arcs_set:
    #     (x_start, y_start), (x_end, y_end) = arc
    #     label = 'Arcs'
    #     existing_labels = ax.get_legend_handles_labels()[1]
    #     if label in existing_labels:
    #         label = None
    #     ax.plot([x_start, x_end], [y_start, y_end], color='orange', linewidth=1, alpha=0.8, label=label)
    # print("Finished plotting arcs.")

    print("Step 7: Plotting obstacles...")
    if obstacles: # plot obstacles if provided
        for rect in obstacles:  # Assume all obstacles are rectangular for now
            if len(rect) == 4:
                x_min, y_min, x_max, y_max = rect
                rect_patch = plt.Rectangle(
                    (x_min, y_min),  # Bottom-left corner
                    x_max - x_min,  # Width
                    y_max - y_min,  # Height
                    color='red',
                    alpha=0.5,
                    label='Obstacle' if 'Obstacle' not in plt.gca().get_legend_handles_labels()[1] else ""
                )
                plt.gca().add_patch(rect_patch)
            else:
                print(f"Invalid rectangle format: {rect}")
    print("Finished plotting obstacles.")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Graph Connections Visualization")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    print("Visualization complete. Displaying plot.")
    plt.show()

def visualize_graph_flow_and_low_arcs_nodes(
    graph,
    show_flow=True, show_low_arcs_nodes=False, sample_for_visualization=True,
    sample_size=300,
):
    """
    Visualize the flow from level set t to t+1 in a 3D plot.
    Args:
        graph (dict): The graph structure containing nodes and edges for a single transition.
        show_flow (bool): Whether to visualize flow-related elements.
        show_low_arcs_nodes (bool): Whether to visualize nodes with low outgoing arcs and their arcs.
        sample_for_visualization (bool): Whether to use random sampling for unhighlighted nodes and arcs.
        sample_size (int): Number of nodes to randomly sample if sampling is enabled.

    Highlights:
        - Nodes in level set t:
          - Yellow: Imperfect outgoing flows (less than ideal).
          - Light coral: Perfect outgoing flows (equal to ideal).
        - Nodes in level set t+1:
          - Green: Imperfect incoming flows (less than ideal).
          - Blue: Perfect incoming flows (equal to ideal).
        - Arcs:
          - Green: Connect to nodes in level set t+1 with imperfect incoming flows.
          - Yellow: Originate from nodes in level set t with imperfect outgoing flows.
          - Blue: Default arcs for flows not highlighted.
    """
    # Initialize 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    current_level_set = graph["nodes"][list(graph["nodes"].keys())[0]].level_set
    next_level_set = current_level_set + 1

    num_nodes_t = len([node for node in graph["nodes"].values() if node.level_set == current_level_set])
    num_nodes_t1 = len([node for node in graph["nodes"].values() if node.level_set == next_level_set])

    # Check that only one of the flags is enabled
    assert not (show_low_arcs_nodes and show_flow), (
        "Only one visualization mode can be active at a time: "
        "'show_low_arcs_nodes' or 'show_flow'. Set one to True and the other to False."
        "Overlap functionality may be implemented in the future."
    )
    if show_flow:
        print("    Visualizing flow...")
        ideal_max_flow_capacity = num_nodes_t1* num_nodes_t / num_nodes_t1
        ideal_outgoing_flow = num_nodes_t1 # the ideal sum of outgoing flow for a node a level set t,
        # Check outgoing flow for nodes in level set t
        outgoing_flow = {node.id: 0 for node in graph["nodes"].values() if node.level_set == current_level_set}
        for node in graph["nodes"].values():
            for to_node_id, flow in node.outgoing_arcs:
                outgoing_flow[node.id] += flow

        highlight_outgoing = {}
        for key, value in outgoing_flow.items():
            # Highlight nodes based on outgoing flow from level set t
            highlight_outgoing[key] = True if outgoing_flow[key] < ideal_outgoing_flow else False

        # Check incoming flow for nodes in the next level set
        incoming_flow = {node.id: 0 for node in graph["nodes"].values() if node.level_set == next_level_set}
        for node in graph["nodes"].values():
            for to_node_id, flow in node.outgoing_arcs:
                if to_node_id in incoming_flow:
                    incoming_flow[to_node_id] += flow
        highlight_incoming = {}
        for key, value in incoming_flow.items():
            highlight_incoming[key] = True if incoming_flow[key] < ideal_max_flow_capacity else False
        
        # Random sampling for unhighlighted nodes in current and next level sets
        unhighlighted_nodes_t = [
            node for node in graph["nodes"].values() if node.level_set == current_level_set and not highlight_outgoing[node.id]
        ]
        unhighlighted_nodes_t1 = [
            node for node in graph["nodes"].values() if node.level_set == next_level_set and not highlight_incoming[node.id]
        ]

        if sample_for_visualization:
            sampled_nodes_t = random.sample(
                unhighlighted_nodes_t, min(len(unhighlighted_nodes_t), sample_size)
            )
            sampled_nodes_t1 = random.sample(
                unhighlighted_nodes_t1, min(len(unhighlighted_nodes_t1), sample_size)
            )
            sampled_node_ids_t = {node.id for node in sampled_nodes_t}
            sampled_node_ids_t1 = {node.id for node in sampled_nodes_t1}
        else:
            sampled_node_ids_t = {node.id for node in unhighlighted_nodes_t}
            sampled_node_ids_t1 = {node.id for node in unhighlighted_nodes_t1}

        # Plot nodes
        for node in graph["nodes"].values():
            x, y, theta = node.point
            if node.level_set == current_level_set:
                if highlight_outgoing[node.id]:
                    ax.scatter(x, y, theta, color='yellow', alpha=0.7, s=20)
                elif node.id in sampled_node_ids_t:
                    ax.scatter(x, y, theta, color='lightcoral', alpha=0.2, s=10)
            elif node.level_set == next_level_set:
                if highlight_incoming[node.id]:
                    ax.scatter(x, y, theta, color='g', alpha=0.8, s=20)
                    # ax.text(x, y, theta, str(node.id), size=10, zorder=10000) # label the node with its ID, only label if node is 'highlight'
                elif node.id in sampled_node_ids_t1:
                    ax.scatter(x, y, theta, color='b', alpha=0.3, s=10)

            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Level Set t (Imperfect Outgoing Flow)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Level Set t (Perfect Outgoing Flow)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Level Set t+1 (Imperfect Incoming Flow)'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Level Set t+1 (Perfect Incoming Flow)'),
                # plt.Line2D([0], [0], color='g', lw=2, label='Highlighted Arcs Connect to Imperfect Incoming Flow Node'),
                # plt.Line2D([0], [0], color='yellow', lw=2, label='Highlighted Arcs Connect to Imperfect Outgoing Flow Node'),
            ]
            ax.legend(handles=legend_handles, loc='upper right')

        ''' # ignore arcs visualization
        # Plot outgoing arcs using the node's outgoing_arcs attribute
        for to_node_id, flow in node.outgoing_arcs:
            if sample_for_visualization:
                # Skip unhighlighted arcs not in the sampled nodes
                if node.level_set == current_level_set and node.id not in sampled_node_ids_t:
                    continue
                if node.level_set == next_level_set and to_node_id not in sampled_node_ids_t1:
                    continue

            end_node = graph["nodes"].get(to_node_id) 
            if end_node is None:
                print(f"end_node is none for one arc of node {node.id}")
                continue

            # Get the coordinates for the start and end nodes
            x_start, y_start, theta_start = node.point
            x_end, y_end, theta_end = end_node.point

            if highlight_incoming[to_node_id]:
                assert not highlight_outgoing[node.id], "SOMETHING WRONG IN NETWORK SOLUTION, imperfect outgoing flow nodes should have flow distributed to imperfect incoming flow nodes"
                alpha = 0.3
                color = 'g'
                # ax.text(x_mid, y_mid, theta_mid, f"{flow:.2f}", color='black', size=6, zorder=10000)
                ax.plot([x_start, x_end], [y_start, y_end], [theta_start, theta_end], color=color, alpha=alpha) 
            elif highlight_outgoing[node.id]:
                alpha = 0.2
                color = 'yellow'
                ax.plot([x_start, x_end], [y_start, y_end], [theta_start, theta_end], color=color, alpha=alpha) 
                # ax.text(x_mid, y_mid, theta_mid, f"{flow:.2f}", color='black', size=6, zorder=10000)
            # else: # skip drawing the "normal" arcs
            #     alpha = 0.015
            #     color = 'b'
            # ax.plot([x_start, x_end], [y_start, y_end], [theta_start, theta_end], color=color, alpha=alpha) # draw a line between the nodes in 3D
        '''
    if show_low_arcs_nodes:
        # Extract the number of outgoing arcs for nodes in the current level set
        outgoing_arcs_counts = [
            len(node.outgoing_arcs) for node in graph["nodes"].values() if node.level_set == current_level_set
        ]
        
        # Filter positive counts for statistical calculations
        positive_outgoing_arcs = [count for count in outgoing_arcs_counts if count > 0]

        if positive_outgoing_arcs:  # Ensure there are positive counts to avoid errors
            min_outgoing_arcs = min(positive_outgoing_arcs)
            max_outgoing_arcs = max(positive_outgoing_arcs)
            mean_outgoing_arcs = np.mean(positive_outgoing_arcs)
            median_outgoing_arcs = np.median(positive_outgoing_arcs)
        else:
            min_outgoing_arcs = max_outgoing_arcs = mean_outgoing_arcs = median_outgoing_arcs = 0

        if current_level_set == 0:
            print(f"  Highlighting nodes with 0 outgoing arcs (red) and nodes with the minimum positive outgoing arcs ({min_outgoing_arcs}, orange). Randomly sampling additional nodes for context.")
        print(f"    Level set {current_level_set}:")
        print(f"        Min outgoing arcs: {min_outgoing_arcs}")
        print(f"        Max outgoing arcs: {max_outgoing_arcs}")
        print(f"        Mean outgoing arcs: {mean_outgoing_arcs:.2f}")
        print(f"        Median outgoing arcs: {median_outgoing_arcs}")

        # Nodes in level set t and t+1
        nodes_t = [node for node in graph["nodes"].values() if node.level_set == current_level_set]
        nodes_t1 = [node for node in graph["nodes"].values() if node.level_set == next_level_set]

        # Identify nodes with low outgoing arcs in current level set
        low_outgoing_nodes_t = [
            node for node in nodes_t if len(node.outgoing_arcs) <= min_outgoing_arcs 
        ]

        # Randomly sample context nodes in current level set if sampling is enabled
        if sample_for_visualization:
            sampled_context_nodes_t = random.sample(
                [node for node in nodes_t if node not in low_outgoing_nodes_t],
                min(sample_size, len(nodes_t) - len(low_outgoing_nodes_t))
            )
        else:
            sampled_context_nodes_t = [node for node in nodes_t if node not in low_outgoing_nodes_t]

        # Randomly sample nodes in the next level set for context
        if sample_for_visualization:
            sampled_context_nodes_t1 = random.sample(
                nodes_t1, min(sample_size, len(nodes_t1))
            )
        else:
            sampled_context_nodes_t1 = nodes_t1

        # Visualize nodes in the current level set
        for node in low_outgoing_nodes_t:
            x, y, theta = node.point
            if len(node.outgoing_arcs) == 0:  # Nodes with no outgoing arcs
                ax.scatter(x, y, theta, color='red', alpha=0.9, s=40, label="No outgoing arcs")
            else:  # Nodes with few outgoing arcs
                ax.scatter(x, y, theta, color='orange', alpha=0.8, s=30, label="Few outgoing arcs")
        
        # Visualize outgoing arcs for nodes with low outgoing arcs
        for node in low_outgoing_nodes_t:
            x_start, y_start, theta_start = node.point
            for to_node_id, flow in node.outgoing_arcs:
                end_node = graph["nodes"].get(to_node_id)
                if end_node is None:
                    continue
                x_end, y_end, theta_end = end_node.point
                ax.plot(
                    [x_start, x_end], [y_start, y_end], [theta_start, theta_end],
                    color='orange', alpha=0.8, linewidth=1, label="Outgoing arc"
                )

        for node in sampled_context_nodes_t:
            x, y, theta = node.point
            ax.scatter(x, y, theta, color='lightgray', alpha=0.4, s=10, label="Sampled context (t)")

        # Visualize sampled nodes in the next level set
        for node in sampled_context_nodes_t1:
            x, y, theta = node.point
            ax.scatter(x, y, theta, color='blue', alpha=0.5, s=20, label="Sampled context (t+1)")

        # Add legend
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='No outgoing arcs'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=f'Few outgoing arcs (≤ {min_outgoing_arcs})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Sampled context (t)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Sampled context (t+1)'),
        ]
        ax.legend(handles=legend_handles, loc='upper right')

        # Add custom legend
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Dead Nodes (0 Outgoing Arcs)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=f'Nodes with {min_outgoing_arcs} Outgoing Arc(s)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Level set t (sampled)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Level set t+1 (sampled)'),
        ]
        ax.legend(handles=legend_handles, loc='upper right')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Theta')

    # Set equal scaling for all axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Find the maximum range for equal scaling
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    # Calculate the midpoints
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    # Set equal range for all axes
    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])

    title_parts = [] # descriptive title
    if show_flow:
        title_parts.append(f"Flow visualization (Level set {current_level_set} to {next_level_set})")
    if show_low_arcs_nodes:
        title_parts.append(f"Low-outgoing-arcs nodes (≤ {min_outgoing_arcs}) highlighted")
    if sample_for_visualization:
        title_parts.append(f"Sample size: {sample_size} (context)")
    else:
        title_parts.append("All context nodes shown")
    title = " | ".join(title_parts) # combine all parts into a concise title
    plt.title(title)
    # plt.title(f'3D Visualization of Flow from Level Set {current_level_set} to {next_level_set}')
    plt.show()

############################################### Helper Functions Below ###############################################
def setup_single_transition_flow(graph):
    """
    Set up and solve the network flow for a single graph transition(from level set t to t+1).
    Args:
        graph (dict): The graph structure containing nodes for a single transition.
    Returns:
        return the ratio between all incoming flow and outgoing flow
    """
    # Get the number of nodes in current level set (t) and next level set (t+1)
    start_time = time.time()
    num_nodes_t = len([node for node in graph["nodes"].values() if node.level_set == graph["nodes"][list(graph["nodes"].keys())[0]].level_set])
    num_nodes_t1 = len([node for node in graph["nodes"].values() if node.level_set == graph["nodes"][list(graph["nodes"].keys())[0]].level_set + 1])
    scale = num_nodes_t1
    assert num_nodes_t != 0 and num_nodes_t1 != 0, (
        f"Error in setup_single_transition_flow: "
        f"Number of nodes in current level set (t): {num_nodes_t}, "
        f"Number of nodes in next level set (t+1): {num_nodes_t1}. "
        f"Ensure that both level sets have non-zero nodes."
    )

    # Initialize the start and end nodes, capacities for max flow
    start_nodes = []
    end_nodes = []
    capacities = []

    # Set up source and sink nodes
    source = 0
    max_node_id = max(node.id for node in graph["nodes"].values()) # Find the maximum node ID to determine the sink node ID
    sink = max_node_id + 1  # Set the sink node ID to the next integer after the maximum node ID

    # Add arcs from source to nodes in level set t
    for i in range(num_nodes_t):
        node = list(graph["nodes"].values())[i]
        start_nodes.append(source)  # Connect source to each node in level set t
        end_nodes.append(node.id)
        capacities.append(scale)  # Capacity from source to each node in t

    current_level_set = graph["nodes"][list(graph["nodes"].keys())[0]].level_set
    # Add arcs from nodes in level set t to nodes in level set t+1 based on reachability
    for node in graph["nodes"].values():
        from_node_id = node.id
        if node.level_set != current_level_set: # skip processing the next level set nodes
            break
        assert len(node.outgoing_arcs) > 0, f"Error, node {node.id} does not have outgoing arcs, pruning stage should not let this happen."
        # if len(node.outgoing_arcs) == 1: # single outgoing arc is still a problem, stay with them for now
        #     print(f"    len(node.outgoing_arcs) is {len(node.outgoing_arcs)}, state: {node.point}. Level set: {node.level_set}. Node.id: {node.id}")
        for (to_node_id, _) in node.outgoing_arcs:
            # Ensure that 'to_node_id' is within the valid range for nodes in level set t+1
            assert to_node_id <= sink, f'Error, node {node.id} has to_node_id out of bound'
            start_nodes.append(from_node_id)
            end_nodes.append(to_node_id)
            capacities.append(scale)  # Capacity of each connection

    # Add arcs from nodes in level set t+1 to sink
    for i in range(num_nodes_t1):
        node = list(graph["nodes"].values())[num_nodes_t + i]
        start_nodes.append(node.id)  # Connect each node in level set t+1 to sink
        end_nodes.append(sink)
        capacities.append(scale * num_nodes_t / num_nodes_t1)  # Scaled capacity

    # Solve the max flow using the OR-Tools SimpleMaxFlow solver
    smf = max_flow.SimpleMaxFlow()
    all_arcs = smf.add_arcs_with_capacity(start_nodes, end_nodes, capacities)
    status = smf.solve(source, sink)

    solution_flows = smf.flows(all_arcs)
    total_incoming_flow_ideal = num_nodes_t * num_nodes_t1
    total_outgoing_flow = 0

    # mapping (from_node_id, to_node_id) to a tuple (node, i) to access a particular arc
    arc_lookup = {}
    for node_id, node in graph["nodes"].items():
        if node.level_set != current_level_set: # skip processing the next level set nodes
            break
        for i, (to_node_id, _) in enumerate(node.outgoing_arcs):
            if to_node_id is not None:
                arc_lookup[(node_id, to_node_id)] = (node, i)

    for arc, flow in zip(all_arcs, solution_flows):
        tail = smf.tail(arc)
        head = smf.head(arc)
        # Calculate total incoming and outgoing flows
        if head == sink:
            total_outgoing_flow += flow  # All flows to the sink are outgoing
        else:
            lookup = arc_lookup.get((tail, head))
            if lookup: # to skip the case when 'tail' is 0
                node, index = lookup
                to_node_id, current_flow = node.outgoing_arcs[index]
                node.outgoing_arcs[index] = (to_node_id, current_flow + flow)

    # Print the incoming and outgoing flow ratio for uniformity check, if the ratio is 1, means true C-Uniform is achieved
    flow_ratio = total_outgoing_flow / total_incoming_flow_ideal 
    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(f"  Level set {current_level_set:>2} -> {current_level_set + 1:>2}: "
      f"Outgoing/Incoming flow ratio: {total_outgoing_flow:>12}/{total_incoming_flow_ideal:<12} = {flow_ratio:.4f} "
      f"| elapsed time: {elapsed_time:>7.4f}s")
    return flow_ratio
def perturb_state(center_point, grid, division_factor=2, num_samples=5):
    """
    Generates perturbed points around the center point within half the cell dimensions.
    Returns: a List of perturbed points as tuples.
    """
    if division_factor <= 0.00000001: # don't do perturbation in this case
        return [center_point]
    # Convert center point to NumPy array for vectorized operations
    center_point_array = np.array(center_point)

    # Calculate perturbations for all dimensions at once using broadcasting
    perturbations = np.random.uniform(
        -grid.thresholds / division_factor,  # Lower bound for each dimension
        grid.thresholds / division_factor,   # Upper bound for each dimension
        (num_samples, grid.dims)     # Generate an array of size (num_samples, dims)
    )

    # Generate perturbed points by adding perturbations to the center point
    perturbed_points_array = center_point_array + perturbations
    # Convert perturbed points to tuples
    perturbed_points = [tuple(point) for point in perturbed_points_array]
    return perturbed_points

def generate_actions(arange, num_a, steering_angle_range, num_steering_angle, deg2rad_conversion):
    """
    Generates a NumPy array of all possible (steering, velocity) action pairs
    based on the given ranges and discretization parameters.
    Returns: np.ndarray: Array of shape (num_steering_angle * num_v, 2) where each row is [steering, acceleration]
    """
    if deg2rad_conversion:
        steering_values = np.deg2rad(np.linspace(steering_angle_range[0], steering_angle_range[1], num_steering_angle))
    else:
        steering_values = np.linspace(steering_angle_range[0], steering_angle_range[1], num_steering_angle)
    a_value = np.linspace(arange[0], arange[1], num_a)
    actions = np.array([[s, a] for s in steering_values for a in a_value], dtype=np.float32)
    return actions

def generate_uniform_trajectories(config, num_trajectories, trajectory_length, output_file):
    print("Sampling Uniform action trajectories...")
    """ Generates trajectories that uniformly sample actions and saves them to a pickle file.  """
    initial_state = np.zeros(config["state_dim"]).astype(np.float32) 
    actions = config['actions']
    dynamics = config['dynamics']
    vrange = config['vrange']
    t_step = config['dt']
    all_trajectories = []

    for _ in range(num_trajectories):
        trajectory = []
        current_state = initial_state
        for _ in range(trajectory_length):
            # Uniformly sample an action
            if np.ndim(actions) == 1:
                action = np.random.choice(actions)
            else:
                idx = np.random.randint(0, actions.shape[0])
                action = actions[idx]
            trajectory.append((current_state, action))  # Append state and action as a tuple
            current_state = dynamics(current_state, action, t_step, vrange)
        # Append final state with None action as it has no associated action
        trajectory.append((current_state, None))
        all_trajectories.append(trajectory)
    visualize_trajectories(all_trajectories)
    
    # Save the generated trajectories to a pickle file
    with open(output_file, 'wb') as file:
        pickle.dump(all_trajectories, file)
    print(f"Generated {num_trajectories} trajectories and saved to {output_file}")

def analyze_trajectory_distribution(ReaBoxIndices_LSs, trajectories_filename, grid, level_set_range=None, nf_assertion=False):
    """
    Analyze the distribution of trajectories across reachable cells (bins) to check for uniformity.

    Args:
        ReaBoxIndices_LSs (list of sets): List of sets where each set contains the reachable grid indices for each level set.
        trajectories_filename (str): Path to the pickle file containing the generated trajectories.
        grid (Grid): Grid object that can get indices of states in the configuration space.
        level_set_range (tuple, optional): A tuple (start, end) specifying the range of level sets to analyze. Inclusive
                                           If None, all level sets will be analyzed.
    
    Prints:
        - The expected uniform count per bin.
        - The actual count of occurrences per bin for each level set.
    """
    with open(trajectories_filename, 'rb') as file:
        all_trajectories = pickle.load(file)
    if level_set_range:
        start_index, end_index = level_set_range
        if not (0 <= start_index <= end_index < len(ReaBoxIndices_LSs)):
            raise ValueError(f"Invalid level_set_range: ({start_index}, {end_index}). Must be within 0 and {len(ReaBoxIndices_LSs) - 1}.")
    else:
        start_index, end_index = 0, len(ReaBoxIndices_LSs) - 1

    # 'middle' points counts will be split 
    left_counts_ratio = []
    right_counts_ratio = []
    level_set_labels = []

    trajectory_counts_per_level = [{} for _ in range(len(ReaBoxIndices_LSs))] # a list to store occurrence counts for each level set
    for level_set_index in range(start_index, end_index + 1):
        reachable_indices = ReaBoxIndices_LSs[level_set_index]
        print(f"Analyzing Level Set {level_set_index}...")
   
        level_set_counts = {index: 0 for index in reachable_indices} # counts for each reachable cell in the current level set
        for trajectory in all_trajectories: # loop through all trajectories and process each at the current level set
            state, _ = trajectory[level_set_index]  # ignore the action component
            state_index = grid.get_index(state)
            if nf_assertion and state_index not in reachable_indices: # if state_index is not a valid reachable cell
                raise ValueError(f"State index {state_index} at level set {level_set_index} is out of bounds of reachable indices. Double check the pass in parameters")
            if state_index not in level_set_counts:
                level_set_counts[state_index] = 0  # Initialize if it doesn't exist
            level_set_counts[state_index] += 1
        trajectory_counts_per_level[level_set_index] = level_set_counts

    # Print distribution results for each level set in the specified range
    for level_set_index in range(start_index, end_index + 1):
        reachable_indices = ReaBoxIndices_LSs[level_set_index]
        level_set_counts = trajectory_counts_per_level[level_set_index]
        num_reachable_cells = len(reachable_indices)
        num_trajectories = len(all_trajectories)
        
        # Calculate expected uniform count per cell
        uniform_count = num_trajectories / num_reachable_cells
        print(f"Level Set {level_set_index} - Expected Uniform Count per Bin: {uniform_count:.2f}")

        if level_set_index == 0:
            # Print actual counts for each reachable index
            print("Grid Index | Count | Count/Expected Uniform Count")
            for index in sorted(level_set_counts.keys()):
                print(f"{index}: {level_set_counts[index]}, {level_set_counts[index]/uniform_count:.2f}")
        else:
            # Separate indices into left, middle, and right
            left_indices = {index: count for index, count in level_set_counts.items() if index[1] > 0} # if y value > 0
            middle_indices = {index: count for index, count in level_set_counts.items() if index[1] == 0}
            right_indices = {index: count for index, count in level_set_counts.items() if index[1] < 0}

            # Calculate and print ratios for left and right indices
            total_left_count = sum(left_indices.values())
            total_right_count = sum(right_indices.values())

            left_counts_ratio.append((total_left_count + sum(middle_indices.values()) / 2) / len(all_trajectories))
            right_counts_ratio.append((total_right_count + sum(middle_indices.values()) / 2) / len(all_trajectories))
            level_set_labels.append(level_set_index)
            
            DETAILED_PRINT = False
            if DETAILED_PRINT:
                # Print actual counts for each category of indices
                print("    Left Indices:")
                print("      Grid Index | Count | Count/Expected Uniform Count")
                for index in sorted(left_indices.keys()):
                    print(f"      {index}: {left_indices[index]}, {left_indices[index]/uniform_count:.2f}")

                print("    Middle Indices:")
                print("      Grid Index | Count | Count/Expected Uniform Count")
                for index in sorted(middle_indices.keys()):
                    print(f"      {index}: {middle_indices[index]}, {middle_indices[index]/uniform_count:.2f}")

                print("    Right Indices:")
                print("      Grid Index | Count | Count/Expected Uniform Count")
                for index in sorted(right_indices.keys()):
                    print(f"      {index}: {right_indices[index]}, {right_indices[index]/uniform_count:.2f}")
    plt.figure(figsize=(10, 6))
    plt.plot(level_set_labels, left_counts_ratio, label="Left Count Ratio", color="blue", marker="o")
    plt.plot(level_set_labels, right_counts_ratio, label="Right Count Ratio", color="red", marker="o")
    plt.xlabel("Level Set")
    plt.ylabel("Percentage of Points")
    plt.title(f"Evolution of Percentage of Points on Left vs. Right Across Level Sets for {len(all_trajectories)} Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_reachable_level_sets_adaptive_resolution(
        config, disjoint_level_set=True, obstacles=None, adaptive_resolution=False
    ):
    """
    NOTE: the function below provide another way of calculating reachable cells, 
        There are mainly 3 difference vs calculate_reachable_level_sets() function.
        1. It's using fully vectorized operation, which speed up the reachable cell calculation significantly.
        2. This support disjoint level set as parameter for level set calculation.
        3. Support adaptive resolution as parameter to provide high fidelity level sets approximation.

    Calculates the reachable level sets (ReaBox_LSs) and their corresponding grid indices (ReaBoxIndices_LSs)
    Returns:
        list: ReaBoxIndices_LSs - A list of sets containing reachable grid indices for each level set.
        list: ReaBox_LSs - A list of sets containing representative points for all centers of each reachable grid cell.
        list: finer_repre_LSs- A list of np array containing representatives in finer resolution that is used for system propagation
    """
    uniformity_grid = Grid(thresholds=config["thresholds"])
    actions = config["actions"]
    total_t = config["total_t"]
    t_step = config["dt"]
    vrange = config["vrange"]
    vectorized_dynamics = config["vectorized_dynamics"]
    initial_state = np.zeros(config["state_dim"]).astype(np.float32) 

    start_time_all = time.time()
    print("=" * 80)
    print("Exploration Phase: Calculating reachable level sets...")
    print("=" * 80)
    # Number of steps to compute based on total time and time step
    num_steps = int(total_t / t_step)

    block_size = 1000000
    if adaptive_resolution:
        finerGrid_1sixteenth = Grid(thresholds=np.array(uniformity_grid.thresholds)/16.0)
        finerGrid_1eighth    = Grid(thresholds=np.array(uniformity_grid.thresholds)/8.0)
        finerGrid_1fourth    = Grid(thresholds=np.array(uniformity_grid.thresholds)/4.0)
        finerGrid_1half      = Grid(thresholds=np.array(uniformity_grid.thresholds)/2.0)

        # thresholds at which we switch to coarser grids ensure a one-way transition: 1/16 -> 1/8 -> 1/4 -> 1/2
        transition_to_1sixteenth = 100000       # these numbers are obtained through empirical observations
        transition_to_1eighth    = 20000000     # depends on computational resources
        transition_to_1fourth    = 20000000     # if you have powerful CPU & memory, increase thresholds
        transition_to_1half      = 20000000
        transition_to_full       = 20000000
        ROUND_TOLERANCE_DECIMALS = 8            # Define the rounding tolerance for unique pruning

        # Start with the finest resolution NP_FLOAT32
        system_propagation_grid = None 
        system_propagation_grid_resolution = "NP_FLOAT32"
    else:
        system_propagation_grid = uniformity_grid
        system_propagation_grid_resolution = "Full_Uniformity_Resolution"

    # Initialize reachable level sets
    ReaBoxIndices_LSs = []  # store reachable grid indices for each level set
    ReaBox_LSs = []         # store corresponding grid representative points for all centers
    finer_repre_LSs = []    # store corresponding grid representative points for all centers
    level_set_runtimes = []
    actual_samples_last_level_set = np.array([initial_state])

    # Initialize the first level set with the initial state
    initial_index = uniformity_grid.get_index(initial_state)
    initial_center = uniformity_grid.get_grid_center(initial_index)
    ReaBoxIndices_LSs.append({initial_index})  # Start with the initial grid index
    ReaBox_LSs.append({tuple(initial_center)})  # Store the center of the initial grid cell
    finer_repre_LSs.append(actual_samples_last_level_set)
    assert np.array_equal(np.array(initial_center), np.zeros(uniformity_grid.dims)), "Error: Initial center is not all zeros"
    assert np.array_equal(np.array(initial_index), np.zeros(uniformity_grid.dims)), "Error: Initial index is not all zeros"

    # Keep track of all visited indices if disjoint_level_set is True
    all_visited_indices = set([initial_index]) if disjoint_level_set else None
    # Compute reachable sets for each time step
    for step in range(1, num_steps + 1):
        step_start_time = time.time()
        prev_level_set_representatives = actual_samples_last_level_set

        total_samples = 0             # For tracking total underlying samples (before pruning)
        accumulated_new_states = []   # Accepted new states (for propagation).
        accumulated_sys_indices = []  # Their corresponding system propagation indices.
        for i in range(0, len(prev_level_set_representatives), block_size): # Block-based propagation and pruning.
            block = prev_level_set_representatives[i:i+block_size]
            if not adaptive_resolution:
                block = uniformity_grid.perturb_state_deterministic_vectorized(
                    points=block, division_factor=2.01  # Perturb samples within the same cell 
                )
            block_new_states = vectorized_dynamics(block, actions, dt=t_step, vrange=vrange)
            total_samples += block_new_states.shape[0]  # Count all samples generated in this block

            if adaptive_resolution:
                if system_propagation_grid:
                    block_sys_indices = system_propagation_grid.get_index_vectorized(block_new_states)
                else: # Use NP_FLOAT32 resolution: round the new states.
                    block_sys_indices = block_new_states.copy().round(decimals=ROUND_TOLERANCE_DECIMALS)
            else:
                block_sys_indices = system_propagation_grid.get_index_vectorized(block_new_states)

            # Always compute uniformity indices for final reachable cells.
            unique_sys_indices, unique_idx = np.unique(block_sys_indices, axis=0, return_index=True)
            unique_block_states = block_new_states[unique_idx]
            block_uniformity_indices = uniformity_grid.get_index_vectorized(unique_block_states)

            if disjoint_level_set: # filter out states already visited based on uniformity indices
                mask = np.array([tuple(idx) not in all_visited_indices for idx in block_uniformity_indices])
                for idx in block_uniformity_indices:
                    all_visited_indices.add(tuple(idx))
                unique_block_states = unique_block_states[mask]
                unique_sys_indices = unique_sys_indices[mask]
            accumulated_new_states.append(unique_block_states)
            accumulated_sys_indices.append(unique_sys_indices)
        if len(accumulated_new_states) == 0:
            raise RuntimeError("No new states were generated in this step. Check grid resolution or dynamics.")

        if adaptive_resolution:
            if system_propagation_grid_resolution == "NP_FLOAT32" and total_samples > transition_to_1sixteenth:
                system_propagation_grid = finerGrid_1sixteenth
                system_propagation_grid_resolution = "1/16"
            elif system_propagation_grid_resolution == "1/16" and total_samples > transition_to_1eighth:
                system_propagation_grid = finerGrid_1eighth
                system_propagation_grid_resolution = "1/8"
            elif system_propagation_grid_resolution == "1/8" and total_samples > transition_to_1fourth:
                system_propagation_grid = finerGrid_1fourth
                system_propagation_grid_resolution = "1/4"
            elif system_propagation_grid_resolution == "1/4" and total_samples > transition_to_1half:
                system_propagation_grid = finerGrid_1half
                system_propagation_grid_resolution = "1/2"
            elif system_propagation_grid_resolution == "1/2" and total_samples > transition_to_full:
                system_propagation_grid = uniformity_grid 
                system_propagation_grid_resolution = "Full_Uniformity_Resolution"

        new_states_accumulate = np.concatenate(accumulated_new_states, axis=0)
        if system_propagation_grid is not None:
            temp_indices = system_propagation_grid.get_index_vectorized(new_states_accumulate)
            unique_indices = np.unique(temp_indices, axis=0)
            actual_samples_last_level_set = system_propagation_grid.get_grid_centers_vectorized(unique_indices)
        else: # round to the nearest x decimals and get unique points
            rounded_samples = new_states_accumulate.round(decimals=ROUND_TOLERANCE_DECIMALS)
            actual_samples_last_level_set = np.unique(rounded_samples, axis=0) #NOTE: these are not actually indices

        # For final uniformity reachable cells, compute the unique uniformity indices from new_states.
        new_uniformity_indices = set()
        new_uniformity_representatives = set()
        block_uniformity_all = uniformity_grid.get_index_vectorized(new_states_accumulate)
        for idx in block_uniformity_all:
            new_uniformity_indices.add(tuple(idx))
        for index in new_uniformity_indices:
            new_uniformity_representatives.add(tuple(uniformity_grid.get_grid_center(index)))

        ReaBoxIndices_LSs.append(new_uniformity_indices)
        ReaBox_LSs.append(new_uniformity_representatives)
        finer_repre_LSs.append(actual_samples_last_level_set)

        step_end_time = time.time() # measure runtime for a step
        step_elapsed_time = step_end_time - step_start_time
        level_set_runtimes.append(step_elapsed_time)
        print(
                f"Step: {step:<2} -> # of underlying samples before pruning= {total_samples:>11}; "
                f"{actual_samples_last_level_set.shape[0]:>9} and {len(new_uniformity_indices):<9} unique points "
                f"under resolution {system_propagation_grid_resolution}/uniformity respectively."
            )
    # Print the total number of nodes in each level set before returning
    print("\nNumber of reachable cells in each level set:")
    for step, (indices, points) in enumerate(zip(ReaBoxIndices_LSs, ReaBox_LSs)):
        print(f"Level set {step}: {len(points)} cells")
    assert len(ReaBoxIndices_LSs) > 0 and len(ReaBox_LSs) > 0, "Error: ReaBoxIndices_LSs or ReaBox_LSs is empty!"
    elapsed_time_all = time.time() - start_time_all
    print(f"Exploration phase take {elapsed_time_all:.4f} seconds.\n")
    print()
    return ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtimes

# NOTE: this function is not vectorized
# it provide a simple and intuitive way of approximate possible upper bound of reachable area
def calculate_reachable_level_sets(config):
    """
    Calculates the reachable level sets (ReaBox_LSs) and their corresponding grid indices (ReaBoxIndices_LSs)
    Returns:
        list: ReaBoxIndices_LSs - A list of sets containing reachable grid indices for each level set.
        list: ReaBox_LSs - A list of sets containing representative points for all corners of each reachable grid cell.
    """
    print("Calculating reachable level sets...")
    start_explore = time.time()
    grid = Grid(thresholds=config['thresholds'])
    dynamics = config["dynamics"]
    initial_state = np.zeros(config["state_dim"]).astype(np.float32) 
    actions = config["actions"]
    lookback_length = config["total_t"]
    t_step = config["dt"]
    vrange = config["vrange"]
    # Number of steps to compute based on lookback length and time step
    num_steps = int(lookback_length / t_step)
    
    # Initialize reachable level sets
    ReaBoxIndices_LSs = []  # To store reachable grid indices for each level set
    ReaBox_LSs = []  # To store corresponding grid representative points for all corners

    # Initialize the first level set with the initial state
    initial_index = grid.get_index(initial_state)
    initial_center = grid.get_grid_center(initial_index)

    ReaBoxIndices_LSs.append({initial_index})  # Start with the initial grid index
    ReaBox_LSs.append({tuple(initial_center)})  # Store the center of the initial grid cell

    print("initial_index: ", initial_index)
    print("initial_center: ", initial_center)

    # Compute reachable sets for each time step
    for step in range(1, num_steps + 1):
        current_level_set = ReaBox_LSs[step - 1]  # Get the previous level set
        next_level_indices = set()
        next_level_points = set()

        # Iterate over each state in the current level set
        for state in current_level_set:
            # points_to_sample = perturb_state(center_point=state, grid=grid, num_samples=20, division_factor=2.1)
            points_to_sample = perturb_state(center_point=state, grid=grid, num_samples=20, division_factor=0.0)
            points_to_sample = [state] + points_to_sample  # Include original state
            for sampled_state in points_to_sample: # Propagate each sampled point using all actions
                for action in actions:
                    # Propagate state using the dynamics function
                    new_state = dynamics(sampled_state, action, t_step, vrange=vrange)
                    # Get the grid index of the new state
                    new_index = grid.get_index(new_state)
                    # Compute the representative grid point (center) of the new index
                    new_center = grid.get_grid_center(new_index)
                    next_level_indices.add(new_index)
                    next_level_points.add(tuple(new_center))

        # Store the computed level sets for this step
        ReaBoxIndices_LSs.append(next_level_indices)
        ReaBox_LSs.append(next_level_points)

    # Print the total number of nodes in each level set before returning
    print("\nSummary of nodes in each level set:")
    print(f"Runtime for exploration phase is {time.time() - start_explore}")
    for step, (indices, points) in enumerate(zip(ReaBoxIndices_LSs, ReaBox_LSs)):
        print(f"Level set {step}: {len(points)} cells")
    return ReaBoxIndices_LSs, ReaBox_LSs, None, []

def precompute_graph_structure(ReaBox_LSs, ReaBoxIndices_LSs, model_config):
    """
    Precomputes the graph structure for all level sets.

    Args:
        ReaBox_LSs (list): List of sets containing the representative points for each reachable grid cell.
        ReaBoxIndices_LSs (list): List of sets containing reachable grid indices for each level set.
        model_config (dict): Configuration parameters for the selected model.

    Returns:
        graphs (list): A list of graph dictionaries, one per level set transition.
    """
    start_time_all = time.time()
    # Extract model parameters
    vectorized_dynamics = model_config["vectorized_dynamics"]
    actions = model_config["actions"]
    t_step = model_config["dt"]
    perturbation_param = model_config["perturbation_param"]
    slack_param = model_config["slack_parameter"]
    vrange = model_config["vrange"]
    grid = Grid(thresholds=model_config["thresholds"])

    graphs = []
    id_offset = 0
    total_arcs = 0  # track total arcs across all graphs

    print("=" * 80)
    print(f"Building Raw Graph Connections")
    print("=" * 80)
    precompute_runtimes = []

    # Iterate over each level set
    for t in range(len(ReaBox_LSs) - 1):
        start_time = time.time()
        graph = {
            "nodes": {},
            "node_lookup": {},  # Secondary dictionary to map (level_set, point) to node_id
        }
        current_level_set = ReaBox_LSs[t]
        next_level_set = ReaBox_LSs[t + 1]
        next_level_indices = set(ReaBoxIndices_LSs[t + 1]) 

        # Add nodes for current and next levels
        for point in current_level_set:
            node = Node(t, point)
            node.id -= id_offset
            graph["nodes"][node.id] = node
            graph["node_lookup"][(t, point)] = node.id
        for point in next_level_set:
            node = Node(t + 1, point)
            node.id -= id_offset
            graph["nodes"][node.id] = node
            graph["node_lookup"][(t + 1, point)] = node.id
        id_offset += len(next_level_set)

        # Compute reachability by sampling actions (if slack is provided, then dynamics constrained is not strictly enforced)
        arc_count = 0  # Track arcs for this specific graph
        for point in current_level_set:
            node1_id = graph["node_lookup"][(t, point)]
            node1 = graph["nodes"][node1_id]

            # Sample reachable states, keep perturbation param and slack param separate
            points_to_sample = grid.perturb_state_deterministic_vectorized(
                points=np.array([point]), division_factor=perturbation_param
            )
            new_states = vectorized_dynamics(
                states=points_to_sample, actions=actions, dt=t_step, vrange=vrange
            )
            noised_new_states = grid.perturb_state_deterministic_vectorized(
                new_states, division_factor=slack_param
            )

            box_index_entire_LS = grid.get_index_vectorized(noised_new_states)
            box_index_set = set(map(tuple, box_index_entire_LS))
            unique_arc_keys = set()

            for unique_box_index in box_index_set:
                if unique_box_index in next_level_indices:
                    new_center_point = grid.get_grid_center(unique_box_index)
                    node2_id = graph["node_lookup"][(t + 1, tuple(new_center_point))]
                    node2 = graph["nodes"][node2_id]

                    arc_key = (node1.id, node2.id)
                    if arc_key not in unique_arc_keys:
                        unique_arc_keys.add(arc_key)
                        node1.outgoing_arcs.append((node2.id, 0))  # Initialize flow to 0
                        arc_count += 1

        total_arcs += arc_count
        graphs.append(graph)
        end_time = time.time()
        elapsed_time = end_time - start_time
        precompute_runtimes.append(elapsed_time)
        print(f"  Graph {t:>2} (Level set {t:>2} -> {(t+1):>2}): Precomputed in {elapsed_time:.4f} seconds (Nodes: {len(graph['nodes'])}, Arcs: {arc_count}).")
    end_time_all = time.time()
    elapsed_time_all = end_time_all - start_time_all
    print(f"Total graphs built: {len(graphs)}")
    print(f"Total arcs processed: {total_arcs}")
    print(f"Precompute all level set transition graphs takes {elapsed_time_all:.4f} seconds.\n")
    return graphs, precompute_runtimes

def collision_checker_node(point, obstacles):
    """
    Checks if a point collides with any obstacle.
    Args:
        point (tuple): The (x, y) or (x, y, theta) coordinates of the node.
        obstacles (list): List of obstacle definitions. 
                          Rectangles are defined as (x_min, y_min, x_max, y_max).
    Returns:
        bool: True if the point collides with any obstacle, False otherwise.
    """
    # Assume open space at the moment
    return False

def prune_graph(raw_graphs, obstacles, collision_checker_node):
    """
    prunes the graph by to sample safe trajectories
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
    #NOTE: arc collision detection is not really needed if time discretization is small
    #NOTE: in the case where all trajectory lead to collision, 
            all of them will be pruned because there is no safe trajectory
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

        # Step 1: Mark nodes that collide with obstacles as dead
        for node_id, node in current_graph["nodes"].items(): 
            # Check if any node in level set t or t+1 collides with any obstacle
            if collision_checker_node(node.point, obstacles):
                node.outgoing_arcs = []
                dead_nodes.add(node_id)

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
    print(f"  Total pruning time: {total_elapsed_time:.4f}s\n")
    return pruned_graphs 


############################ Functions For Corresponding Arc to specific action ############################
def flow_to_action_prob_approximation_all(graphs, model_config):
    """
    Apply flow_to_action_prob_approximation to all graphs in the input.
    Args:
        graphs (list): List of graph dictionaries, each representing a level set transition.
        model_config (dict): Configuration for the robot model.
    Returns:
        probabilities_list (list): A list of probabilities for each graph.
    """
    start_time_all = time.time()  # start overall timer
    probabilities_list = []

    print("=" * 80)
    print("Action Probability Approximation")
    print("=" * 80)
    for graph_idx, graph in enumerate(graphs):
        probabilities = flow_to_action_prob_approximation(graph, model_config)
        probabilities_list.append(probabilities)
    end_time_all = time.time()  # end overall timer
    elapsed_time_all = end_time_all - start_time_all
    print(f"Approximation of action probabilities for all level set transitions completed in {elapsed_time_all:.4f} seconds.\n")
    assert len(probabilities_list) > 0, (
        "Error: The action probability list is empty. This could be due to trajectories being too short (length < dt) "
        "or an issue in the graph structure or flow computation."
    )
    print("Initial level set action prob pred")
    pprint(probabilities_list[0])
    return probabilities_list

def flow_to_action_prob_approximation(graph, model_config):
    """
    Convert the network flow solution to the action probabilities distribution
    Returns:
        dict: A dictionary where each state (node) has a list of tuple (state, a_p_pairs, corresponding states)
        a_p_pairs: a list of [(a_0, prob_0), (a_1, prob_1), ... (a_n-1, prob_n-1)] where n is the number of outgoing_arcs for the node,
                and prob_0 + prob_1 + ... prob_n-1 should sum up to 1, a_0 is the closest action that lead to s_0(see below)
        corresponding_states: a list of [s_0, s_1, s_2, ..., s_n-1], s_0 is the representative for outgoing_arc[0]
    """
    # Extract model parameters
    t_step = model_config["dt"]
    model = model_config["model_name"]
    inverse_dynamic = model_config["inverse_dynamics"]

    probabilities = []
    current_level_set = graph["nodes"][list(graph["nodes"].keys())[0]].level_set
    max_a = 0
    zero_flow_count = 0  # Counter for the cases where total flow equals 0
    id, node = next(iter(graph["nodes"].items()))

    for node_id, node in graph["nodes"].items():
        if node.level_set == current_level_set:
            a_p_pairs = []
            corresponding_states = []
            n = len(node.outgoing_arcs)
            for to_node_id, flow in node.outgoing_arcs:
                to_node = graph["nodes"][to_node_id]
                corresponding_states.append(to_node.point)
                if model == "2D_RANDOM_WALK": 
                    nearest_action_closed_form = inverse_dynamic(node.point, to_node.point, t_step)
                elif model == "DUBINS": 
                    desired_v, desired_steering_angle = inverse_dynamic(node.point, to_node.point, t_step)
                    nearest_action_closed_form = desired_steering_angle # NOTE: only need steering angle at the moment
                elif model == "KS_3D_STEERING_ANGLE":
                    desired_v, desired_steering_angle = inverse_dynamic(node.point, to_node.point, t_step)
                    nearest_action_closed_form = desired_steering_angle # NOTE: only need steering angle at the moment
                else:
                    raise ValueError(f"MODEL unknown: {model}")
                if abs(nearest_action_closed_form) > max_a:
                    max_a = nearest_action_closed_form
                # NOTE: be careful about the clipping as it is losing information and may not reflect max flow's solution
                # nearest_action_closed_form = np.clip(nearest_action_closed_form, actions[0], actions[-1])
                a_p_pairs.append((nearest_action_closed_form, flow))

            # Normalize flow to probabilities
            total_flow = sum(flow for _, flow in a_p_pairs)
            if total_flow > 0:  # Avoid division by zero
                a_p_pairs = [(action, flow / total_flow) for action, flow in a_p_pairs]
                probabilities.append((node.point, a_p_pairs, corresponding_states))
                assert len(a_p_pairs) == len(corresponding_states) == n, "Error in flow_to_action_prob_approximation() function: action probability pair not having same length as the corresponding states"
            else:
                zero_flow_count += 1 # if a node has 0 outgoing flow, get rid of it from 
                # uniform_prob = 1.0 / n if n > 0 else 0.0  # Uniform probability, avoid division by zero
                # a_p_pairs = [(action, uniform_prob) for action, _ in a_p_pairs]
                # a_p_pairs = [(action, 0.0) for action, flow in a_p_pairs]  # Set all probabilities to 0 if total flow is 0
            # probabilities.append((node.point, a_p_pairs, corresponding_states))
    print(f"    Transition {current_level_set:>2} -> {current_level_set + 1:>2}: "
      f"Max action magnitude = {max_a:<8.4f}, Zero outgoing flow count = {zero_flow_count:<4}")
    return probabilities