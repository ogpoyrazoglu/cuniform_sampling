'''
This file contains the utility helper functions for C-Uniform sampling codebase, functionalities below:
    1) Trajectory generation 
    2) Visualization
    3) Analysis code
'''
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no display/Qt required (works in WSL/headless)
import matplotlib.pyplot as plt
import os
import numpy as np
import hashlib
import pickle
import time
from ortools.graph.python import max_flow
from classes.grid import Grid, AdaptiveGrid
from classes.graph_structure import Node
import random
from pprint import pprint
import multiprocessing as mp

################################################ Trajectory Generation Code ################################################
def sample_trajectories_network_flow(graphs, num_trajectories, seed, uniform_sampling=False):
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
        uniform_sampling (bool): If True, sample trajectories using uniform probabilities 
                                 instead of flow-based probabilities. Default is False.
    
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
    
    # collect all roots (level_set == 0)
    initial_nodes = [n for n in graphs[0]["nodes"].values() if n.level_set == 0]
    uniform_probs = np.ones(len(initial_nodes)) / len(initial_nodes)

    for _ in range(num_trajectories):
        trajectory = []
        idx0 = np.random.choice(len(initial_nodes), p=uniform_probs)
        current_node = initial_nodes[idx0]
        trajectory.append((current_node.point, None))

        for level in range(traj_len):
            # Extract outgoing node ids and flows
            outgoing_arcs = [(to_node_id, flow) for to_node_id, flow in current_node.outgoing_arcs]
            assert outgoing_arcs, f"No outgoing arcs found at level {level} for node {current_node.id}."

            to_node_ids = [to_node_id for to_node_id, flow in outgoing_arcs]
            flows = np.array([flow for _, flow in outgoing_arcs])

            if uniform_sampling or flows.sum() == 0: # Use uniform probabilities if requested or no flow
                # print(f"Using uniform probabilities at level {level} for state: {current_node.point}.")
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
    plt.close()

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
            plt.plot(x_coords, y_coords, marker='o', linewidth=0.4, markersize=2, alpha=0.4)

        if obstacles: # plot obstacles if provided
            for obs in obstacles: #NOTE: right now assume all obstacles are rectangular
                if len(obs) == 4:  # Ensure the rectangle is defined properly
                    x_min, y_min, x_max, y_max = obs
                    rect_patch = plt.Rectangle(
                        (x_min, y_min),  # Bottom-left corner
                        x_max - x_min,  # Width
                        y_max - y_min,  # Height
                        color='red',
                        alpha=0.5,
                        label='Obstacle'
                    )
                    plt.gca().add_patch(rect_patch)
                elif len(obs) == 3:  # Circle: (pos_x, pos_y, radius)
                    cx, cy, radius = obs
                    circle_patch = plt.Circle(
                        (cx, cy),
                        radius,
                        color='blue',
                        alpha=0.5,
                        label='Obstacle'
                    )
                    plt.gca().add_patch(circle_patch)
                else:
                    print(f"Invalid rectangle format: {obs}")

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axis('equal')
        plt.title(f'Trajectories Visualization ({num_trajectories} Trajectories)')
        plt.grid(True)
        save_path = os.path.join(os.getcwd(), f"trajectories_{num_trajectories}_xy.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Trajectory plot saved to {save_path}")
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
        save_path = os.path.join(os.getcwd(), f"trajectories_{num_trajectories}_3d.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"3D trajectory plot saved to {save_path}")

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
            pt = node.point
            # skip anything with fewer than 3 dims, and only take the first 3 if more
            if len(pt) < 3:
                print("visualize_graph_flow_and_low_arcs_nodes() funciton skip the node visual for 1d/2d case")
                continue
            x, y, theta = pt[:3]
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


        ''' # ignore arcs visualization for now
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
    save_path = os.path.join(os.getcwd(), f"graph_flow_LS{current_level_set}_to_{next_level_set}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Graph flow plot saved to {save_path}")

############################################### Helper Functions Below ###############################################
def obstacles_to_str(obstacles, max_length=30):
    """
    Converts a list of obstacles to a short string representation.
    For rectangular obstacles, format is (x_min, y_min, x_max, y_max)
    For circular obstacles, format is (cx, cy, r)
    """
    if obstacles is None:
        return "none"
    obstacles_str = str(obstacles)
    if len(obstacles_str) > max_length:
        obstacles_str = f'obstacles_{hashlib.md5(obstacles_str.encode()).hexdigest()[:10]}'
    return obstacles_str

def line_collision_check(start_state, end_state, obstacles, sdf, collision_checker_nodes, num_samples=5):
    """
    Checks for collisions along a line segment between two states.
    
    This function samples multiple points along the straight-line path between
    start_state and end_state, and checks each point for collisions. This helps
    detect collisions with thin obstacles that might be missed when checking only
    at discrete states.
    
    Implementation details:
    1. Extracts (x,y) coordinates from the start and end states
    2. Samples `num_samples` intermediate points along the straight line
    3. Uses the existing collision checker to test each sampled point
    4. Returns True if ANY sampled point collides with an obstacle
    
    Args:
        start_state (np.ndarray): The starting state.
        end_state (np.ndarray): The ending state.
        obstacles (list): List of obstacles.
        sdf (np.ndarray): 2D array representing the signed distance function.
        collision_checker_nodes (function): Function to check collisions at discrete points.
        num_samples (int): Number of intermediate points to sample along the line.
        
    Returns:
        bool: True if there is a collision along the line, False otherwise.
    """
    # Extract (x,y) coordinates for collision checking
    start_xy = start_state[:2]
    end_xy = end_state[:2]
    
    # Generate intermediate points along the line
    t_values = np.linspace(0, 1, num_samples+2)[1:-1]  # exclude start and end points
    
    if len(t_values) == 0:  # If no intermediate points, return False
        return False
        
    # Create interpolated points
    interpolated_points = np.array([start_xy * (1-t) + end_xy * t for t in t_values])
    
    # Check collisions at interpolated points
    collisions = collision_checker_nodes(interpolated_points, obstacles, sdf)
    
    return np.any(collisions)

def line_collision_check_vectorized(start_states, end_states, obstacles, sdf, collision_checker_nodes, num_samples=5):
    """
    Vectorized version that checks multiple line segments at once, greatly improving performance.
    
    Instead of checking one line at a time, this function:
    1. Takes arrays of start and end states
    2. Computes interpolated points for all lines in parallel
    3. Performs a single batch collision check for all points
    4. Determines which lines have collisions
    
    This vectorized approach is much faster than the sequential version for larger batches
    as it eliminates Python loop overhead and enables efficient batch processing.
    
    Args:
        start_states (np.ndarray): Array of shape (N, D) with start states
        end_states (np.ndarray): Array of shape (N, D) with end states
        obstacles (list): List of obstacles.
        sdf (np.ndarray): 2D array representing the signed distance function.
        collision_checker_nodes (function): Function to check collisions at discrete points.
        num_samples (int): Number of intermediate points per line segment.
        
    Returns:
        np.ndarray: Boolean array of shape (N,) where True indicates a collision along that line
    """
    # Extract (x,y) coordinates for collision checking (first 2 dimensions)
    start_xy = start_states[:, :2]  # Shape: (N, 2)
    end_xy = end_states[:, :2]      # Shape: (N, 2)
    
    # Generate parameter values for interpolation (exclude start and end points)
    t_values = np.linspace(0, 1, num_samples+2)[1:-1]
    
    if len(t_values) == 0:  # If no intermediate points, return all False
        return np.zeros(len(start_states), dtype=bool)
    
    # Create all interpolated points for all line segments efficiently
    # This is done by creating arrays of points for each t-value, then stacking them
    all_points = []
    num_lines = len(start_states)
    
    for t in t_values:
        # For a specific t value, interpolate between all start and end points at once
        # This computes: start * (1-t) + end * t for all pairs in parallel
        interpolated = start_xy * (1-t) + end_xy * t  # Shape: (N, 2)
        all_points.append(interpolated)
    
    # Reshape the points to check them all at once:
    # From list of arrays with shape (N, 2) to a single array of shape (N*num_samples, 2)
    all_interpolated_points = np.vstack(all_points)
    
    # Perform a single collision check for all points
    all_collisions = collision_checker_nodes(all_interpolated_points, obstacles, sdf)
    
    # Reshape the results to separate by line segment
    # From shape (N*num_samples,) to shape (num_samples, N)
    collision_results = all_collisions.reshape(len(t_values), num_lines)
    
    # Check if any point along each line segment has a collision
    # This gives a boolean array of length N indicating which lines have collisions
    return np.any(collision_results, axis=0)

def calculate_reachable_level_sets_adaptive_uniformity(
    config, disjoint_level_set, obstacles=None, sdf=None, collision_checker_nodes=None,
    enable_line_collision_check=True, line_collision_samples=5
):
    """
    Like the original calculate_reachable_level_sets(),
    but always uses the uniformity grid for propagation and pruning.
    
    Args:
        config (dict): Configuration dictionary with system parameters.
        disjoint_level_set (bool): If True, ensures level sets are disjoint.
        obstacles (list, optional): List of obstacles. Defaults to None.
        sdf (np.ndarray, optional): Signed distance function. Defaults to None.
        collision_checker_nodes (function, optional): Function to check point collisions. Defaults to None.
        enable_line_collision_check (bool, optional): If True, checks for collisions along line 
                                                    segments between consecutive states. Defaults to False.
        line_collision_samples (int, optional): Number of points to sample along each line segment. Defaults to 5.
    
    Returns: 
        tuple: (ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtimes)
    """
    # 1) unpack config
    adaptive_grid       = AdaptiveGrid(
        base_thresholds=config["thresholds"],
        dt=config["dt"],
        max_velocity=config["vrange"][1],
        max_acceleration=max(abs(config["arange"][0]), abs(config["arange"][1])),
        max_steering_deg=max(abs(config["steering_angle_range"][0]), abs(config["steering_angle_range"][1])),
        wheelbase=0.324  # Vehicle wheelbase from dynamics
    )
    actions             = config["actions"]
    total_t             = config["total_t"]
    t_step              = config["dt"]
    vrange              = config["vrange"]
    vectorized_dynamics = config["vectorized_dynamics"]
    initial_state_set   = np.array(config["initial_state_set"],
                                   dtype=np.float32).reshape(-1, config["state_dim"])
    print("=" * 80)
    print("Exploration Phase: Calculating reachable level sets with adaptive uniformity resolution...")
    print("=" * 80)
    start_time_all = time.time()

    # 2) compute number of steps & block size
    num_steps  = int(total_t / t_step)
    block_size = 1_000_000  # for splitting very large state arrays

    # 3) Level‐set 0: discretize initial states into grid cells
    init_indices = adaptive_grid.get_index_vectorized(initial_state_set, level=0, zero_index=True)
    init_centers = adaptive_grid.get_grid_centers_vectorized(init_indices, level=0, zero_index=True)

    ReaBoxIndices_LSs = [ set(map(tuple, init_indices)) ]
    ReaBox_LSs        = [ set(map(tuple, init_centers)) ]
    finer_repre_LSs   = [ initial_state_set.copy() ]
    level_set_runtimes = []

    # 4) iterate levels 1…num_steps
    for step in range(1, num_steps + 1):
        t0 = time.time()
        # a) propagate each block of states through dynamics
        prev_states = finer_repre_LSs[-1]
        blocks = np.array_split(prev_states, max(1, len(prev_states)//block_size))
        new_states_list = []
        for b in blocks:
            b_new = vectorized_dynamics(b, actions, t_step, vrange) # propagate
            if collision_checker_nodes and (obstacles is not None or sdf is not None): # collision filtering
                # First check endpoint collisions
                points_to_check = b_new[:, :2]
                valid_mask = ~collision_checker_nodes(points_to_check, obstacles, sdf)
                
                # If line collision checking is enabled, perform additional checks
                if enable_line_collision_check:
                    # Create an array to track which transitions need line collision checking
                    # (only check lines where endpoints are valid)
                    valid_indices = np.where(valid_mask)[0]
                    
                    # Perform vectorized line collision checking for all valid endpoints
                    if len(valid_indices) > 0:
                        # Extract start and end states for all valid transitions
                        # start_states = b[valid_indices % len(b)]  # Handle the case when b_new has multiple states per input
                        #NOTE: assume the memoery layout is state-first, use the line above if using action-first layout
                        num_actions = len(actions)
                        start_indices = valid_indices // num_actions
                        start_states = b[start_indices]
                        end_states = b_new[valid_indices]
                        
                        # Check all lines at once using vectorized function
                        line_collisions = line_collision_check_vectorized(
                            start_states, end_states, obstacles, sdf, 
                            collision_checker_nodes, line_collision_samples
                        )
                        
                        # Update valid_mask to exclude transitions with line collisions
                        valid_mask[valid_indices[line_collisions]] = False
                
                # Apply collision mask to filter states
                b_new = b_new[valid_mask]
            new_states_list.append(b_new)
        all_new_states = np.concatenate(new_states_list, axis=0)
        if all_new_states.size == 0:
            raise RuntimeError(f"No states at level {step}; check dynamics or collisions")

        # b) assign every new state to its uniform cell, then dedupe
        new_indices = adaptive_grid.get_index_vectorized(all_new_states, level=step, zero_index=True)
        uniq_idx, keep_idx = np.unique(new_indices, axis=0, return_index=True)
        uniq_states = all_new_states[keep_idx]

        # c) if disjoint, drop cells we've already seen under resolution at index 'step'
        if disjoint_level_set:
            # build "visited" set by re‐indexing all prior representatives at this level's resolution
            prev_states = np.concatenate(finer_repre_LSs[:step], axis=0)
            prev_idxs   = adaptive_grid.get_index_vectorized(prev_states, level=step, zero_index=True)
            visited     = set(map(tuple, prev_idxs))
            # prune any new index already seen
            mask        = [tuple(idx) not in visited for idx in uniq_idx]
            uniq_states = uniq_states[mask]
            uniq_idx    = uniq_idx[mask]

        # d) record this level's reachable set
        ReaBoxIndices_LSs.append(set(map(tuple, uniq_idx)))
        # grid‐centers in world‐space
        centers = adaptive_grid.get_grid_centers_vectorized(uniq_idx, level=step, zero_index=True) 
        ReaBox_LSs.append(set(map(tuple, centers)))
        finer_repre_LSs.append(uniq_states)
        level_set_runtimes.append(time.time() - t0)
        th = adaptive_grid.get_thresholds(level=step, zero_index=True)
        print(
            f"Step: {step:<2} | thresholds: "
            f"[{th[0]:.4f}, {th[1]:.4f}, {th[2]:.4f}, {th[3]:.4f}] -> "
            f"# propagated= {all_new_states.shape[0]:>9}; "
            f"{uniq_states.shape[0]:>7} unique points under adaptive uniformity resolution."
        )

    print("\nNumber of reachable cells in each level set:")
    for i, pts in enumerate(ReaBox_LSs):
        print(f"Level set {i:>2}: {len(pts)} cells")
    elapsed_time_all = time.time() - start_time_all
    print(f"Exploration phase take {elapsed_time_all:.4f} seconds.\n")
    return ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtimes

def calculate_reachable_level_sets(
        config, disjoint_level_set, adaptive_resolution=False, obstacles=None, sdf=None, collision_checker_nodes=None,
        enable_line_collision_check=True, line_collision_samples=8
    ):
    """
    Calculates the reachable level sets (ReaBox_LSs) and their corresponding grid indices (ReaBoxIndices_LSs)
    
    Args:
        config (dict): Configuration dictionary with system parameters.
        disjoint_level_set (bool): If True, ensures level sets are disjoint.
        adaptive_resolution (bool, optional): If True, uses adaptive resolution. Defaults to False.
        obstacles (list, optional): List of obstacles. Defaults to None.
        sdf (np.ndarray, optional): Signed distance function. Defaults to None.
        collision_checker_nodes (function, optional): Function to check point collisions. Defaults to None.
        enable_line_collision_check (bool, optional): If True, checks for collisions along line 
                                                    segments between consecutive states. Defaults to False.
        line_collision_samples (int, optional): Number of points to sample along each line segment. Defaults to 5.
    
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
    initial_state_set = config["initial_state_set"]

    start_time_all = time.time()
    print("=" * 80)
    print("Exploration Phase: Calculating reachable level sets...")
    print("=" * 80)
    # Number of steps to compute based on total time and time step
    num_steps = int(round(total_t / t_step))

    block_size = 500000
    if adaptive_resolution:
        finerGrid_1sixteenth = Grid(thresholds=np.array(uniformity_grid.thresholds)/16.0)
        finerGrid_1eighth    = Grid(thresholds=np.array(uniformity_grid.thresholds)/8.0)
        finerGrid_1fourth    = Grid(thresholds=np.array(uniformity_grid.thresholds)/4.0)
        finerGrid_1half      = Grid(thresholds=np.array(uniformity_grid.thresholds)/2.0)

        # thresholds at which we switch to coarser grids ensure a one-way transition: 1/16 -> 1/8 -> 1/4 -> 1/2
        transition_to_1sixteenth = 10000       # these numbers are obtained through empirical observations
        transition_to_1eighth    = 500000      # depends on computational resources
        transition_to_1fourth    = 500000      # if you have powerful CPU & memory, increase thresholds
        transition_to_1half      = 500000
        transition_to_full       = 500000
        ROUND_TOLERANCE_DECIMALS = 8         # Define the rounding tolerance for unique pruning

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
    actual_samples_last_level_set = np.array(initial_state_set).reshape(-1, config['state_dim'])

    # Initialize the first level set with the initial state set
    initial_indices_set = uniformity_grid.get_index_vectorized(initial_state_set)
    initial_center_set = uniformity_grid.get_grid_center(initial_indices_set)
    ReaBoxIndices_LSs.append({tuple(idx)    for idx    in initial_indices_set})
    ReaBox_LSs.append(       {tuple(ctr)    for ctr    in initial_center_set}) # store the center of cells
    finer_repre_LSs.append(actual_samples_last_level_set)

    # Keep track of all visited indices if disjoint_level_set is True
    if disjoint_level_set:
        all_visited_indices = set(tuple(idx) for idx in initial_indices_set)
    else:
        all_visited_indices = None
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
                    points=block, division_factor=0.00  # Perturb samples within the same cell 
                )
            block_new_states = vectorized_dynamics(block, actions, dt=t_step, vrange=vrange)
            
            # Perform collision checking
            if (obstacles is not None or sdf is not None) and collision_checker_nodes is not None:
                # First check endpoint collisions
                points_to_check = block_new_states[:, :2]
                valid_mask = ~collision_checker_nodes(points_to_check, obstacles, sdf)
                
                # If line collision checking is enabled, perform additional checks
                if enable_line_collision_check:
                    # Get indices of valid transitions that need line checking
                    valid_indices = np.where(valid_mask)[0]
                    
                    # Perform vectorized line collision checking for all valid endpoints
                    if len(valid_indices) > 0:
                        # Extract start and end states for all valid transitions
                        # start_states = block[valid_indices % len(block)]  # Handle the case when block_new_states has multiple states per input
                        #NOTE: assume the memoery layout is state-first, use the line above if using action-first layout
                        num_actions = len(actions)
                        start_indices = valid_indices // num_actions
                        start_states = block[start_indices]
                        end_states = block_new_states[valid_indices]
                        
                        # Check all lines at once using vectorized function
                        line_collisions = line_collision_check_vectorized(
                            start_states, end_states, obstacles, sdf, 
                            collision_checker_nodes, line_collision_samples
                        )
                        
                        # Update valid_mask to exclude transitions with line collisions
                        valid_mask[valid_indices[line_collisions]] = False
                
                # Apply collision mask to filter states
                block_new_states = block_new_states[valid_mask]
                
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
            if np.array(unique_block_states).size == 0:
                return None, None, None, None  # don't use this environment
                # all states are blocked by obstacles, return the generated level set so far
                # return ReaBoxIndices_LSs, ReaBox_LSs, finer_repre_LSs, level_set_runtimes
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

        # Do collision checking with respect to the uniformity resolution representatives
        if (obstacles is not None or sdf is not None) and collision_checker_nodes is not None:
            block_uniformity_all_reps = uniformity_grid.get_grid_centers_vectorized(block_uniformity_all)
            temp_points_to_check = block_uniformity_all_reps[:, :2]
            valid_mask = ~collision_checker_nodes(temp_points_to_check, obstacles, sdf)
            all_new_states = block_uniformity_all_reps[valid_mask]
            block_uniformity_all = uniformity_grid.get_index_vectorized(all_new_states)
            if np.array(block_uniformity_all).size == 0:
                return None, None, None, None  # don't use this environment

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

def _process_points_batch(args):
    """
    Worker function to process a batch of points from the current level set in parallel.
    
    Args:
        args: Tuple containing (points_batch, level_t, model_config, next_level_indices, 
                               next_level_set, id_offset)
    
    Returns:
        List of tuples: [(from_node_id_offset, to_node_id_offset, arc_data), ...]
    """
    (points_batch, level_t, model_config, next_level_indices, 
     next_level_set, id_offset) = args
    
    # Extract model parameters
    vectorized_dynamics = model_config["vectorized_dynamics"]
    actions = model_config["actions"]
    t_step = model_config["dt"]
    perturbation_param = model_config["perturbation_param"]
    vrange = model_config["vrange"]
    
    # Initialize grid objects
    if model_config["adaptive_uniformity"]:
        grid = None
        adaptive_grid = AdaptiveGrid(
            base_thresholds=model_config["thresholds"],
            dt=model_config["dt"],
            max_velocity=model_config["vrange"][1],
            max_acceleration=max(abs(model_config["arange"][0]), abs(model_config["arange"][1])),
            max_steering_deg=max(abs(model_config["steering_angle_range"][0]), abs(model_config["steering_angle_range"][1])),
            wheelbase=0.324  # Vehicle wheelbase from dynamics
        )
    else:
        adaptive_grid = None
        grid = Grid(thresholds=model_config["thresholds"])
    
    # Create lookup for next level points to their node IDs
    next_level_lookup = {}
    node_id_offset = 0
    for point in next_level_set:
        next_level_lookup[tuple(point)] = node_id_offset
        node_id_offset += 1
    
    batch_arcs = []
    
    for point_idx, point in enumerate(points_batch):
        # Calculate node1 ID (from current level)
        node1_id_offset = point_idx  # Relative to this batch
        
        # Sample reachable states
        if model_config["adaptive_uniformity"]:
            points_to_sample = adaptive_grid.perturb_state_deterministic_vectorized(
                points=np.array([point]), division_factor=perturbation_param, level=level_t, zero_index=True,
            )
        else:
            points_to_sample = grid.perturb_state_deterministic_vectorized(
                points=np.array([point]), division_factor=perturbation_param
            )
        new_states = vectorized_dynamics(
            states=points_to_sample, actions=actions, dt=t_step, vrange=vrange
        )

        if model_config["adaptive_uniformity"]:
            box_index_entire_LS = adaptive_grid.get_index_vectorized(new_states, level_t+1, True)
        else:
            box_index_entire_LS = grid.get_index_vectorized(new_states)
        box_index_set = set(map(tuple, box_index_entire_LS))
        unique_targets = set()

        for unique_box_index in box_index_set:
            if unique_box_index in next_level_indices:
                if model_config["adaptive_uniformity"]:
                    new_center_point = adaptive_grid.get_grid_center(unique_box_index, level_t+1, True)
                else:
                    new_center_point = grid.get_grid_center(unique_box_index)
                
                new_center_tuple = tuple(new_center_point)
                if new_center_tuple in next_level_lookup and new_center_tuple not in unique_targets:
                    unique_targets.add(new_center_tuple)
                    node2_id_offset = next_level_lookup[new_center_tuple]
                    batch_arcs.append((node1_id_offset, node2_id_offset, new_center_tuple))
    
    return batch_arcs

def precompute_graph_structure_parallel(ReaBox_LSs, ReaBoxIndices_LSs, model_config, num_processes=None):
    """
    Parallelized version of precompute_graph_structure that processes points in batches across multiple cores.
    
    Args:
        ReaBox_LSs (list): List of sets containing the representative points for each reachable grid cell.
        ReaBoxIndices_LSs (list): List of sets containing reachable grid indices for each level set.
        model_config (dict): Configuration parameters for the selected model.
        num_processes (int, optional): Number of processes to use. If None, uses all available cores.
    
    Returns:
        graphs (list): A list of graph dictionaries, one per level set transition.
        precompute_runtimes (list): List of computation times for each level set.
    """
    start_time_all = time.time()
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    graphs = []
    id_offset = 0
    total_arcs = 0
    precompute_runtimes = []

    print("=" * 80)
    print(f"Building Raw Graph Connections (Parallel - {num_processes} processes)")
    print("=" * 80)

    # Iterate over each level set
    for t in range(len(ReaBox_LSs) - 1):
        start_time = time.time()
        graph = {
            "nodes": {},
            "node_lookup": {},  # Secondary dictionary to map (level_set, point) to node_id
        }
        current_level_set = list(ReaBox_LSs[t])  # Convert to list for indexing
        next_level_set = list(ReaBox_LSs[t + 1])
        next_level_indices = set(ReaBoxIndices_LSs[t + 1])

        # Add nodes for current and next levels
        current_level_node_lookup = {}
        for i, point in enumerate(current_level_set):
            node = Node(t, point)
            node.id -= id_offset
            graph["nodes"][node.id] = node
            graph["node_lookup"][(t, point)] = node.id
            current_level_node_lookup[tuple(point)] = node.id
        
        for point in next_level_set:
            node = Node(t + 1, point)
            node.id -= id_offset
            graph["nodes"][node.id] = node
            graph["node_lookup"][(t + 1, point)] = node.id
        id_offset += len(next_level_set)

        # Compute reachability by sampling actions in parallel
        arc_count = 0
        
        # Split current level set into batches for parallel processing
        batch_size = max(1, len(current_level_set) // (num_processes * 2))  # 2 batches per process
        point_batches = [current_level_set[i:i + batch_size] 
                        for i in range(0, len(current_level_set), batch_size)]
        
        if len(point_batches) > 0:
            # Prepare arguments for parallel processing
            batch_args = []
            for batch in point_batches:
                batch_args.append((
                    batch, t, model_config, next_level_indices, 
                    next_level_set, id_offset
                ))
            
            # Process batches in parallel
            with mp.Pool(num_processes) as pool:
                batch_results = pool.map(_process_points_batch, batch_args)
            
            # Aggregate results and build arcs
            current_batch_start = 0
            unique_arc_keys = set()
            
            for batch_idx, batch_arcs in enumerate(batch_results):
                for node1_offset, node2_offset, target_point in batch_arcs:
                    # Convert relative offsets to actual node IDs
                    point_idx = current_batch_start + node1_offset
                    point = current_level_set[point_idx]
                    node1_id = current_level_node_lookup[tuple(point)]
                    node2_id = graph["node_lookup"][(t + 1, target_point)]
                    
                    arc_key = (node1_id, node2_id)
                    if arc_key not in unique_arc_keys:
                        unique_arc_keys.add(arc_key)
                        node1 = graph["nodes"][node1_id]
                        node1.outgoing_arcs.append((node2_id, 0))  # Initialize flow to 0
                        arc_count += 1
                
                current_batch_start += len(point_batches[batch_idx])

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

def precompute_graph_structure(ReaBox_LSs, ReaBoxIndices_LSs, model_config):
    """
    Precomputes the graph structure for all level sets.

    Args:
        ReaBox_LSs (list): List of sets containing the representative points for each reachable grid cell.
        ReaBoxIndices_LSs (list): List of sets containing reachable grid indices for each level set.
        model_config (dict): Configuration parameters for the selected model, including:
            - vectorized_dynamics (callable): Function for vectorized dynamics.
            - actions (np.ndarray): Array of discretized actions.
            - dt (float): Time discretization step size (seconds).
            - v (float): Constant velocity (m/s).

    Returns:
        graphs (list): A list of graph dictionaries, one per level set transition.
    """
    start_time_all = time.time()
    # Extract model parameters
    vectorized_dynamics = model_config["vectorized_dynamics"]
    actions = model_config["actions"]
    t_step = model_config["dt"]
    perturbation_param = model_config["perturbation_param"]
    vrange = model_config["vrange"]
    if model_config["adaptive_uniformity"]:
        grid = None
        adaptive_grid = AdaptiveGrid(
            base_thresholds=model_config["thresholds"],
            dt=model_config["dt"],
            max_velocity=model_config["vrange"][1],
            max_acceleration=max(abs(model_config["arange"][0]), abs(model_config["arange"][1])),
            max_steering_deg=max(abs(model_config["steering_angle_range"][0]), abs(model_config["steering_angle_range"][1])),
            wheelbase=0.324  # Vehicle wheelbase from dynamics
        )
    else:
        adaptive_grid = None
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

        # Compute reachability by sampling actions
        arc_count = 0  # Track arcs for this specific graph
        for point in current_level_set:
            node1_id = graph["node_lookup"][(t, point)]
            node1 = graph["nodes"][node1_id]

            # Sample reachable states
            if model_config["adaptive_uniformity"]:
                points_to_sample = adaptive_grid.perturb_state_deterministic_vectorized(
                    points=np.array([point]), division_factor=perturbation_param, level=t, zero_index=True,
                )
            else:
                points_to_sample = grid.perturb_state_deterministic_vectorized(
                    points=np.array([point]), division_factor=perturbation_param
                )
            new_states = vectorized_dynamics(
                states=points_to_sample, actions=actions, dt=t_step, vrange=vrange
            )

            if model_config["adaptive_uniformity"]:
                box_index_entire_LS = adaptive_grid.get_index_vectorized(new_states, t+1, True)
            else:
                box_index_entire_LS = grid.get_index_vectorized(new_states)
            box_index_set = set(map(tuple, box_index_entire_LS))
            unique_arc_keys = set()

            for unique_box_index in box_index_set:
                if unique_box_index in next_level_indices:
                    if model_config["adaptive_uniformity"]:
                        new_center_point = adaptive_grid.get_grid_center(unique_box_index, t+1, True)
                    else:
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

def prune_graph(raw_graphs):
    """
    prunes the graph by to sample safe trajectories
    Args:
        raw_graphs (list): List of raw graph dictionaries. A "raw graph" is an unprocessed graph.

    Returns:
        pruned_graphs (list): List of pruned graph dictionaries.

    High-level pruning logic:
    1. Backward pass: Prune nodes that have no outgoing arcs.
    2. Forward pass: Identify and prune nodes that have no incoming arcs
                    make sure all nodes are reachable from the initial state in level set 0.

    Assumptions:
        Assume the collision checking is already done in calculate_reachable_level_sets()
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

        # Prune arcs lead to dead nodes and nodes with no valid outgoing arcs
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
        print(
            f"  Transition {i:02d} -> {i+1:02d} | "
            f"Removed: {nodes_removed:5d} | "
            f"Time: {elapsed_time:7.4f}s | "
            f"Nodes in L{i:02d}: {nodes_in_level_set_i:5d}"
        )

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
        print(
            f"  Forward pass Level set {i:>2} -> {i+1:>2}: "
            f"Removed: {nodes_removed:<5} | "
            f"Time: {elapsed_time:>7.4f}s"
        )

    total_elapsed_time = time.time() - total_start_time # total elapsed time
    print("Summary:")
    print(f"  Total nodes removed (all passes): {total_removed_nodes}")
    print(f"  Total pruning time: {total_elapsed_time:.4f}s\n")
    return pruned_graphs 

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

def get_state_u_distribution_across_LS(grid, graphs, probabilities_list):
    """
    Helper function to extract states and corresponding action probability distributions across level sets.
    Args:
        grid: Grid class
        graphs (list): A list of graph structures, each representing a level set.
        probabilities_list (list): A list of dictionaries where each index corresponds to a level set,
                                   and each dictionary maps node IDs to action probability arrays.
    NOTE: the return arguments do not have the last level set because we don't have action probabilities for the last level set
    """
    state_u_distribution_across_LS = []  # To store states and action probability distribution pairs across level sets
    grid_ind_u_distribution_across_LS = []
    for level_set_index, probabilities in enumerate(probabilities_list):
        st_level_set_data = []  # To store state-action distribution pairs for this level set
        ind_level_set_data = {}
        # For each item in the dictionary (key is node ID, value is action probability array)
        for node_id, action_probabilities in probabilities.items():
            # Look up the node in the corresponding graph using node_id to retrieve the state [x, y, theta]
            node = graphs[level_set_index]["nodes"][node_id]
            state = list(node.point)  # Extract the state as [x, y, theta]
            # Support both static Grid and AdaptiveGrid
            if grid.__class__.__name__ == 'AdaptiveGrid':
                ind = grid.get_index(state, level_set_index, True)
            else:
                ind = grid.get_index(state)
            st_level_set_data.append((state, action_probabilities)) # Store the state & corresponding action probability distribution
            ind_level_set_data[ind] = action_probabilities
        state_u_distribution_across_LS.append(st_level_set_data) # Append the level set data to the main list
        grid_ind_u_distribution_across_LS.append(ind_level_set_data)
    return state_u_distribution_across_LS, grid_ind_u_distribution_across_LS

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

def analyze_trajectory_distribution(
        ReaBoxIndices_LSs, trajectories_filename, grid, level_set_range=None, nf_assertion=False
    ):
    """
    Analyze the distribution of trajectories across reachable cells (bins) to check for uniformity.

    Args:
        ReaBoxIndices_LSs (list of sets): List of sets where each set contains the reachable grid indices for each level set.
        trajectories_filename (str): Path to the pickle file containing the generated trajectories.
        grid (Grid or AdaptiveGrid): Grid object (static or adaptive) that can get indices of states.
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
            # support both static Grid and AdaptiveGrid (which requires level and zero_index)
            if grid.__class__.__name__ == 'AdaptiveGrid':
                state_index = grid.get_index(state, level_set_index, True)
            else:
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
        # print(f"Level Set {level_set_index} - Expected Uniform Count per Bin: {uniform_count:.2f}")

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

############################ Functions For Corresponding Arc to specific action ############################
def flow_to_action_prob_approximation_all(graphs, model_config):
    """
    Convert network flow solutions directly to compressed action probabilities format.
    
    Args:
        graphs (list): List of graph dictionaries, each representing a level set transition.
        model_config (dict): Configuration for the robot model.
    Returns:
        probabilities_list (list): A list of numpy arrays for each graph. Each array has shape (N, state_dim + num_actions)
                                 where N is the number of nodes in the level set. The first state_dim columns contain
                                 the state values (x, y, theta, etc.), and the remaining num_actions columns
                                 contain the action probabilities. For single-dimensional action models, num_actions
                                 equals num_steering_angle. For multi-dimensional action models like KS_4D_STEERING_ANGLE_V,
                                 num_actions equals num_steering_angle * num_acceleration.
    """
    start_time_all = time.time()  # start overall timer
    
    # Pre-compute all graph statistics once to avoid expensive repeated counting
    print("=" * 80)
    print("Action Probability Approximation")
    print("=" * 80)
    print("  Pre-computing graph statistics...")
    
    graph_stats = []  # Will store (current_level_nodes, total_arcs_in_graph) for each graph
    
    for graph in graphs:
        # Get current level set efficiently (get first key without creating list)
        first_node_id = next(iter(graph["nodes"]))
        current_level_set = graph["nodes"][first_node_id].level_set
        
        # Single pass through nodes to count both nodes and arcs
        current_level_nodes = 0
        total_arcs_in_graph = 0
        
        for node in graph["nodes"].values():
            if node.level_set == current_level_set:
                current_level_nodes += 1
                total_arcs_in_graph += len(node.outgoing_arcs)
        
        graph_stats.append((current_level_nodes, total_arcs_in_graph))
    
    total_nodes_processed = sum(stats[0] for stats in graph_stats)
    total_arcs_processed = sum(stats[1] for stats in graph_stats)

    print("  Converting action probabilities to compressed format...")
    probabilities_list = []
    total_processing_time = 0.0
    graph_times = []
    
    for graph_idx, graph in enumerate(graphs):
        graph_start_time = time.time()
        current_level_nodes, total_arcs_in_graph = graph_stats[graph_idx] # pre-computed statistics
        
        # Generate compressed action probabilities directly
        compressed_probabilities = _convert_flow_to_compressed_format(graph, model_config)
        probabilities_list.append(compressed_probabilities)
            
        graph_end_time = time.time()
        graph_total_time = graph_end_time - graph_start_time
        total_processing_time += graph_total_time
        
        # Store profiling data
        graph_times.append(graph_total_time)
        
        print(f"    Graph {graph_idx:2d}: {current_level_nodes:6d} nodes, {total_arcs_in_graph:8d} arcs | "
              f"Time: {graph_total_time:6.3f}s")
            
    end_time_all = time.time()  # end overall timer
    elapsed_time_all = end_time_all - start_time_all
    
    # Calculate overhead
    remaining_overhead = elapsed_time_all - total_processing_time
    
    # Print simplified profiling summary
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY - Action Probability Approximation")
    print("=" * 80)
    print(f"Total processing time:           {elapsed_time_all:.4f}s")
    print(f"Action probability generation:   {total_processing_time:.4f}s ({total_processing_time/elapsed_time_all*100:.1f}%)")
    print(f"Overhead time:                   {remaining_overhead:.4f}s ({remaining_overhead/elapsed_time_all*100:.1f}%)")
    
    print(f"\nTotal graphs processed:          {len(graphs)}")
    print(f"Total nodes processed:           {total_nodes_processed:,}")
    print(f"Total arcs processed:            {total_arcs_processed:,}")
    
    # Show scaling analysis
    print("\nGraph scaling analysis:")
    print("Graph | Nodes  |   Arcs   | Time(s) | Time/Node(ms) | Time/Arc(ms)")
    print("-" * 65)
    for i, (graph_time, (nodes, arcs)) in enumerate(zip(graph_times, graph_stats)):
        time_per_node = graph_time / nodes * 1000 if nodes > 0 else 0
        time_per_arc = graph_time / arcs * 1000 if arcs > 0 else 0
        print(f"{i:5d} | {nodes:6d} | {arcs:8d} | {graph_time:7.3f} | {time_per_node:13.4f} | {time_per_arc:11.4f}")
    
    print("=" * 80)
    
    assert len(probabilities_list) > 0, (
        "Error: The action probability list is empty. This could be due to trajectories being too short (length < dt) "
        "or an issue in the graph structure or flow computation."
    )
    
    print(f"DEBUG: Compressed format - Initial level set shape: {probabilities_list[0].shape}")
    print(f"       First row (state + action probs): {probabilities_list[0][0]}")
    
    return probabilities_list

def _convert_flow_to_compressed_format(graph, model_config):
    """
    Convert network flow solution directly to compressed action probabilities format.
    
    This function processes the network flow results and generates discretized action
    probabilities in a compressed numpy array format for efficient storage and processing.
    
    Args:
        graph (dict): Graph structure containing nodes for a single transition
        model_config (dict): Model configuration containing action discretization parameters
        
    Returns:
        np.ndarray: Compressed array of shape (N, state_dim + num_actions) where:
                   - N is the number of nodes in current level set
                   - First state_dim columns contain state values
                   - Remaining num_actions columns contain discretized action probabilities
    """
    # Extract model parameters
    t_step = model_config["dt"]
    model = model_config["model_name"]
    inverse_dynamic = model_config["inverse_dynamics"]
    state_dim = model_config["state_dim"]
    
    # Assertions for core assumptions
    assert "nodes" in graph, "Graph must contain nodes"
    assert len(graph["nodes"]) > 0, "Graph must have at least one node"
    
    current_level_set = graph["nodes"][list(graph["nodes"].keys())[0]].level_set
    
    # Get nodes in current level set and validate
    current_level_nodes = [node for node in graph["nodes"].values() if node.level_set == current_level_set]
    
    # Verify pruning worked correctly - no nodes should have zero outgoing arcs
    nodes_with_zero_arcs = [node for node in current_level_nodes if len(node.outgoing_arcs) == 0]
    assert len(nodes_with_zero_arcs) == 0, f"Found {len(nodes_with_zero_arcs)} nodes with zero outgoing arcs after pruning. Pruning stage failed!"
    
    total_arcs = sum(len(node.outgoing_arcs) for node in current_level_nodes)
    assert total_arcs > 0, f"Expected arcs to process, but found {total_arcs} arcs"
    
    # === BLOCK 1: Action Space Configuration ===
    # Setup action discretization bins (same logic as _convert_to_compressed_format)
    if model == "KS_4D_STEERING_ANGLE_V":
        # Multi-dimensional action space: steering angle + acceleration
        num_steering = model_config["num_steering_angle"]
        num_accel = model_config["num_a"]
        num_actions = num_steering * num_accel
        
        # Create discrete action bins for 2D action space
        steering_bins = np.deg2rad(np.linspace(
            model_config["steering_angle_range"][0], 
            model_config["steering_angle_range"][1], num_steering))
        accel_bins = np.linspace(
            model_config["arange"][0], 
            model_config["arange"][1], num_accel)
        vrange = model_config["vrange"]
    elif model == "KS_3D_V_CMD":
        # Multi-dimensional action space: steering angle + v_cmd
        num_steering = model_config["num_steering_angle"]
        num_accel = model_config["num_v"]  # num_v bins over vrange
        num_actions = num_steering * num_accel
        
        steering_bins = np.deg2rad(np.linspace(
            model_config["steering_angle_range"][0], 
            model_config["steering_angle_range"][1], num_steering))
        accel_bins = np.linspace(  # these are v_cmd bins, discretized from vrange
            model_config["vrange"][0], 
            model_config["vrange"][1], num_accel)
        vrange = model_config["vrange"]
    else:
        # Single-dimensional action space: steering angle only
        num_steering = model_config["num_steering_angle"]
        num_actions = num_steering
        
        # Create discrete action bins for 1D action space
        if model == "2D_RANDOM_WALK":
            steering_bins = np.linspace(
                model_config["steering_angle_range"][0],
                model_config["steering_angle_range"][1], num_steering)
        else:
            # Convert degrees to radians for DUBINS and KS_3D_STEERING_ANGLE
            steering_bins = np.deg2rad(np.linspace(
                model_config["steering_angle_range"][0],
                model_config["steering_angle_range"][1], num_steering))
    
    # === BLOCK 2: Initialize Processing Variables ===
    valid_node_data = []  # Will store valid node data
    zero_flow_count = 0  # Count nodes with zero total flow (not zero arcs)
    
    if model in ("KS_4D_STEERING_ANGLE_V", "KS_3D_V_CMD"):
        max_steering = 0.0
        max_acceleration = 0.0  # for KS_3D_V_CMD this tracks max v_cmd magnitude
    else:
        max_action = 0.0
    
    # === BLOCK 3: Process Each Node ===
    # All nodes should have outgoing arcs after pruning
    for node_idx, node in enumerate(current_level_nodes):
        
        # === BLOCK 4: Vectorized Inverse Dynamics for This Node ===
        # Collect arc data for this node
        target_states = []
        flows = []
        
        for to_node_id, flow in node.outgoing_arcs:
            to_node = graph["nodes"][to_node_id]
            target_states.append(to_node.point)
            flows.append(flow)
        
        # Convert to numpy arrays for vectorized processing
        start_states = np.tile(node.point, (len(target_states), 1)).astype(np.float32)
        target_states = np.array(target_states, dtype=np.float32)
        flows = np.array(flows, dtype=np.float32)
        
        # Compute actions using vectorized inverse dynamics
        if model == "2D_RANDOM_WALK":
            actions = (target_states[:, 1] - start_states[:, 1]) / t_step
        elif model in ("DUBINS", "KS_3D_STEERING_ANGLE"):
            # For these models, compute actions individually (could be vectorized later)
            actions = []
            for i in range(len(start_states)):
                desired_v, desired_steering_angle = inverse_dynamic(
                    start_states[i], target_states[i], t_step
                )
                actions.append(desired_steering_angle)
            actions = np.array(actions, dtype=np.float32)
        elif model == "KS_4D_STEERING_ANGLE_V":
            # Use vectorized inverse dynamics
            steering_angles, accelerations = inverse_dynamic(
                start_states, target_states, t_step, vrange
            )
            actions = list(zip(steering_angles, accelerations))
        elif model == "KS_3D_V_CMD":
            # Vectorized inverse dynamics: returns (steering_angles, v_cmds)
            steering_angles, accelerations = inverse_dynamic(
                start_states, target_states, t_step, vrange
            )
            actions = list(zip(steering_angles, accelerations))
        else:
            raise ValueError(f"MODEL unknown: {model}")
        
        # Update magnitude statistics
        if model in ("KS_4D_STEERING_ANGLE_V", "KS_3D_V_CMD"):
            max_steering = max(max_steering, np.max(np.abs(steering_angles)))
            max_acceleration = max(max_acceleration, np.max(np.abs(accelerations)))
        else:
            if isinstance(actions, np.ndarray):
                max_action = max(max_action, np.max(np.abs(actions)))
            else:
                max_action = max(max_action, max(abs(action) for action in actions))
        
        # === BLOCK 5: Check Flow and Store Valid Nodes ===
        # Normalize flows to probabilities
        total_flow = np.sum(flows)
        if total_flow > 0:
            probs = flows / total_flow
            
            # Store this node's data for later processing
            valid_node_data.append({
                'node': node,
                'actions': actions,
                'probs': probs,
                'state': node.point
            })
        else:
            # This can happen if all flows are 0 (rare but possible after network flow)
            zero_flow_count += 1
    
    # === BLOCK 4: Initialize Output Array (only for valid nodes) ===
    num_valid_nodes = len(valid_node_data)
    compressed_array = np.zeros((num_valid_nodes, state_dim + num_actions), dtype=np.float32)
    
    # === BLOCK 5: Process Valid Nodes Only ===
    for valid_idx, node_data in enumerate(valid_node_data):
        # Store state values in first columns
        compressed_array[valid_idx, :state_dim] = node_data['state'][:state_dim]
        
        # Initialize action probability array for this node
        action_probs = np.zeros(num_actions, dtype=np.float32)
        
        actions = node_data['actions']
        probs = node_data['probs']
        
        # Vectorized action discretization (same logic as _convert_to_compressed_format)
        if model in ("KS_4D_STEERING_ANGLE_V", "KS_3D_V_CMD"):
            # Process 2D actions (steering + acceleration) in batch
            actions_array = np.array(actions)
            
            # Find nearest bins for steering angles and accelerations simultaneously
            steering_diffs = np.abs(steering_bins[np.newaxis, :] - actions_array[:, 0:1])
            steering_indices = np.argmin(steering_diffs, axis=1)
            
            accel_diffs = np.abs(accel_bins[np.newaxis, :] - actions_array[:, 1:2])
            accel_indices = np.argmin(accel_diffs, axis=1)
            
            # Convert 2D indices to flat indices (matches generate_actions order)
            action_indices = steering_indices * num_accel + accel_indices
        else:
            # Process 1D actions (steering only) in batch
            actions_array = np.array(actions)
            
            # Find nearest bins for all actions simultaneously
            action_diffs = np.abs(steering_bins[np.newaxis, :] - actions_array[:, np.newaxis])
            action_indices = np.argmin(action_diffs, axis=1)
        
        # Accumulate probabilities into discrete bins
        np.add.at(action_probs, action_indices, probs)
        
        # Store action probabilities
        compressed_array[valid_idx, state_dim:] = action_probs
    
    # Print results based on action dimensionality
    if model in ("KS_4D_STEERING_ANGLE_V", "KS_3D_V_CMD"):
        print(f"    Transition {current_level_set:>2} -> {current_level_set + 1:>2}: "
              f"Max steering = {max_steering:<8.4f}, Max acceleration = {max_acceleration:<8.4f}, Zero outgoing flow count = {zero_flow_count:<4}")
    else:
        print(f"    Transition {current_level_set:>2} -> {current_level_set + 1:>2}: "
              f"Max action magnitude = {max_action:<8.4f}, Zero outgoing flow count = {zero_flow_count:<4}")
    
    return compressed_array