import pickle
import numpy as np 

seed = 2025
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
                print(f"No flow detected at level {level} for state: {current_node.point}. Using uniform probabilities.")
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

## Getting the graphs for NN-Cuniform trajectories
with open("NN_Graph_pruned.pkl", 'rb') as file:
    nf_data_loaded = pickle.load(file)
print(f"File loaded successfully")

## Getting the metadata
v = nf_data_loaded['v']
actions = nf_data_loaded['actions']
t_step = nf_data_loaded['t_step']
g = nf_data_loaded['grid']
total_t = nf_data_loaded['total_t']
graphs = nf_data_loaded['graphs']
ReaBoxIndices_LSs = nf_data_loaded['reachable_indicex_across_LS']
num_trajectories_list = [10000]
# Loop through each number of trajectories
print("Sampling trajectories...")
for num_trajectories in num_trajectories_list:
    traj_filename = f"Dubins_C_Uniform_{num_trajectories}_v{v}_graph.pickle"
    sampled_trajectories_graph = sample_trajectories_network_flow(graphs, num_trajectories, seed)
    print(len(sampled_trajectories_graph))
    with open(traj_filename, 'wb') as file:
        pickle.dump(sampled_trajectories_graph, file)
        print(f"Sampled {num_trajectories} trajectories(graph) and saved to {traj_filename}.")