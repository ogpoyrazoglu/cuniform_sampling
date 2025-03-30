import pickle
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
from classes.grid import Grid
import math
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def coverage_visualization(g, resolution, reachable_indices, sampled_trajectories_filename, num_traj,all_trajectories, is_there_action_in_trajs=True, filename = "reachable_area_trajectories_c_uniform.pdf"):
    '''
    resolution tell you the resolution for [x, y, theta] respectively
    reachable_indices contains a set of reachable indices
    sampled_trajectories contain N number ofsampled trajectories
    num_traj is an integer, if num_traj < N, onnly check the first num_traj number of trajectories from sampled_trajectories
    is_there_action_in_trajs tell you whether trajectories contain action or not

    this function visualize what are the 'reachbale areas' in 2 dimension, visualize them as grids
      showing theta coverage as a shade of grey where more grey means you can cover the same xy with different possible yaw angle
    '''
    ###### visualize reachable_cell using grid like structure ######
    ## Getting the X,Y and Theta for the reachable cells
    coverage = {}
    for level_set in reachable_indices:
        for (x, y, theta) in level_set:
            xy_index = (x, y)
            if xy_index not in coverage:
                coverage[xy_index] = set()
            coverage[xy_index].add(theta)
    print("len(coverage)")
    print(len(coverage))
    max_theta_count = max(len(v) for v in coverage.values())
    intensities = {k: len(v) / max_theta_count for k, v in coverage.items()}
    # Plot the coverage
    fig, ax = plt.subplots(figsize=(15, 15))
    ## Adding rectangles for reachable cells
    for (x, y), intensity in intensities.items():
        color = (0.678, 0.847, 0.902)
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))
    
    num_trajectories = len(all_trajectories)
    theta_grad = {}
    ## Putting areas which is covered by the trajectories and intensity decided by theta coverage
    for i in range(num_trajectories): # Loop through each trajectory
        trajectory = all_trajectories[i]
        x_coords = []
        y_coords = []
        theta_coords = []
        for entry in range(len(trajectory)): # Extract x, y coordinates for each state in the trajectory
            # if isinstance(entry, tuple) and len(entry) == 2:
            grid_index = g.get_index(trajectory[entry][0])
            if grid_index in reachable_indices[entry]:
                state, action = trajectory[entry]  # unpack state and action
                x, y, theta = grid_index[0], grid_index[1], grid_index[2] # Extract x and y coordinates
                x_coords.append(x)
                y_coords.append(y)
                theta_coords.append(theta)
                if (x,y) not in theta_grad:
                    theta_grad[(x,y)] = set()
                theta_grad[(x,y)].add(theta)
    max_theta_count = max(len(v) for v in theta_grad.values())
    intensities = {k: len(v) / max_theta_count for k, v in theta_grad.items()}

    cmap = cm.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=min(intensities.values()), vmax=max(intensities.values()))

    for (x, y), intensity in intensities.items():
        ## Adding gradient colors based on theta
        color = cmap(norm(intensity))
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color))
    ax.set_xlim(min(k[0] for k in coverage.keys()) - 1, max(k[0] for k in coverage.keys()) + 2)
    ax.set_ylim(min(k[1] for k in coverage.keys()) - 1, max(k[1] for k in coverage.keys()) + 2)
    ax.axis("equal")
    plt.axis(False)

    plt.savefig(filename)
    return 


# Main function
def main():
    input_filename = "/home/isleri/mahes092/C_Uniform/Datasets/C_Uniform_processed_disjoint_KS_3D_STEERING_ANGLE_perturb_2.01_slack_0.0_seed_2025_grid_0.03_0.03_6.00deg_t3.01_ts0.2vrange_1.0_1.0_steering_13.pkl"
    ## Getting Config File
    with open(input_filename, 'rb') as f:
        data = pickle.load(f)
    ## Getting thresholds
    thresholds = data["config"]["thresholds"]
    g = Grid(thresholds=thresholds)
    resolution = None
    ## Getting reachable cells
    reachable_indices = data['reachable_indicex_across_LS']
    sampled_trajectories_filename = None
    num_traj = 10000
    is_there_action_in_trajs = True
    ## Getting trajectory filename
    traj_filename = "C_Uniform_10000_trajectories_disjoint_KS_3D_STEERING_ANGLE_perturb_2.01_slack_0.0_seed_2025_grid_0.03_0.03_6.00deg_t3.01_ts0.2vrange_1.0_1.0_steering_13.pkl"
    with open(traj_filename, 'rb') as file:
        traj = pickle.load(file)
    ## Getting coverage visualization
    coverage_visualization(g, resolution, reachable_indices, sampled_trajectories_filename, num_traj, traj, is_there_action_in_trajs)

if __name__ == "__main__":
    main()