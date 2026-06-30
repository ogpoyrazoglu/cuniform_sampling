import pickle
import numpy as np
import math
from pprint import pprint
import os
import matplotlib.pyplot as plt
from classes.grid import Grid

def debug_visualize_3d(grid, level_set_idx, reachable_rep, trajectory_indices):
    """
    Plots a 3D scatter of:
      1) The reachable region for the given level set (blue points)
      2) The coverage from the actual trajectories (red points)

    Args:
        grid (Grid): the grid object used to get centers
        level_set_idx (int): which level set we are visualizing
        reachable_rep (set of tuples): the representative real coordinates for all
            reachable cells at this level set
        trajectory_indices (set of tuples): the unique grid indices hit by this level set
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert sets of reachable and coverage points into numpy arrays
    if len(reachable_rep) > 0:
        reachable_xyz = np.array(list(reachable_rep))
    else:
        reachable_xyz = np.empty((0,3))

    if len(trajectory_indices) > 0:
        coverage_centers = np.array([grid.get_grid_center(idx) for idx in trajectory_indices])
    else:
        coverage_centers = np.empty((0,3))

    # Convert them into sets of tuples in real space, so we can find overlap
    reachable_coords_set = {tuple(pt) for pt in reachable_xyz}
    coverage_coords_set = {tuple(pt) for pt in coverage_centers}

    # Intersection = points in both sets
    overlap_points = reachable_coords_set.intersection(coverage_coords_set)

    # Separate out unique-only sets (so we can color them differently)
    only_reachable = np.array([pt for pt in reachable_coords_set - overlap_points])
    only_coverage = np.array([pt for pt in coverage_coords_set - overlap_points])
    overlap_coords = np.array(list(overlap_points))
    # Plot: only reachable (blue)
    if only_reachable.size > 0:
        ax.scatter(
            only_reachable[:, 0],
            only_reachable[:, 1],
            only_reachable[:, 2],
            c='blue', marker='o', s=20, alpha=0.7,
            label='Reachable only'
        )

    # Plot: only coverage (red)
    if only_coverage.size > 0:
        ax.scatter(
            only_coverage[:, 0],
            only_coverage[:, 1],
            only_coverage[:, 2],
            c='red', marker='^', s=20, alpha=0.7,
            label='Coverage only'
        )

    # Plot: overlap (purple)
    if overlap_coords.size > 0:
        ax.scatter(
            overlap_coords[:, 0],
            overlap_coords[:, 1],
            overlap_coords[:, 2],
            c='purple', marker='s', s=40, alpha=0.9,
            label='Overlap'
        )

    ax.axis("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Theta")
    ax.set_title(f"Debug Visualization for Level Set {level_set_idx}")
    ax.legend()

    plt.show()

def add_folder_prefix(file_list, folder_prefix):
    return [os.path.join(folder_prefix, file_name) for file_name in file_list]

def load_trajectory_file(file_name):
    """Loads the trajectories from a pickle file."""
    with open(file_name, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories

def analyze_coverage(grid, trajectories, reachable_indices, only_count_reachable=True):
    """
    Analyzes the coverage by counting the number of unique grid indices
    hit by all level sets of each trajectory.
    Args:
        grid (Grid): The Grid object for indexing.
        trajectories (list): A list of trajectories, where each trajectory
                             is a list of states at each time step.
    Returns:
        int: The number of unique grid indices hit by all level sets.
    """
    unique_indices = set()
    
    for trajectory in trajectories:
        for i, state in enumerate(trajectory):
            grid_index = grid.get_index(state) # Get the grid index for the current state
            if only_count_reachable:
                if grid_index in reachable_indices[i]:
                    unique_indices.add(grid_index) # Add the grid index to the set of unique indices only if it's reachable
            else:
                unique_indices.add(grid_index) # Add the grid index to the set of unique indices
    return len(unique_indices)


def process_files(
        file_list, 
        grid, 
        type, 
        reachable_normalization=True, 
        total_reachable_cells_count=-1, 
        reachable_cells_count_list=[],
        reachable_cell_representatives=None,
        reachable_indices=None
    ):
    """
    Processes a list of pickle files for a single setting/type
    and returns:
      - total_coverage_percent: coverage percentage across all level sets combined
      - level_set_coverage_percentages: a list of coverage percentages for each level set

        if `reachable_normalization` parameter set to True, the coverage ratio of the trajectories will be normalized with respect to the 'reachable cells' count
    `reachable_cells_count` is simply put all reachable cells across all computed level set together, check how many unique cells they hit as a group.
    """
    for file_name in file_list:
        # Load the trajectory data from the file and do data processing
        trajectories = load_trajectory_file(file_name)
        if type == 'mppi':
            traj = trajectories['rollout_states_vis_list'][0] 
        elif type == 'c_uniform':
            traj = [[state for state, _ in trajectory] for trajectory in trajectories]
        else:
            print("WRONG TYPE")
            return -1
        
        only_count_reachable = True
        total_unique_grid_indices = analyze_coverage(grid, traj, reachable_indices, only_count_reachable) # Perform coverage analysis
        if reachable_normalization:
            denominator = total_reachable_cells_count
        else:
            denominator = 24000 # Or any other number
        total_coverage_percent = (total_unique_grid_indices / denominator) * 100

        
        # Calculate the coverage ratio as percentage
        # total_ratio = (total_unique_grid_indices / total_reachable_cells_count) * 100 if reachable_normalization else (total_unique_grid_indices / 24000) * 100

        # Output the result with ratio as a percentage rounded to 2 decimal places
        # print(
        #     f"File: {file_name} - Total unique grid indices hit across ALL level sets: {total_unique_grid_indices}, "
        #     f"Total Coverage Percentage = {round(total_ratio, 2)}%"
        # )
        print("  Per level-set coverage statistics:")
        n = len(traj[0])  # Length of each level set assuming all trajectories have the same number of time steps
        level_set_coverage_percentages = []
        for i in range(n):
            level_set_states = [trajectory[i] for trajectory in traj]  # Extract states for the i-th level set
            if only_count_reachable:
                unique_indices = {grid.get_index(state) for state in level_set_states if grid.get_index(state) in reachable_indices[i]}  
            else:
                unique_indices = set(grid.get_index(state) for state in level_set_states)  # Unique grid indices for this level set
            # if i == 2:
            #     print("unique indices: ")
            #     pprint(unique_indices)
            #     print("coverage only, not reachable: ")
            #     reachbale_cell_indices = set()
            #     for rep in reachable_cell_representatives[2]:
            #         reachbale_cell_indices.add(grid.get_index(rep))
            #     coverage_only_indices = unique_indices - reachbale_cell_indices
            #     print("coverage only indices: ", coverage_only_indices)
            #     coverage_only_states = [state for state in level_set_states if grid.get_index(state) in coverage_only_indices]
            #     for j, st in enumerate(coverage_only_states):
            #         print(f"coverage_only_states: {coverage_only_states[j]}, corresponding index: {grid.get_index(coverage_only_states[j])}")
            if i == 1:
                print("=" * 80)
                print(file_list)
                print(len(unique_indices))
                pprint(unique_indices)
                print("=" * 80)
            # level_set_coverage = min(len(unique_indices), reachable_cells_count_list[i])
            level_set_coverage = len(unique_indices)

            coverage_ratio = (level_set_coverage / reachable_cells_count_list[i]) * 100 if reachable_normalization else -1
            level_set_coverage_percentages.append(coverage_ratio)
            print(
                f"    Level Set {i:<2}: "
                f"Per-Level-Set Coverage Percentage = {round(coverage_ratio, 2):>5} %     "
                f"Reachable cells = {reachable_cells_count_list[i]:<10} "
                f"Unique grid indices = {level_set_coverage:<6} "
            )

            if coverage_ratio > 100.0 and reachable_cell_representatives is not None:
            # if reachable_cell_representatives is not None:
                print(f"    [DEBUG ALERT] Coverage ratio > 100% at level set {i}!")
                print("    Launching debug visualization for that level set...")
                debug_visualize_3d(
                    grid=grid,
                    level_set_idx=i,
                    reachable_rep=reachable_cell_representatives[i],  # 3D reps for that LS
                    trajectory_indices=unique_indices
                )
    return total_coverage_percent, level_set_coverage_percentages


def meta_process_and_visualize(settings_file_dict,
                               grid,
                               reachable_normalization=True,
                               total_reachable_cells_count=-1,
                               reachable_cells_count_list=[],
                               reachable_cell_representatives=None,
                               reachable_indices=None,
                            ):
    """
    Meta-function that:
      1. Iterates over multiple settings (key) and their file lists (value).
      2. Processes coverage for each setting (combining all its files).
      3. Creates a grouped bar chart comparing coverage across settings,
         level set by level set.

    Args:
        settings_file_dict (dict): 
            Example:
            {
                'mppi_high_var': ['trajectories_0_10000_0.3.pickle', 'trajectories_1_10000_0.3.pickle'],
                'c_uniform_no_slack': ['C_Uniform_10000_trajectories_disjoint_...'],
                ...
            }
        grid (Grid): The Grid object for indexing
        reachable_normalization (bool): Whether to normalize coverage by reachable cells
        total_reachable_cells_count (int): total reachable cells across all level sets
        reachable_cells_count_list (List[int]): per-level-set reachable cells
    """
    settings = list(settings_file_dict.keys())

    # We'll store the coverage percentages in dictionaries keyed by setting
    total_coverage_by_setting = {}
    per_level_coverage_by_setting = {}

    # 1. For each setting, compute coverage
    for setting_name, (setting_type, file_list) in settings_file_dict.items():
        # setting_type is either 'mppi' or 'c_uniform'
        # file_list is the actual list of files for that setting.
        total_cov, level_cov_list = process_files(
            file_list=file_list,
            grid=grid,
            type=setting_type,
            reachable_normalization=reachable_normalization,
            total_reachable_cells_count=total_reachable_cells_count,
            reachable_cells_count_list=reachable_cells_count_list,
            reachable_cell_representatives=reachable_cell_representatives,
            reachable_indices=reachable_indices
        )
        total_coverage_by_setting[setting_name] = total_cov
        per_level_coverage_by_setting[setting_name] = level_cov_list

        # Print numeric results:
        print(f"=== Setting: {setting_name} ===")
        print(f"  --> Total coverage percentage: {round(total_cov, 2)}%")
        for i, lc in enumerate(level_cov_list):
            print(f"     Level Set {i} coverage: {round(lc, 2)}%")

    # 2. Visualization: side-by-side bar chart for per-level-set coverage
    #    assume all settings have the same # of level sets.
    # Get the number of level sets from the first setting
    example_setting = settings[0]
    num_level_sets = len(per_level_coverage_by_setting[example_setting])

    # Prepare data for plotting
    x_indices = np.arange(num_level_sets)  # e.g. 0,1,2,... for each level set
    width = 0.8 / len(settings)            # distribute bar width evenly among settings

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']  # Define some marker styles

    for idx, setting_name in enumerate(settings):
        coverage_list = per_level_coverage_by_setting[setting_name]
        
        # Use line plot with markers
        ax.plot(
            x_indices,
            coverage_list,
            marker=markers[idx % len(markers)],  # Cycle through markers
            label=setting_name,
            linewidth=1.5
        )

    ax.set_xlabel("Level Set Index")
    ax.set_ylabel("Coverage Percentage (%)")
    ax.set_title("Per-Level-Set Coverage Comparison")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(i) for i in range(num_level_sets)])
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    ''' # bar plots below, a bit crowded 
    for idx, setting_name in enumerate(settings):
        coverage_list = per_level_coverage_by_setting[setting_name]
        # coverage_list_for_plot = coverage_list[1:] # skip the initial level set because it's boring

        # shift each setting's bars horizontally
        offset = idx * width
        ax.bar(
            x_indices + offset,
            coverage_list,
            # coverage_list_for_plot,
            width,
            label=setting_name
        )

    ax.set_xlabel("Level Set Index")
    ax.set_ylabel("Coverage Percentage (%)")
    ax.set_title("Per-Level-Set Coverage Comparison")
    ax.set_xticks(x_indices + width*(len(settings)-1)/2, labels=[str(i) for i in range(num_level_sets)])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    '''

def count_reachable_cells(filename):
    ''' use uc to denote unique cells count of that set
    2 ways to count unique reachable cells:
        one way is uc(L0) + uc(L1) + uc(L2) ... + uc(Ln), so overlapping cells in different level set will be counted multiple times.
        the second way is uc(union(L0, L1, L2, ..., Ln)
    Since the paper version used the second way, here I am proceeding with the second way
    '''
    print(f"Reading reachable cells data from file: {filename}")
    reachable_cells_list = []
    with open(filename, "rb") as f:
        data = pickle.load(f)
    ReaBoxIndices_LSs  = data['reachable_indicex_across_LS']
    total_set = set()
    for i in range(len(ReaBoxIndices_LSs)):
        total_set.update(ReaBoxIndices_LSs[i]) 
        reachable_cells_list.append(len(ReaBoxIndices_LSs[i]))
    return len(total_set), reachable_cells_list

def main():
    # Ground truth and neural trajectory files
    # gt_file = "C_Uniform_feasible_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_slack_0.0_seed_2025_grid_0.100_0.100_6.000deg_t1.41_ts0.2_vrange_2.0_2.0_steer_range_-30.0_30.0_steering_31.pkl"
    gt_file = "C_Uniform_10000_trajectories_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_slack_0.0_seed_2025_grid_0.100_0.100_6.000deg_t1.41_ts0.2_vrange_2.0_2.0_steer_range_-30.0_30.0_steering_31.pkl"
    supervised_neural_file = "C_Uniform_neural_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_slack_0.0_seed_2025_grid_0.100_0.100_6.000deg_t1.41_ts0.2_vrange_2.0_2.0_steer_range_-30.0_30.0_steering_31.pkl"
    uniform_action_file = "/home/mikasa/RSN/traj_sampling/flow_Cuniform/uniform_sampled_actions_trajectories_10000.pickle"
    
    # Load the configuration from ground truth file
    with open(gt_file, 'rb') as f:
        trajectories = pickle.load(f)
    
    # Load the original data file for configuration
    original_data_file = "C_Uniform_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_slack_0.0_seed_2025_grid_0.100_0.100_6.000deg_t1.41_ts0.2_vrange_2.0_2.0_steer_range_-30.0_30.0_steering_31.pkl"
    with open(original_data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Define the grid used for the analysis
    g = Grid(thresholds=data["config"]["thresholds"])
    reachable_cell_representatives = data["reachable_representative_across_LS"]
    thresholds = g.thresholds
    print(f"Grid size is {thresholds}")
    
    # Get reachable cells count
    total_reachable_cells_count, reachable_cells_count_list = count_reachable_cells(filename=original_data_file)
    print("Total reachable cells: ", total_reachable_cells_count)
    print("reachable cells list: ", reachable_cells_count_list)
    
    # Setup comparison dictionary
    settings_file_dict = {
        "Ground Truth": ("c_uniform", [gt_file]),
        "Supervised Neural Network": ("c_uniform", [supervised_neural_file]),
        "Uniform Action": ("c_uniform", [uniform_action_file])
    }
    
    # Run comparison analysis
    meta_process_and_visualize(
        settings_file_dict=settings_file_dict,
        grid=g,
        reachable_normalization=True,
        total_reachable_cells_count=total_reachable_cells_count,
        reachable_cells_count_list=reachable_cells_count_list,
        reachable_cell_representatives=reachable_cell_representatives,
        reachable_indices=data["reachable_indicex_across_LS"]
    )

if __name__ == "__main__":
    main()