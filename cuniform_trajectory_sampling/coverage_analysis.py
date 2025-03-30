import pickle
import numpy as np
import math
from pprint import pprint
import os
import matplotlib.pyplot as plt
from classes.grid import Grid

def add_folder_prefix(file_list, folder_prefix):
    return [os.path.join(folder_prefix, file_name) for file_name in file_list]

def load_trajectory_file(file_name):
    """Loads the trajectories from a pickle file."""
    with open(file_name, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories

def analyze_coverage(grid, trajectories):
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
        for state in trajectory:
            grid_index = grid.get_index(state) # Get the grid index for the current state
            unique_indices.add(grid_index) # Add the grid index to the set of unique indices
    
    return len(unique_indices)


def process_files(
        file_list, 
        grid, 
        type, 
        reachable_normalization=False, 
        total_reachable_cells_count=-1, 
        reachable_cells_count_list=[],
    ):
    """
    Processes a list of pickle files and outputs coverage analysis results.
        if `reachable_normalize` parameter set to True, the coverage ratio of the trajectories will be normalized with respect to the 'reachable cells' count
    """
    for file_name in file_list:
        # Load the trajectory data from the file and do data processing
        trajectories = load_trajectory_file(file_name)
        if type == 'mppi':
            traj = trajectories['rollout_states_vis_list'][0] 
        elif type == 'c_uniform':
            # traj = [[state for state, _ in trajectory[:11]] for trajectory in trajectories] # make sure 2s long
            traj = [[state for state, _ in trajectory] for trajectory in trajectories]
        else:
            print("WRONG TYPE")
            return -1
        
        total_unique_grid_indices = analyze_coverage(grid, traj) # Perform coverage analysis
        
        # Calculate the coverage ratio as percentage
        total_ratio = (total_unique_grid_indices / total_reachable_cells_count) * 100 if reachable_normalization else (total_unique_grid_indices / 24000) * 100
        # print(f"File: {file_name} - Number of unique grid indices hit across ALL level sets: {total_unique_grid_indices}, ratio={round(total_ratio, 2)}%")

        # Output the result with ratio as a percentage rounded to 2 decimal places
        print(
            f"File: {file_name} - Total unique grid indices hit across ALL level sets: {total_unique_grid_indices}, "
            f"Total Coverage Percentage = {round(total_ratio, 2)}%"
        )
        print("  Per level-set coverage statistics:")
        n = len(traj[0])  # Length of each level set assuming all trajectories have the same number of time steps
        for i in range(n):
            level_set_states = [trajectory[i] for trajectory in traj]  # Extract states for the i-th level set
            unique_indices = set(grid.get_index(state) for state in level_set_states)  # Unique grid indices for this level set
            level_set_coverage = len(unique_indices)

            coverage_ratio = (level_set_coverage / reachable_cells_count_list[i]) * 100 if reachable_normalization else -1
            print(
                f"    Level Set {i:<2}: "
                f"Per-Level-Set Coverage Percentage = {round(coverage_ratio, 2):>5} %     "
                f"Reachable cells = {reachable_cells_count_list[i]:<10} "
                f"Unique grid indices = {level_set_coverage:<6} "
            )


def meta_process_and_visualize(settings_file_dict,
                               grid,
                               reachable_normalization=True,
                               total_reachable_cells_count=-1,
                               reachable_cells_count_list=[],
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

def count_reachable_cells(filename):
    ''' use uc to denote unique cells count of that set
    2 ways to count unique reachable cells:
        one way is uc(L0) + uc(L1) + uc(L2) ... + uc(Ln), so overlapping cells in different level set will be counted multiple times.
        the second way is uc(union(L0, L1, L2, ..., Ln)
    Here I am proceeding with the second way
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
    # List of pickle files to process
    file_list_mppi_2s = [
        # For 250 sampled trajectories
        "trajectories_0_250_0.03.pickle",  # Low variance (0.03)
        "trajectories_0_250_0.1.pickle",   # Medium variance (0.1)
        "trajectories_0_250_0.3.pickle",   # High variance (0.3)
        "trajectories_1_250_0.03.pickle",  # Low variance (log-MPPI)
        "trajectories_1_250_0.1.pickle",   # Medium variance (log-MPPI)
        "trajectories_1_250_0.3.pickle",   # High variance (log-MPPI)

        # For 500 sampled trajectories
        "trajectories_0_500_0.03.pickle",  # Low variance (0.03)
        "trajectories_0_500_0.1.pickle",   # Medium variance (0.1)
        "trajectories_0_500_0.3.pickle",   # High variance (0.3)
        "trajectories_1_500_0.03.pickle",  # Low variance (log-MPPI)
        "trajectories_1_500_0.1.pickle",   # Medium variance (log-MPPI)
        "trajectories_1_500_0.3.pickle",   # High variance (log-MPPI)

        # For 1000 sampled trajectories
        "trajectories_0_1000_0.03.pickle", # Low variance (0.03)
        "trajectories_0_1000_0.1.pickle",  # Medium variance (0.1)
        "trajectories_0_1000_0.3.pickle",  # High variance (0.3)
        "trajectories_1_1000_0.03.pickle", # Low variance (log-MPPI)
        "trajectories_1_1000_0.1.pickle",  # Medium variance (log-MPPI)
        "trajectories_1_1000_0.3.pickle",  # High variance (log-MPPI)

        # For 2500 sampled trajectories
        "trajectories_0_2500_0.03.pickle", # Low variance (0.03)
        "trajectories_0_2500_0.1.pickle",  # Medium variance (0.1)
        "trajectories_0_2500_0.3.pickle",  # High variance (0.3)
        "trajectories_1_2500_0.03.pickle", # Low variance (log-MPPI)
        "trajectories_1_2500_0.1.pickle",  # Medium variance (log-MPPI)
        "trajectories_1_2500_0.3.pickle",  # High variance (log-MPPI)

        # For 5000 sampled trajectories
        "trajectories_0_5000_0.03.pickle", # Low variance (0.03)
        "trajectories_0_5000_0.1.pickle",  # Medium variance (0.1)
        "trajectories_0_5000_0.3.pickle",  # High variance (0.3)
        "trajectories_1_5000_0.03.pickle", # Low variance (log-MPPI)
        "trajectories_1_5000_0.1.pickle",  # Medium variance (log-MPPI)
        "trajectories_1_5000_0.3.pickle",  # High variance (log-MPPI)

        # For 10000 sampled trajectories
        "trajectories_0_10000_0.03.pickle", # Low variance (0.03)
        "trajectories_0_10000_0.1.pickle",  # Medium variance (0.1)
        "trajectories_0_10000_0.3.pickle",  # High variance (0.3)
        "trajectories_1_10000_0.03.pickle", # Low variance (log-MPPI)
        "trajectories_1_10000_0.1.pickle",  # Medium variance (log-MPPI)
        "trajectories_1_10000_0.3.pickle"   # High variance (log-MPPI)
    ]
    # file_list_mppi_high_var = [ # 4 seconds long trajectories
    #     "trajectories_0_10000_0.3_timeHorizon_4.pickle",
    # ]
    # file_list_mppi_low_var = [ # 4 seconds long trajectories
    #     "trajectories_0_10000_0.1_timeHorizon_4.pickle",
    # ]
    # file_list_log_mppi_high_var = [ # 4 seconds long trajectories
    #     "trajectories_1_10000_0.3_timeHorizon_4.pickle",
    #     # "trajectories_1_10000_0.2_timeHorizon_4.pickle"
    # ]
    # file_list_log_mppi_low_var = [ # 4 seconds long trajectories
    #     "trajectories_1_10000_0.1_timeHorizon_4.pickle",
    # ]

    # Process each file and analyze coverage
    folder_prefix = 'mppi_trajs'
    file_list_mppi_with_prefix = add_folder_prefix(file_list_mppi_2s, folder_prefix)
    # file_list_mppi_high_var_with_prefix = add_folder_prefix(file_list_mppi_high_var, folder_prefix)
    # file_list_mppi_low_var_with_prefix = add_folder_prefix(file_list_mppi_low_var, folder_prefix)
    # file_list_log_mppi_high_var_with_prefix = add_folder_prefix(file_list_log_mppi_high_var, folder_prefix)
    # file_list_log_mppi_low_var_with_prefix = add_folder_prefix(file_list_log_mppi_low_var, folder_prefix)

    file_list_c_uniform_NF_no_slack = [
        "C_Uniform_250_trajectories_overlapping_DUBINS_perturb_2.01_slack_0.0_seed_2024_grid_0.050_0.050_9.000deg_t2.01_ts0.2_vrange_1.0_1.0_steering_5.pkl",
        "C_Uniform_500_trajectories_overlapping_DUBINS_perturb_2.01_slack_0.0_seed_2024_grid_0.050_0.050_9.000deg_t2.01_ts0.2_vrange_1.0_1.0_steering_5.pkl",
        "C_Uniform_1000_trajectories_overlapping_DUBINS_perturb_2.01_slack_0.0_seed_2024_grid_0.050_0.050_9.000deg_t2.01_ts0.2_vrange_1.0_1.0_steering_5.pkl",
        "C_Uniform_2500_trajectories_overlapping_DUBINS_perturb_2.01_slack_0.0_seed_2024_grid_0.050_0.050_9.000deg_t2.01_ts0.2_vrange_1.0_1.0_steering_5.pkl",
        "C_Uniform_5000_trajectories_overlapping_DUBINS_perturb_2.01_slack_0.0_seed_2024_grid_0.050_0.050_9.000deg_t2.01_ts0.2_vrange_1.0_1.0_steering_5.pkl",
        "C_Uniform_10000_trajectories_overlapping_DUBINS_perturb_2.01_slack_0.0_seed_2024_grid_0.050_0.050_9.000deg_t2.01_ts0.2_vrange_1.0_1.0_steering_5.pkl",
    ]
    file_list_uniform_sample_action = [
        "uniform_sampled_actions_trajectories_10000.pickle"
    ]

    # settings_file_dict = {
    #     "MPPI(var=0.3)": ("mppi", file_list_mppi_high_var_with_prefix),
    #     "MPPI(var=0.1)": ("mppi", file_list_mppi_low_var_with_prefix), 
    #     "Uniform Action": ("c_uniform", file_list_uniform_sample_action),
    #     "Log MPPI(var=0.3)": ("mppi", file_list_log_mppi_high_var_with_prefix),
    #     "Log MPPI(var=0.1)": ("mppi", file_list_log_mppi_low_var_with_prefix),
    #     "NF_no_slack": ("c_uniform", file_list_c_uniform_NF_no_slack),
    # }

    filename_for_reading_config = "##########################PLACEHOLDER##########################"
    filename_for_reading_config = "C_Uniform_processed_overlapping_DUBINS_perturb_2.01_slack_0.0_seed_2024_grid_0.050_0.050_9.000deg_t2.01_ts0.2_vrange_1.0_1.0_steering_5.pkl"
    with open(filename_for_reading_config, 'rb') as f:
        data = pickle.load(f)

    # Define the grid used for the analysis
    g = Grid(thresholds=data["config"]["thresholds"])
    thresholds = g.thresholds
    print(f"Grid size is {thresholds}")

    total_reachable_cells_count, reachable_cells_count_list = count_reachable_cells(filename=filename_for_reading_config)
    print("Total reachable cells: ", total_reachable_cells_count)
    print("reachable cells list: ", reachable_cells_count_list)

    # meta_process_and_visualize(
    #     settings_file_dict=settings_file_dict,
    #     grid=g,
    #     reachable_normalization=True,
    #     total_reachable_cells_count=total_reachable_cells_count,
    #     reachable_cells_count_list=reachable_cells_count_list,
    # )
    # return

    print()
    print("=" * 80)
    print("MPPI trajectories below, 0 means vanilla mppi, 1 means log-mppi")
    # e.g. trajectories_1_250_0.1.pickle would mean log mppi with 250 trajectories, 0.1 variance
    print("=" * 80)
    process_files(file_list_mppi_with_prefix, 
                  g,
                  'mppi',
                  reachable_normalization=False, 
                  total_reachable_cells_count=total_reachable_cells_count, 
                  reachable_cells_count_list=reachable_cells_count_list
                )

    print()
    print("=" * 80)
    print("C_uniform coverage stats below")
    print("=" * 80)
    process_files(file_list_c_uniform_NF_no_slack,
                  g,
                  'c_uniform',
                  reachable_normalization=False,
                  total_reachable_cells_count=total_reachable_cells_count,
                  reachable_cells_count_list=reachable_cells_count_list,
                )
    print()
    print("=" * 80)
    print("uniform action sampling coverage stats below")
    print("=" * 80)
    process_files(file_list_uniform_sample_action,
                  g,
                  'c_uniform',
                  reachable_normalization=False,
                  total_reachable_cells_count=total_reachable_cells_count,
                  reachable_cells_count_list=reachable_cells_count_list,
                )
    return

if __name__ == "__main__":
    main()