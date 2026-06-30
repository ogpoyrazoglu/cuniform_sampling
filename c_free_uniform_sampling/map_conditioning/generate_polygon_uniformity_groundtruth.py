import os
import sys
import numpy as np
import pickle
import json
import yaml
import time
import math
import contextlib
import gc
import torch # Imported for CUDA memory management if available

# Visualization and Geometry Libraries
# Set backend early to prevent GUI conflicts
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation

# =============================================================================
# Configuration Parameters
# =============================================================================
# Define parameters for the dataset generation
NUM_POSITIONS_PER_ENV = 1      # Configurable: Number of valid positions (scenarios) to generate per environment
START_ENV_ID = 0
END_ENV_ID = 299               # Process environments 0 through 299 inclusive
MIN_CLEARANCE_METERS = 0.4     # Minimum clearance from obstacles for sampling poses

LOCAL_GRID_SIZE = 121          # Size of the local robot-centric costmap/SDF
LIDAR_MAX_RANGE = 3.0          # Max range for local LiDAR simulation
LIDAR_NUM_BEAMS = 1440          # Number of beams for LiDAR simulation
LOCAL_RESOLUTION = 0.05        # Resolution of the local costmap/SDF

SUPPRESS_DETAILED_OUTPUT = True # Suppress detailed logs from the reachability library
INFLATION_RADIUS = 1           # Inflation radius in the unit of cells

# =============================================================================
# 1. Setup and Path Handling
# =============================================================================

def safe_convert_from_serializable(obj):
    """Convert serialized objects back to their original format."""
    if isinstance(obj, dict):
        if obj.get('__numpy_array__', False):
            return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
        else:
            return {k: safe_convert_from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_from_serializable(item) for item in obj]
    else:
        return obj

def load_json_data(json_path):
    """Load data from JSON file and convert back to original format."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return safe_convert_from_serializable(data)

def setup_paths():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        print("Warning: Running in an environment where __file__ is not defined. Assuming CWD is map_conditioning.")
        script_dir = os.getcwd()

    project_root = os.path.dirname(script_dir)
        
    if project_root not in sys.path:
        sys.path.append(project_root)
            
    return script_dir, project_root

SCRIPT_DIR, PROJECT_ROOT = setup_paths()


# =============================================================================
# 2. Imports from flow_Cuniform (External Dependency)
# =============================================================================

from flow_Cuniform.dynamics_helpers import (
    dynamics_KS_3d_steering_angle,
    inverse_dynamics_KS_3d_steering_angle,
    vectorized_dynamics_KS_3d_steering_angle,
)
from flow_Cuniform.utility_helpers import (
    generate_actions,
    calculate_reachable_level_sets,
    precompute_graph_structure_parallel,
    prune_graph,
)
from flow_Cuniform.c_uniform_sampling import (
    parallelized_collision_checker_nodes,
)
from utility_helper_map import (
    save_level_set_visualization,
    visualize_sdf,
)
from barn_dataset_helpers import (
    visualize_barn_with_robot_area,
    _check_clearance_in_grid,
    simulate_lidar_in_barn,
    create_robot_centered_costmap,
)
FLOW_CUNIFORM_AVAILABLE = True

# =============================================================================
# 3. Helper Functions (Adapted from various modules)
# =============================================================================
# Global Map (World): Cartesian (Y-up). Origin defined by 'origin' (e.g., bottom-left).
# Global/Local Grids: Image (Y-down). Numpy index (0,0) is top-left. (Matches barn_dataset_helpers.py)

# -----------------------------------------------------------------------------
# 3.1 Global Map Generation (Adapted from geometry_utils.py)
# -----------------------------------------------------------------------------

def rasterize_polygon_interior(vertices, map_size, resolution, origin):
    """
    Rasterizes a polygon into a global occupancy grid.
    Input: World Coords (Y-up). Output: Image Grid (Y-down indexing)
    """
    width, height = map_size
    # Start with a fully occupied grid in Image coordinates (Y-down) for CV2
    image_grid = np.ones((height, width), dtype=np.uint8)

    # Helper to convert World (Y-up) to CV2 Image (Y-down)
    def world_to_image_coords(pt):
        x, y = pt
        # 1. Convert world meters to grid indices (Cartesian Y-up)
        gx = (x - origin[0]) / resolution
        gy = (y - origin[1]) / resolution
        # 2. Convert Cartesian to Image indices (Flip Y-axis)
        ix = int(round(gx))
        iy = int(round(height - 1 - gy))
        return ix, iy

    # Convert vertices
    image_coords = np.array([world_to_image_coords(v) for v in vertices], dtype=np.int32)

    if image_coords.size > 0:
         # Robustness: Clip coordinates to prevent out-of-bounds access
        image_coords[:, 0] = np.clip(image_coords[:, 0], 0, width - 1)
        image_coords[:, 1] = np.clip(image_coords[:, 1], 0, height - 1)
        
        # Ensure contiguous memory (Robustness for CV2)
        if not image_coords.flags['C_CONTIGUOUS']:
             image_coords = np.ascontiguousarray(image_coords)

        # Fill the interior with FREE space (0)
        try:
            cv2.fillPoly(image_grid, [image_coords.reshape((-1, 1, 2))], 0)
        except Exception as e:
            print(f"Warning: cv2.fillPoly failed: {e}. Returning fully occupied map.")
            return np.ones((height, width), dtype=np.int8)

    # The image_grid is already in Image coordinates (Y-down indexing), 
    # which is what BARN helper functions expect for the input grid.

    # Ensure boundary cells are marked as obstacles
    if height > 1 and width > 1:
        image_grid[0, :] = 1; image_grid[-1, :] = 1
        image_grid[:, 0] = 1; image_grid[:, -1] = 1
    return image_grid.astype(np.int8)

# -----------------------------------------------------------------------------
# 3.2 Sampling (Using functions from barn_dataset_helpers.py)
# -----------------------------------------------------------------------------
def sample_robot_positions_in_polygon(global_grid, num_positions, resolution, origin, min_clearance):
    """
    Sample valid robot positions in polygon grid using BARN clearance checking.
    Input: Image Grid (Y-down). Output: World Coords (Y-up).
    
    This function samples only from free space within the polygon and uses
    proven clearance checking from barn_dataset_helpers.py
    """
    clearance_cells = int(np.ceil(min_clearance / resolution))
    height, width = global_grid.shape
    
    print(f"     Polygon sampling: {height}×{width} grid, origin={origin}, resolution={resolution}")
    print(f"     World bounds: X=[{origin[0]:.3f}, {origin[0] + width*resolution:.3f}], Y=[{origin[1]:.3f}, {origin[1] + height*resolution:.3f}]")
    
    # Find all free cells (0 = free space). np.where returns (row_indices, col_indices)
    free_y, free_x = np.where(global_grid == 0)
    free_space_cells = len(free_x)
    total_cells = height * width
    
    print(f"     Free space: {free_space_cells}/{total_cells} cells ({100*free_space_cells/total_cells:.1f}%)")
    if free_space_cells == 0:
        print("     ERROR: No free space found in polygon grid!")
        return []
    
    valid_positions = []
    max_attempts = min(free_space_cells, num_positions * 500)  # More attempts for better success rate
    attempts = 0
    
    # Sample from free cells only
    while len(valid_positions) < num_positions and attempts < max_attempts:
        attempts += 1
        
        # Randomly select a free cell
        free_idx = np.random.randint(0, free_space_cells)
        grid_x, grid_y = free_x[free_idx], free_y[free_idx]
        
        # Use BARN clearance checking function
        if _check_clearance_in_grid(global_grid, grid_x, grid_y, clearance_cells):
            # Convert grid coordinates (Image Y-down) to polygon world coordinates (Y-up)
            world_x = origin[0] + grid_x * resolution
            world_y = origin[1] + (height - 1 - grid_y) * resolution
            world_yaw = np.random.uniform(0, 2 * np.pi)
            
            print(f"       Valid position {len(valid_positions)+1}: grid=({grid_x}, {grid_y}) -> world=({world_x:.3f}, {world_y:.3f})")
            valid_positions.append((world_x, world_y, world_yaw))
    
    print(f"     Found {len(valid_positions)} valid positions in {attempts} attempts")
    return valid_positions


# -----------------------------------------------------------------------------
# 3.3 Perception Simulation (Using functions from barn_dataset_helpers.py)
# -----------------------------------------------------------------------------
def create_sdf_from_costmap(costmap, resolution):
    """
    Compute the SDF from a costmap. Input: Image Grid (Y-down). Output: SDF (Y-down).
    """
    # Ensure costmap is boolean
    costmap_bool = costmap > 0.5
    
    # Handle edge cases for stability
    if costmap_bool.all():
        # Fully occupied
        dist_out = np.zeros_like(costmap_bool, dtype=float)
        dist_in = distance_transform_edt(costmap_bool) * resolution
    elif (~costmap_bool).all():
         # Fully free
         dist_out = distance_transform_edt(~costmap_bool) * resolution
         dist_in = np.zeros_like(costmap_bool, dtype=float)
    else:
        # Standard case
        dist_out = distance_transform_edt(~costmap_bool) * resolution
        dist_in = distance_transform_edt(costmap_bool) * resolution

    sdf = dist_out.copy()
    sdf[costmap_bool] = -dist_in[costmap_bool]
    return sdf

# =============================================================================
# 4. Configuration Loading and Initialization
# =============================================================================

# Hardcoded Model Configuration (from supervised_dataset_generation.py)
KS_3D_STEERING_ANGLE_CONFIG = {
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
    "thresholds": [0.10, 0.10, (2 * math.pi)/60],
    "state_dim": 3,
    "dt": 0.10,
    "total_t": 2.41,
    # Robot always starts at origin (0,0,0) in the LOCAL frame
    "initial_state_set": np.array([[0.0, 0.0, 0.0]]),
    "adaptive_uniformity": False,
    "sdf_inflation": 0.0,
}

def initialize_model_config(config):
    if FLOW_CUNIFORM_AVAILABLE and generate_actions:
        config["actions"] = generate_actions(
            config["arange"], config["num_a"],
            config["steering_angle_range"],
            config["num_steering_angle"], deg2rad_conversion=True
        )
    return config

MODEL_CONFIG = initialize_model_config(KS_3D_STEERING_ANGLE_CONFIG.copy())

# Configuration Loading Helpers
def load_experiment_config():
    config_path = os.path.join(PROJECT_ROOT, "map_conditioning", "navigation_experiments", "configs", "experiment_config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

def get_polygon_config(experiment_config):
    """Extracts polygon configuration and resolves the dataset path."""
    try:
        poly_config = experiment_config['experiment_suite']['polygon_config']
        relative_path = poly_config['dataset_path']
        
        # Construct absolute path (preferring relative to navigation_experiments)
        nav_exp_dir = os.path.join(PROJECT_ROOT, "map_conditioning", "navigation_experiments")
        absolute_path = os.path.join(nav_exp_dir, relative_path)
        
        # Check for JSON exported version first
        exported_path = absolute_path.replace("A_Star_Generated", "Exported")
        if os.path.exists(exported_path):
            print(f"Using exported JSON dataset: {exported_path}")
            return exported_path, poly_config
        
        if not os.path.exists(absolute_path):
             absolute_path_fallback = os.path.join(PROJECT_ROOT, relative_path)
             if os.path.exists(absolute_path_fallback):
                 absolute_path = absolute_path_fallback
             else:
                 print(f"Warning: Dataset path not found. Tried {absolute_path} and {absolute_path_fallback}")

        return absolute_path, poly_config
    except KeyError:
        print("Error: Could not find polygon configuration.")
        return None, None

def conditional_stdout_suppress():
    """Context manager that suppresses stdout only if SUPPRESS_DETAILED_OUTPUT is True"""
    if SUPPRESS_DETAILED_OUTPUT:
        return contextlib.redirect_stdout(open(os.devnull, 'w'))
    else:
        return contextlib.nullcontext()

# =============================================================================
# 5. Core Logic - Reachability Analysis
# =============================================================================
def compute_pruned_level_sets_for_environment(config, sdf, resolution):
    """ 
    Computes the PRUNED (Safe) reachable Level Sets for a given (local) environment SDF.
    This involves reachability analysis, graph construction, and pruning unsafe states.
    """
    total_start_time = time.time()
    print("     Starting C-Safe computation (Reachability + Pruning)...")
    config_for_computation = config.copy()

    # Define the collision checker lambda function
    collision_checker = lambda *args, **kwargs: parallelized_collision_checker_nodes(
        *args, resolution=resolution, sdf_inflation=config_for_computation['sdf_inflation'], **kwargs
    )

    # Perform Reachability Analysis
    with conditional_stdout_suppress():
        # Call the core reachability function
        ReaBoxIndices_LSs, ReaBox_LSs, _, _ = calculate_reachable_level_sets(
            config_for_computation,
            disjoint_level_set=True,
            adaptive_resolution=True,
            obstacles=None, # Environment defined by SDF
            sdf=sdf,
            collision_checker_nodes=collision_checker,
        )
    if ReaBox_LSs is None or not ReaBox_LSs:
        print("      Reachability analysis returned empty sets.")
        return None

    # Step 2: Build graph structure (Connectivity Phase)
    with conditional_stdout_suppress():
        raw_graphs, _ = precompute_graph_structure_parallel(ReaBox_LSs, ReaBoxIndices_LSs, config_for_computation)

    # Step 3: Prune graphs (Safety Phase)
    with conditional_stdout_suppress():
        pruned_graphs = prune_graph(raw_graphs)

    # Step 4: Extract pruned level set representatives
    pruned_level_set_representatives = []
    for i in range(len(pruned_graphs)):
        # Extract representatives belonging to level 'i' within graph 'i'
        representatives = {node.point for node in pruned_graphs[i]["nodes"].values() if node.level_set == i}
        pruned_level_set_representatives.append(list(representatives))

    # Add the final level set (Level T)
    if len(pruned_graphs) > 0:
        last_level = len(pruned_graphs)
        # Extract representatives belonging to the final level 'T' within the last graph (T-1)
        last_level_set = {
            node.point 
            for node in pruned_graphs[-1]["nodes"].values()
            if node.level_set == last_level
        }
        pruned_level_set_representatives.append(list(last_level_set))

    # Step 5: Validation - Ensure the last level set is not empty (meaning safe trajectories exist)
    if len(pruned_level_set_representatives) == 0 or not pruned_level_set_representatives[-1]:
        print("     ❌ Pruning resulted in an empty final level set. No safe trajectories exist.")
        return None

    total_time = time.time() - total_start_time
    print(f"     ✓ C-Safe computation complete ({total_time:.2f}s). Levels: {len(pruned_level_set_representatives)}")
    return pruned_level_set_representatives

# =============================================================================
# 6. Dataset Generation Loop
# =============================================================================
def main():
    """ Main function implementing the localized dataset generation workflow  """
    print("="*80)
    print("POLYGON DATASET C-SAFE UNIFORMITY GROUND TRUTH GENERATION (LOCALIZED)")
    print("="*80)

    if not FLOW_CUNIFORM_AVAILABLE:
        print("Execution halted because flow_Cuniform library is missing.")
        return

    # Load configuration
    experiment_config = load_experiment_config()
    if not experiment_config: return

    dataset_path, poly_config = get_polygon_config(experiment_config)
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"Error: Dataset path invalid or does not exist: {dataset_path}")
        return

    # Extract Global Parameters from Config
    file_prefix = poly_config.get('file_prefix', '25_')
    file_suffix = poly_config.get('file_suffix', '_coords.pkl')
    scale_factor = poly_config.get('polygon_scale_factor', 2.5)
    global_map_size_cells = tuple(poly_config.get('map_size_cells', (100, 100)))
    # Use global resolution from config for the global map rasterization and sampling
    global_resolution = poly_config.get('map_resolution', 0.05)
    global_origin = poly_config.get('map_origin', [0.0, 0.0])

    # Define Output Directory
    # output_base_dir = os.path.join(SCRIPT_DIR, f"dataset_polygon_uniformity_localized_gt_dt") #defualt v=2.5 dt=0.2
    # output_base_dir = os.path.join(SCRIPT_DIR, f"dataset_polygon_uniformity_localized_gt_dt0.1")
    # output_base_dir = os.path.join(SCRIPT_DIR, f"dataset_polygon_uniformity_localized_gt_v_1.25_dt0.2")
    # output_base_dir = os.path.join(SCRIPT_DIR, f"dataset_polygon_uniformity_localized_gt_v_1.25_dt0.1")
    output_base_dir = os.path.join(SCRIPT_DIR, f"dataset_polygon_uniformity_localized_gt_v_1.25_dt0.1_t2.41")
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Output Directory: {output_base_dir}")
    print(f"Processing Environments: {START_ENV_ID} to {END_ENV_ID} | Positions per Env: {NUM_POSITIONS_PER_ENV}")
    print(f"Global Config: Res={global_resolution}, Scale={scale_factor}")
    print(f"Local Config: Size={LOCAL_GRID_SIZE}x{LOCAL_GRID_SIZE}, Res={LOCAL_RESOLUTION}")
    print("="*80)

    total_valid_positions = 0
    environments_to_process = range(START_ENV_ID, END_ENV_ID + 1)

    # Loop through environment IDs
    for env_id in environments_to_process:
        print(f"\n--- Processing Environment ID: {env_id} ({environments_to_process.index(env_id)+1}/{len(environments_to_process)}) ---")
        current_clearance = MIN_CLEARANCE_METERS  # Reset clearance for each environment

        # 1. Load Global Polygon Data
        env_filename = f"{file_prefix}{env_id}{file_suffix}"
        env_filepath = os.path.join(dataset_path, env_filename)
        
        # Check if we're using JSON format
        if dataset_path.endswith("Exported"):
            env_filename = env_filename.replace(".pkl", ".json")
            env_filepath = os.path.join(dataset_path, env_filename)

        if not os.path.exists(env_filepath):
            print(f"  Error: Environment file not found: {env_filename}"); continue

        try:
            if env_filepath.endswith(".json"):
                polygon_dict = load_json_data(env_filepath)
            else:
                with open(env_filepath, 'rb') as f:
                    polygon_dict = pickle.load(f)
            vertices = polygon_dict["level_2_polygon"] * scale_factor
        except Exception as e:
            print(f"  Error loading or processing file: {e}"); continue

        # 2. Generate Global Occupancy Grid (Image/Y-down for BARN compatibility)
        try:
            global_grid = rasterize_polygon_interior(
                vertices=vertices,
                map_size=global_map_size_cells,
                resolution=global_resolution,
                origin=global_origin
            )
        except Exception as e:
            print(f"  Error rasterizing global map: {e}"); continue

        # 3. Sample Poses Efficiently - Keep trying until we get enough valid positions
        valid_positions_found = 0
        sampling_attempt = 0
        MAX_SAMPLING_ATTEMPTS = 1000  # Prevent infinite loops
        
        while valid_positions_found < NUM_POSITIONS_PER_ENV:
            sampling_attempt += 1
            
            # Check for infinite loop and temporarily reduce clearance
            if sampling_attempt > MAX_SAMPLING_ATTEMPTS:
                current_clearance = max(0.05, current_clearance - 0.05)  # Reduce by 5cm, minimum 5cm
                sampling_attempt = 0  # Reset attempt counter
                print(f"  Warning: Max attempts reached. Temporarily reducing clearance to {current_clearance:.2f}m")
                if current_clearance <= 0.05:
                    print(f"  Error: Cannot find valid positions even with minimum clearance. Skipping environment {env_id}")
                    break
            
            # Sample one position at a time using polygon-compatible function with BARN clearance checking
            sampled_poses = sample_robot_positions_in_polygon(
                global_grid=global_grid,
                num_positions=1,
                resolution=global_resolution,
                origin=global_origin,
                min_clearance=current_clearance
            )
            
            if not sampled_poses:
                continue

            for pos_idx, robot_pos in enumerate(sampled_poses):
                env_name = f"poly_env_{env_id:04d}_pos_{valid_positions_found:02d}"
                env_output_dir = os.path.join(output_base_dir, env_name)

                # Optional: Skip if the final pickle file already exists
                if os.path.exists(os.path.join(env_output_dir, "uniformity_gt.pkl")):
                    valid_positions_found += 1
                    current_clearance = MIN_CLEARANCE_METERS  # Restore original clearance
                    break  # Break from inner loop and continue to next attempt

                print(f"    Attempt {sampling_attempt}: Testing pose ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {np.degrees(robot_pos[2]):.1f}°)")

                # 4. Simulate Local Scan using BARN function with coordinate conversion
                # Convert polygon world coordinates to BARN coordinates (center origin)
                # BARN helpers expect the robot position relative to the center of the grid.
                robot_world_x, robot_world_y, robot_yaw = robot_pos
                grid_width = global_grid.shape[1]
                grid_height = global_grid.shape[0]
                
                barn_x = robot_world_x - global_origin[0] - (grid_width * global_resolution) / 2.0
                barn_y = robot_world_y - global_origin[1] - (grid_height * global_resolution) / 2.0
                robot_pos_barn = (barn_x, barn_y, robot_yaw)
                
                # global_grid is now Y-down indexed (Image), compatible with simulate_lidar_in_barn.
                lidar_ranges = simulate_lidar_in_barn(
                    barn_grid=global_grid,
                    robot_pos=(barn_x, barn_y),
                    robot_yaw=robot_pos[2],
                    resolution=global_resolution,
                    max_range=LIDAR_MAX_RANGE,
                    num_beams=LIDAR_NUM_BEAMS
                )

                # 5. Create Local Costmap/SDF using BARN function
                # Note: We use LOCAL_RESOLUTION here for the local map generation
                local_costmap_raw = create_robot_centered_costmap(
                    lidar_ranges=lidar_ranges,
                    robot_yaw=robot_yaw,
                    output_grid_size=LOCAL_GRID_SIZE,
                    resolution=LOCAL_RESOLUTION,
                    num_beams=LIDAR_NUM_BEAMS
                )

                local_costmap = binary_dilation(local_costmap_raw, iterations=INFLATION_RADIUS)

                local_sdf = create_sdf_from_costmap(local_costmap, LOCAL_RESOLUTION)

                # 6. Compute Local PRUNED (Safe) Reachability
                level_sets = compute_pruned_level_sets_for_environment(
                    config=MODEL_CONFIG,
                    sdf=local_sdf,
                    resolution=LOCAL_RESOLUTION
                )

                if level_sets is None:
                    print(f"    Attempt {sampling_attempt}: Computation failed or no safe trajectories found. Retrying...")
                    continue

                # 7. Success! Save everything.
                print(f"    Attempt {sampling_attempt}: Success! Saving {env_name}")
                os.makedirs(env_output_dir, exist_ok=True)

                # Save Numpy files
                np.save(os.path.join(env_output_dir, "costmap_raw.npy"), local_costmap_raw)
                np.save(os.path.join(env_output_dir, "costmap.npy"), local_costmap)
                np.save(os.path.join(env_output_dir, "sdf.npy"), local_sdf)

                # Save Ground Truth Pickle
                output_data = {
                    'level_set_representatives': level_sets,
                    'config': MODEL_CONFIG,
                    'robot_pose_global': robot_pos, # Save the global pose used
                }
                with open(os.path.join(env_output_dir, "uniformity_gt.pkl"), 'wb') as f:
                    pickle.dump(output_data, f)

                # Save Visualizations
                # Local SDF visualization
                visualize_sdf(local_sdf, LOCAL_RESOLUTION, os.path.join(env_output_dir, "scene.png"))
                
                # Level set visualization
                level_set_vis_path = os.path.join(env_output_dir, "level_sets.png")
                try:
                    save_level_set_visualization(
                        level_set_representatives=level_sets,
                        filepath=level_set_vis_path,
                        config=MODEL_CONFIG
                    )
                except Exception as e:
                    print(f"    Error during level set visualization: {e}")
                
                # Global/Local verification visualization using unified function
                vis_path = os.path.join(env_output_dir, "polygon_robot_area_verification.png")
                try:
                    visualize_barn_with_robot_area(
                        barn_grid=global_grid,
                        robot_pos=robot_pos_barn,
                        costmap=local_costmap,
                        save_path=vis_path,
                        resolution=global_resolution
                    )
                except Exception as e:
                    print(f"    Error during verification visualization: {e}")

                valid_positions_found += 1
                current_clearance = MIN_CLEARANCE_METERS  # Restore original clearance
                total_valid_positions += 1

                # Memory Cleanup (Mitigation for potential memory issues/SegFaults)
                del local_costmap, local_sdf, level_sets, output_data
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass # Handle potential CUDA context errors during cleanup
                
                # Break out of the inner for loop since we found a valid position
                break

        print(f"  ✓ Found {valid_positions_found}/{NUM_POSITIONS_PER_ENV} valid positions after {sampling_attempt} attempts.")
        
        # Clean up global map memory
        del global_grid, vertices, polygon_dict
        gc.collect()


    print("="*80)
    print("Ground Truth Dataset Generation Complete.")
    print(f"Total valid scenarios generated: {total_valid_positions}")
    print(f"Output directory: {output_base_dir}")
    print("="*80)


# =============================================================================
# 7. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(2025)
    if torch.cuda.is_available():
        torch.manual_seed(2025)

    main()