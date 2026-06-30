import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from typing import Dict, Any

def _process_binary_costmap(
    binary_local_costmap: np.ndarray,
    experiment_config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Processes a binary local costmap to generate inflated costmap and SDF tensor.
    This function encapsulates the shared logic used by both LiDAR and Ground Truth strategies.

    Args:
        binary_local_costmap: The input binary costmap (robot-centered and robot-aligned).
        experiment_config: Configuration dictionary.
        device: The device for the SDF tensor.

    Returns:
        A dictionary containing the processed perception data.
    """
    # Use local costmap resolution as it defines the scale of the local map for SDF
    resolution = experiment_config['local_costmap_resolution']

    # --- Step 1: Create Inflated Costmap (For MPC Cost Functions) ---
    inflation_radius = experiment_config['inflation_radius']
    max_inflation_value = experiment_config['max_inflation_value']

    # Apply distance transform to the binary costmap
    obstacle_mask = binary_local_costmap > 0.5
    
    # Handle edge cases (e.g., fully occupied map)
    if obstacle_mask.all():
        raise ValueError("Fully occupied map. This should never happen.")
    else:
        # distance_transform_edt computes the distance from non-zero (free space) to zero (obstacles)
        distance_map = distance_transform_edt(~obstacle_mask)

    # Apply linear decay to create smooth cost gradients
    if inflation_radius > 0:
        # Inflation radius is defined in grid cells
        inflated_costmap = np.clip(
            (inflation_radius - distance_map) / inflation_radius, 
            0, 1
        ) * max_inflation_value
    else:
        inflated_costmap = np.zeros_like(distance_map, dtype=np.float32)

    # Combine with original obstacles to ensure they have the max cost
    inflated_costmap = np.maximum(binary_local_costmap, inflated_costmap).astype(np.float32)

    # --- Step 2: Create Signed Distance Field (SDF) Tensor (For NN Model) ---
    sdf_inflation_cells = experiment_config['sdf_inflation_cells']

    # Create the binary obstacle map for the SDF.
    sdf_obstacle_map = binary_local_costmap > 0.5

    # If sdf_inflation_cells > 0, perform a direct binary dilation (hard inflation).
    if sdf_inflation_cells > 0:
        sdf_obstacle_map = binary_dilation(sdf_obstacle_map, iterations=sdf_inflation_cells)

    # Generate the SDF from this clean, intentionally-inflated binary map.
    # Handle fully occupied or fully free maps
    if sdf_obstacle_map.all():
        raise ValueError("Fully occupied map. This should never happen in BARN or Polygon environments!")
    elif not sdf_obstacle_map.any():
        # Handle fully free maps (e.g., open space navigation)
        # Create a large positive SDF indicating free space everywhere
        sdf = np.full_like(sdf_obstacle_map, 10.0 * resolution, dtype=np.float32)
    else:
        dist_out = distance_transform_edt(~sdf_obstacle_map) * resolution
        dist_in = distance_transform_edt(sdf_obstacle_map) * resolution
        sdf = dist_out - dist_in
    
    # Convert to PyTorch tensor (B, C, H, W)
    sdf_tensor = torch.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0).to(device)

    # --- Step 3: Return standardized dictionary ---
    return {
        'binary_costmap': binary_local_costmap,
        'inflated_costmap': inflated_costmap,
        'sdf_tensor': sdf_tensor,
    }

#===============================================================================
# LiDAR Perception Strategy (Partial Observation)
#===============================================================================
def generate_perception_data(
    global_occupancy_grid: np.ndarray,
    robot_state: np.ndarray,
    experiment_config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    (LiDAR Strategy) Simulates a LiDAR scan to generate the local perception data.
    The resulting costmap is robot-centered and robot-aligned (egocentric).

    Args:
        global_occupancy_grid: The global ground-truth map.
        robot_state: The current state of the robot [x, y, theta, ...].
        experiment_config: Dictionary with all necessary parameters (lidar, costmap, etc.).
        device: The device to use for the SDF tensor.

    Returns:
        A dictionary containing the generated perception data.
    """
    # --- Step 1: Simulate LiDAR Scan (Ray-casting) ---
    # GET MAP AND ROBOT PARAMETERS
    abs_robot_x, abs_robot_y, robot_yaw = robot_state[:3]
    # Use global resolution for ray casting on the global map
    resolution = experiment_config['global_map_resolution']
    global_origin = experiment_config['global_origin']
    grid_height, grid_width = global_occupancy_grid.shape

    # SIMULATE LIDAR SCAN (RAY CASTING)
    max_range = experiment_config['lidar_scan_range']
    num_beams = experiment_config['lidar_num_beams']
    lidar_ranges = np.full(num_beams, max_range, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)

    # Convert robot's absolute world position to correct grid coordinates
    robot_grid_x = (abs_robot_x - global_origin[0]) / resolution
    robot_grid_y = (abs_robot_y - global_origin[1]) / resolution
    
    # NOTE: Depending on how .npy maps are saved, flip the Y-axis for indexing.
    # If row 0 of the grid is the "bottom" of the map, the line above is correct.
    # If row 0 is the "top", use:
    # robot_grid_y = (grid_height - 1) - (abs_robot_y - global_origin[1]) / resolution

    for i, angle in enumerate(angles):
        beam_angle = robot_yaw + angle
        # Ray casting with small steps
        step_size = resolution * 0.5
        max_steps = int(max_range / step_size)

        for step in range(1, max_steps + 1):
            dist = step * step_size
            # Get the grid cell for this point on the ray
            check_x = int(robot_grid_x + (dist / resolution) * np.cos(beam_angle))
            check_y = int(robot_grid_y + (dist / resolution) * np.sin(beam_angle))

            # Check grid boundaries
            if not (0 <= check_x < grid_width and 0 <= check_y < grid_height):
                break # Beam went off the map

            # Check for obstacle
            if global_occupancy_grid[check_y, check_x] > 0.5:
                lidar_ranges[i] = dist
                break # Obstacle hit

    # --- Step 2: Create Robot-Centered Binary Local Costmap From Lidar Ranges---
    output_grid_size = experiment_config['local_costmap_size'] # e.g., for a 6.05m x 6.05m costmap at 0.05m/res
    local_resolution = experiment_config['local_costmap_resolution']
    binary_local_costmap = np.zeros((output_grid_size, output_grid_size), dtype=np.float32)
    center = output_grid_size // 2

    for i, range_val in enumerate(lidar_ranges):
        if range_val >= max_range:
            continue

        # Obstacle position in the robot's frame (+X is forward)
        # The angle from the scan IS the angle in the robot's frame.
        obs_x_robot = range_val * np.cos(angles[i])
        obs_y_robot = range_val * np.sin(angles[i])

        # Convert to the local costmap's grid coordinates
        grid_x = int(center + obs_x_robot / local_resolution)
        grid_y = int(center - obs_y_robot / local_resolution) # Y-flip for image coordinates

        if 0 <= grid_x < output_grid_size and 0 <= grid_y < output_grid_size:
            binary_local_costmap[grid_y, grid_x] = 1.0

    # --- Step 3: Process the binary costmap (Inflation and SDF) ---
    # Delegate to the shared processing function
    return _process_binary_costmap(binary_local_costmap, experiment_config, device)

#===============================================================================
# Ground Truth Perception Strategy (Full Observation)
#===============================================================================
def generate_ground_truth_perception(
    global_occupancy_grid: np.ndarray,
    robot_state: np.ndarray,
    experiment_config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    (Ground-Truth Strategy) Samples the global map to generate perfect local perception data.
    Uses vectorized coordinate transformation to ensure the output is EGOCENTRIC 
    (robot-centered and robot-aligned), matching the LiDAR strategy output.
    """
    # --- Step 1: Get Parameters and Setup ---
    robot_x, robot_y, robot_yaw = robot_state[:3]
    global_resolution = experiment_config['global_map_resolution']
    local_resolution = experiment_config['local_costmap_resolution']
    global_origin = experiment_config['global_origin']
    output_grid_size = experiment_config['local_costmap_size']
    grid_height, grid_width = global_occupancy_grid.shape
    center = output_grid_size // 2

    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)

    # --- Step 2: Vectorized Coordinate Transformation ---
    # We calculate where each pixel in the local map corresponds to in the global map.

    # 2a. Create coordinate grids for the local costmap indices
    local_indices = np.arange(output_grid_size)
    # Meshgrid creates (Y, X) arrays of indices
    local_x_idx, local_y_idx = np.meshgrid(local_indices, local_indices)

    # 2b. Convert local grid indices to robot frame coordinates (meters)
    # Robot frame: +X forward. Image: +X right, +Y down.
    robot_frame_x = (local_x_idx - center) * local_resolution
    robot_frame_y = (center - local_y_idx) * local_resolution # Y-flip (image coords to robot coords)

    # 2c. Transform robot frame coordinates to world frame coordinates (meters)
    # Apply rotation matrix and translation
    world_frame_x = robot_x + robot_frame_x * cos_yaw - robot_frame_y * sin_yaw
    world_frame_y = robot_y + robot_frame_x * sin_yaw + robot_frame_y * cos_yaw

    # 2d. Convert world frame coordinates to global grid indices
    # This step naturally handles resampling if global/local resolutions differ (nearest neighbor).
    global_grid_x = ((world_frame_x - global_origin[0]) / global_resolution).astype(int)
    global_grid_y = ((world_frame_y - global_origin[1]) / global_resolution).astype(int)

    # --- Step 3: Sample the Global Map (Vectorized Indexing) ---
    # Create masks for coordinates that fall within the global map boundaries
    valid_mask = ((global_grid_x >= 0) & (global_grid_x < grid_width) &
                  (global_grid_y >= 0) & (global_grid_y < grid_height))
    
    extract_boundaries = experiment_config['extract_boundaries']

    # Initialize local map with obstacles (1.0) for out-of-bounds areas (conservative approach)
    # binary_local_costmap = np.ones((output_grid_size, output_grid_size), dtype=np.float32)
    if extract_boundaries:
        # --- Boundary Extraction Mode ---
        # 1. Initialize to free space (0.0). This isolates sampled obstacles for extraction 
        #    without affecting out-of-bounds areas yet.
        binary_local_costmap = np.zeros((output_grid_size, output_grid_size), dtype=np.float32)
        # 2. Sample the global map.
        binary_local_costmap[valid_mask] = global_occupancy_grid[global_grid_y[valid_mask], global_grid_x[valid_mask]]
        # 3. Perform boundary extraction (Boundary = Original - Eroded).
        obstacle_mask = binary_local_costmap > 0.5
        # Use a standard 3x3 structuring element (8-connectivity).
        structure = np.ones((3, 3), dtype=bool)
        eroded_mask = binary_erosion(obstacle_mask, structure=structure)
        boundary_mask = obstacle_mask & (~eroded_mask)
        
        # 4. Update the map with boundaries.
        binary_local_costmap = boundary_mask.astype(np.float32)
        
        # Apply conservative assumption for out-of-bounds areas
        # binary_local_costmap[~valid_mask] = 1.0
    else:
        # --- Dense Map Mode ---
        # Initialize local map with obstacles (1.0) for out-of-bounds areas (conservative approach)
        binary_local_costmap = np.ones((output_grid_size, output_grid_size), dtype=np.float32)

        # Use the mask and the calculated indices to sample the global map efficiently
        binary_local_costmap[valid_mask] = global_occupancy_grid[global_grid_y[valid_mask], global_grid_x[valid_mask]]

    # --- Step 4: Process the binary costmap (Inflation and SDF) ---
    return _process_binary_costmap(binary_local_costmap, experiment_config, device)