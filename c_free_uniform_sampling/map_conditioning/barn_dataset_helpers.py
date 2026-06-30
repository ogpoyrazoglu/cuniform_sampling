#!/usr/bin/env python3
"""
BARN Dataset Helpers - Simplified and Brittle Version
No fallbacks, no hidden assumptions, fails fast with assertions.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Add map_conditioning directory to path for imports
map_conditioning_dir = os.path.dirname(os.path.abspath(__file__))
if map_conditioning_dir not in sys.path:
    sys.path.append(map_conditioning_dir)

from utility_helper_map import create_sdf_from_obstacles

def load_barn_environment(barn_dataset_dir: str, env_id: int) -> Tuple[np.ndarray, float]:
    """Load BARN environment data - BRITTLE VERSION."""
    # Assertions - fail fast
    assert os.path.exists(barn_dataset_dir), f"BARN dataset directory does not exist: {barn_dataset_dir}"
    assert 0 <= env_id <= 299, f"Invalid BARN environment ID: {env_id} (must be 0-299)"
    
    # BARN dataset uses grid_files/grid_X.npy format
    grid_file = os.path.join(barn_dataset_dir, "grid_files", f"grid_{env_id}.npy")
    assert os.path.exists(grid_file), f"BARN grid file does not exist: {grid_file}"
    
    # Load occupancy grid
    occupancy_grid = np.load(grid_file)
    assert isinstance(occupancy_grid, np.ndarray), f"occupancy_grid must be numpy array, got {type(occupancy_grid)}"
    assert occupancy_grid.ndim == 2, f"occupancy_grid must be 2D, got {occupancy_grid.ndim}D"
    
    # Load YAML metadata - NO FALLBACKS, REQUIRED
    yaml_file = os.path.join(barn_dataset_dir, "map_files", f"yaml_{env_id}.yaml")
    assert os.path.exists(yaml_file), f"BARN YAML file does not exist: {yaml_file}"
    
    import yaml
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    resolution = yaml_data.get('resolution')
    assert resolution is not None, f"Missing 'resolution' in YAML file: {yaml_file}"
    assert isinstance(resolution, (int, float)), f"resolution must be numeric, got {type(resolution)}"
    assert resolution > 0, f"resolution must be positive, got {resolution}"
    
    print(f"     Loaded BARN env {env_id}: {occupancy_grid.shape} @ {resolution}m/cell")
    
    return occupancy_grid, resolution


# Removed old brittle functions - using clean modular approach instead


def _costmap_to_obstacles(costmap: np.ndarray, resolution: float) -> List[Tuple[float, float, float]]:
    """Convert costmap to obstacle list - BRITTLE VERSION."""
    assert costmap.ndim == 2, f"costmap must be 2D, got {costmap.ndim}D"
    assert costmap.shape[0] == costmap.shape[1], f"costmap must be square, got {costmap.shape}"
    assert resolution > 0, f"resolution must be positive, got {resolution}"
    
    obstacles = []
    grid_size = costmap.shape[0]
    
    for gy in range(grid_size):
        for gx in range(grid_size):
            if costmap[gy, gx] > 0.5:
                # Convert grid to world coordinates
                # NOTE: costmap uses Y-flip: grid_y = center - obs_y_robot / resolution
                # So we need to reverse this: obs_y_robot = (center - grid_y) * resolution
                world_x = (gx - grid_size // 2) * resolution
                world_y = (grid_size // 2 - gy) * resolution  # Y-flip to match costmap creation
                obstacles.append((world_x, world_y, resolution * 0.7))
    
    return obstacles


# ===========================================================================================
# NEW MODULAR BARN DATASET PROCESSING FUNCTIONS (Clean Implementation)
# ===========================================================================================

def sample_robot_positions_in_barn(barn_grid: np.ndarray, 
                                  num_positions: int,
                                  min_clearance: float = 0.1,
                                  resolution: float = 0.05) -> List[Tuple[float, float, float]]:
    """Sample valid robot positions in the BARN grid with required clearance.
    
    Args:
        barn_grid: 2D binary array (1 = obstacle, 0 = free), typically 90×90@0.05m
        num_positions: Number of positions to sample
        min_clearance: Minimum clearance from obstacles in meters (default: 0.1m)
        resolution: Grid resolution in meters per cell (default: 0.05m)
        
    Returns:
        List of (x, y, yaw) tuples in BARN world coordinates (center of grid is origin)
    """
    assert barn_grid.ndim == 2, f"barn_grid must be 2D, got {barn_grid.ndim}D"
    assert num_positions > 0, f"num_positions must be positive, got {num_positions}"
    assert min_clearance >= 0, f"min_clearance must be non-negative, got {min_clearance}"
    assert resolution > 0, f"resolution must be positive, got {resolution}"
    
    height, width = barn_grid.shape
    clearance_cells = int(np.ceil(min_clearance / resolution))
    
    print(f"     Sampling {num_positions} robot positions in BARN grid")
    print(f"       Grid: {height}×{width}@{resolution}m, clearance: {min_clearance}m ({clearance_cells} cells)")
    
    valid_positions = []
    max_attempts = num_positions * 200  # More attempts for better success rate
    attempts = 0
    
    while len(valid_positions) < num_positions and attempts < max_attempts:
        attempts += 1
        
        # Sample random grid position with clearance margin
        grid_x = np.random.randint(clearance_cells, width - clearance_cells)
        grid_y = np.random.randint(clearance_cells, height - clearance_cells)
        
        # Check if position has required clearance
        if _check_clearance_in_grid(barn_grid, grid_x, grid_y, clearance_cells):
            # Convert to BARN world coordinates (center of BARN grid is origin)
            world_x = (grid_x - width // 2) * resolution
            world_y = (height // 2 - grid_y) * resolution  # Y increases upward
            world_yaw = np.random.uniform(0, 2 * np.pi)  # Random orientation
            
            valid_positions.append((world_x, world_y, world_yaw))
    
    print(f"       Found {len(valid_positions)} valid positions in {attempts} attempts")
    return valid_positions


def _check_clearance_in_grid(grid: np.ndarray, grid_x: int, grid_y: int, clearance_cells: int) -> bool:
    """Check if position has required clearance from obstacles."""
    height, width = grid.shape
    
    # Check bounds
    if (grid_x - clearance_cells < 0 or grid_x + clearance_cells >= width or
        grid_y - clearance_cells < 0 or grid_y + clearance_cells >= height):
        return False
    
    # Check clearance area for obstacles
    for dy in range(-clearance_cells, clearance_cells + 1):
        for dx in range(-clearance_cells, clearance_cells + 1):
            if grid[grid_y + dy, grid_x + dx] > 0.5:  # Obstacle detected
                return False
    
    return True


def visualize_barn_with_robot_area(barn_grid: np.ndarray,
                                   robot_pos: Tuple[float, float, float],
                                   costmap: np.ndarray,
                                   save_path: str,
                                   resolution: float = 0.05):
    """Visualize BARN environment with robot-centered 121×121 area overlay for verification.
    
    This shows:
    1. Background: 90×90@0.05 BARN environment 
    2. Overlay: 121×121@0.05 square centered on robot position
    3. Robot position and orientation
    4. Generated costmap obstacles
    """
    robot_x, robot_y, robot_yaw = robot_pos
    barn_height, barn_width = barn_grid.shape
    costmap_size = costmap.shape[0]  # Should be 121
    
    # Calculate world coverage
    barn_world_size = barn_width * resolution  # Should be 4.5m
    costmap_world_size = costmap_size * resolution  # Should be 6.05m
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # =====================================
    # Plot 1: BARN Environment with Robot-Centered Area Overlay
    # =====================================
    
    # Create extended canvas that can fit both BARN + robot area
    canvas_size = max(barn_width, costmap_size) + 50  # Add padding
    canvas = np.zeros((canvas_size, canvas_size))
    
    # Place BARN grid in center of canvas
    barn_start_x = (canvas_size - barn_width) // 2
    barn_start_y = (canvas_size - barn_height) // 2
    canvas[barn_start_y:barn_start_y + barn_height, barn_start_x:barn_start_x + barn_width] = barn_grid
    
    # Convert robot world position to canvas coordinates
    robot_canvas_x = barn_start_x + (robot_x / resolution) + barn_width // 2
    robot_canvas_y = barn_start_y + (barn_height // 2) - (robot_y / resolution)
    
    # Calculate 121×121 area bounds around robot
    half_costmap = costmap_size // 2
    area_x1 = int(robot_canvas_x - half_costmap)
    area_x2 = int(robot_canvas_x + half_costmap + 1)
    area_y1 = int(robot_canvas_y - half_costmap)
    area_y2 = int(robot_canvas_y + half_costmap + 1)
    
    # Show canvas with BARN environment
    ax1.imshow(canvas, cmap='gray_r', origin='upper')
    
    # Highlight BARN area
    barn_rect = plt.Rectangle((barn_start_x, barn_start_y), barn_width, barn_height, 
                            fill=False, edgecolor='blue', linewidth=2, label='Global @0.05m')
    ax1.add_patch(barn_rect)
    
    # Highlight robot-centered 121×121 area
    robot_rect = plt.Rectangle((area_x1, area_y1), costmap_size, costmap_size,
                             fill=False, edgecolor='red', linewidth=2, label='LiDAR Scan Area 121×121@0.05m')
    ax1.add_patch(robot_rect)
    
    # Plot robot position and orientation
    ax1.plot(robot_canvas_x, robot_canvas_y, 'ro', markersize=10, label='Robot')
    
    # Draw robot orientation arrow
    arrow_length = 15
    arrow_end_x = robot_canvas_x + arrow_length * np.cos(robot_yaw)
    arrow_end_y = robot_canvas_y - arrow_length * np.sin(robot_yaw)  # Negative because image Y is flipped
    ax1.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(robot_canvas_x, robot_canvas_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax1.set_title(f'Global Environment + Robot Area\nRobot: ({robot_x:.2f}, {robot_y:.2f}), θ={robot_yaw:.2f}rad')
    ax1.set_xlabel('Canvas Grid X')
    ax1.set_ylabel('Canvas Grid Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # =====================================
    # Plot 2: Generated Robot-Centered Costmap
    # =====================================
    
    ax2.imshow(costmap, cmap='gray_r', origin='upper')
    ax2.plot(costmap_size // 2, costmap_size // 2, 'ro', markersize=8, label='Robot (center)')
    
    # Draw robot orientation in costmap - robot always faces +X direction
    costmap_arrow_length = 10
    costmap_center = costmap_size // 2
    # Robot faces +X direction in the robot frame (right direction in visualization)
    costmap_arrow_end_x = costmap_center + costmap_arrow_length  # +X direction
    costmap_arrow_end_y = costmap_center  # No Y component
    ax2.annotate('', xy=(costmap_arrow_end_x, costmap_arrow_end_y), xytext=(costmap_center, costmap_center),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, label='Robot Forward (+X)'))
    
    obstacle_count = np.sum(costmap > 0.5)
    ax2.set_title(f'Robot-Frame LiDAR Costmap (Robot faces +X)\n{obstacle_count} obstacles, 121×121@0.05m')
    ax2.set_xlabel('Robot Frame X (Forward)')
    ax2.set_ylabel('Robot Frame Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    

    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      BARN+Robot area visualization saved: {save_path}")

def simulate_lidar_in_barn(barn_grid: np.ndarray, 
                          robot_pos: Tuple[float, float],
                          robot_yaw: float,
                          resolution: float = 0.05,
                          max_range: float = 3.0,
                          num_beams: int = 180) -> np.ndarray:
    """Simulate LiDAR scan in BARN environment.
    
    Args:
        barn_grid: 2D BARN occupancy grid (90×90@0.05m)
        robot_pos: Robot position (x, y) in BARN world coordinates  
        robot_yaw: Robot orientation in radians
        resolution: Grid resolution in meters per cell
        max_range: Maximum LiDAR range in meters
        num_beams: Number of LiDAR beams
        
    Returns:
        Array of ranges for each beam
    """
    height, width = barn_grid.shape
    robot_x, robot_y = robot_pos
    
    # Debug: Check BARN grid content
    obstacle_count = np.sum(barn_grid > 0.5)
    print(f"       BARN grid: {barn_grid.shape}, obstacles: {obstacle_count}/{barn_grid.size} cells")
    print(f"       Robot position: ({robot_x:.3f}, {robot_y:.3f}), yaw: {robot_yaw:.3f}")
    
    # Convert robot position to BARN grid coordinates
    grid_x = int((robot_x / resolution) + width // 2)
    grid_y = int((height // 2) - (robot_y / resolution))  # Y decreases with row index
    
    print(f"       Robot grid coords: ({grid_x}, {grid_y}) in {width}×{height} grid")
    
    # Check if robot is within bounds
    if not (0 <= grid_x < width and 0 <= grid_y < height):
        print(f"        WARNING: Robot outside BARN grid bounds!")
        return np.full(num_beams, max_range)
    
    # Check if robot is in collision
    if barn_grid[grid_y, grid_x] > 0.5:
        print(f"        WARNING: Robot in collision with obstacle!")
    
    angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
    ranges = np.full(num_beams, max_range)
    obstacles_detected = 0
    
    for i, angle in enumerate(angles):
        # Beam direction in world frame
        beam_angle = robot_yaw + angle
        dx = np.cos(beam_angle)  # World frame step in x
        dy = np.sin(beam_angle)  # World frame step in y
        
        # Ray casting with small steps for accuracy
        step_size = resolution * 0.5  # Half resolution for finer steps
        max_steps = int(max_range / step_size)
        
        for step in range(1, max_steps + 1):
            # World coordinates
            world_x = robot_x + step * step_size * dx
            world_y = robot_y + step * step_size * dy
            
            # Convert to grid coordinates
            x = int((world_x / resolution) + width // 2)
            y = int((height // 2) - (world_y / resolution))
            
            # Check bounds - if outside BARN grid, keep max range
            if x < 0 or x >= width or y < 0 or y >= height:
                break  # Beam exits BARN grid - keep max_range
            
            # Check obstacle in BARN grid
            if barn_grid[y, x] > 0.5:
                ranges[i] = step * step_size
                obstacles_detected += 1
                break
    
    print(f"       LiDAR scan: {obstacles_detected}/{num_beams} beams hit obstacles")
    print(f"       Range stats: min={ranges.min():.3f}m, max={ranges.max():.3f}m, mean={ranges.mean():.3f}m")
    
    return ranges


def create_robot_centered_costmap(lidar_ranges: np.ndarray,
                                 robot_yaw: float = 0.0,
                                 output_grid_size: int = 121,
                                 resolution: float = 0.05,
                                 num_beams: int = 180) -> np.ndarray:
    """Create 121×121@0.05m costmap centered on robot from LiDAR data.
    Robot always faces +X direction in the final costmap.
    
    Args:
        lidar_ranges: Array of LiDAR ranges
        robot_yaw: Robot orientation in world frame (used to rotate to robot frame)
        output_grid_size: Output grid size (default: 121)
        resolution: Grid resolution (default: 0.05m)
        num_beams: Number of LiDAR beams (default: 180)
        
    Returns:
        121×121 costmap with robot at center, robot facing +X direction
    """
    assert len(lidar_ranges) == num_beams, f"Range array size mismatch: {len(lidar_ranges)} != {num_beams}"
    assert output_grid_size % 2 == 1, f"output_grid_size must be odd, got {output_grid_size}"
    
    costmap = np.zeros((output_grid_size, output_grid_size))
    center = output_grid_size // 2
    
    # LiDAR angles in sensor frame (0 to 2π)
    sensor_angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
    
    obstacles_in_costmap = 0
    obstacle_positions = []  # Store for inflation
    
    for sensor_angle, range_val in zip(sensor_angles, lidar_ranges):
        if range_val >= 3.0:  # No obstacle detected (max range)
            continue
            
        if range_val < 0.1:  # Filter extremely close readings (likely noise)
            continue
            
        # The LiDAR angles are already relative to robot's heading
        # sensor_angle = 0 corresponds to robot's forward direction
        # In robot frame: +X is forward, so use sensor_angle directly
        robot_frame_angle = sensor_angle
        
        # Obstacle position in robot frame (robot faces +X)
        obs_x_robot = range_val * np.cos(robot_frame_angle)
        obs_y_robot = range_val * np.sin(robot_frame_angle)
        
        # Convert to grid coordinates (robot at center)
        # Robot frame: +X is forward (right in visualization), +Y is left (up in visualization)
        # Image array: +X is right, +Y is down (row index increases downward)
        grid_x = int(center + obs_x_robot / resolution)
        grid_y = int(center - obs_y_robot / resolution)  # Y-flip for image coordinates
        
        # Store obstacle position for inflation
        if 0 <= grid_x < output_grid_size and 0 <= grid_y < output_grid_size:
            obstacle_positions.append((grid_x, grid_y))
            obstacles_in_costmap += 1
    
    # Add obstacles with 1-cell inflation for continuity
    for grid_x, grid_y in obstacle_positions:
        # Mark original position and 8-connected neighbors (1-cell inflation)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < output_grid_size and 0 <= ny < output_grid_size:
                    costmap[ny, nx] = 1.0
    
    print(f"        Costmap: {obstacles_in_costmap} obstacles marked in {output_grid_size}×{output_grid_size} grid")
    
    # CRITICAL: Assert costmap is not empty for debugging
    if obstacles_in_costmap == 0:
        print(f"       ERROR: Empty costmap generated!")
        print(f"         LiDAR range summary: {len(lidar_ranges)} beams")
        print(f"         Obstacles detected: {np.sum(lidar_ranges < 2.5)}")
        print(f"         Range distribution: {np.histogram(lidar_ranges, bins=[0, 0.5, 1.0, 1.5, 2.0, 2.5])[0]}")
        
        # Save debug info
        np.save("debug_lidar_ranges.npy", lidar_ranges)
        np.save("debug_costmap.npy", costmap)
    
    return costmap


def process_single_barn_position(barn_grid: np.ndarray,
                                robot_pos: Tuple[float, float, float],
                                resolution: float = 0.05,
                                debug_dir: str = None) -> dict:
    """Process a single robot position in BARN environment to create supervised learning data.
    
    Args:
        barn_grid: 90×90@0.05m BARN occupancy grid
        robot_pos: (x, y, yaw) in BARN world coordinates
        resolution: Grid resolution (0.05m)
        debug_dir: Optional directory to save debug visualizations
        
    Returns:
        Dictionary with 'costmap', 'sdf', 'robot_pose', 'obstacles' or None if failed
    """
    robot_x, robot_y, robot_yaw = robot_pos
    
    # Step 1: Simulate LiDAR scan in BARN environment
    lidar_ranges = simulate_lidar_in_barn(barn_grid, (robot_x, robot_y), robot_yaw, resolution)
    
    # Step 2: Create 121×121@0.05m robot-centered costmap (only LiDAR obstacles)
    costmap = create_robot_centered_costmap(lidar_ranges, robot_yaw, output_grid_size=121, resolution=resolution)
    
    # Step 2.5: Save BARN + robot area visualization for coordinate verification
    if debug_dir is not None:
        # Create debug directory if it doesn't exist
        os.makedirs(debug_dir, exist_ok=True)
        barn_area_vis_path = os.path.join(debug_dir, f"barn_robot_area_pos_{robot_x:.2f}_{robot_y:.2f}.png")
        visualize_barn_with_robot_area(barn_grid, robot_pos, costmap, barn_area_vis_path, resolution)
    
    # Step 3: CRITICAL ASSERTION - Costmap must not be empty
    obstacle_pixels = np.sum(costmap > 0.5)
    if obstacle_pixels == 0:
        print(f"       CRITICAL ERROR: Generated costmap is completely empty!")
        print(f"         This should not happen with BARN environment obstacles.")
        print(f"         Debug files saved for investigation.")
        
        # Force save debug info
        if debug_dir is not None:
            # Create debug directory if it doesn't exist
            os.makedirs(debug_dir, exist_ok=True)
            np.save(os.path.join(debug_dir, f"debug_barn_grid.npy"), barn_grid)
            np.save(os.path.join(debug_dir, f"debug_lidar_ranges.npy"), lidar_ranges)
            np.save(os.path.join(debug_dir, f"debug_empty_costmap.npy"), costmap)
        
        # Still continue processing for debugging, but mark as problematic
        print(f"        Continuing with empty costmap for debugging purposes...")
    else:
        print(f"       Costmap validation: {obstacle_pixels} obstacle pixels found")
    
    # Step 4: Convert costmap to obstacles for SDF computation
    obstacles = _costmap_to_obstacles(costmap, resolution)
    
    # Step 5: Create SDF
    _, sdf = create_sdf_from_obstacles(obstacles, grid_size=121, resolution=resolution)
    
    return {
        'costmap': costmap,
        'sdf': sdf,
        'robot_pose': robot_pos,
        'obstacles': obstacles,
        'lidar_ranges': lidar_ranges  # Include for debugging
    }


def create_barn_supervised_dataset_clean(barn_dataset_dir: str, 
                                       output_base_dir: str,
                                       num_environments: int,
                                       num_positions_per_env: int,
                                       start_env_id: int,
                                       resolution: float,
                                       skip_existing: bool = True,
                                       config: dict = None) -> None:
    """Create BARN supervised dataset with single-phase approach - 121×121@0.05m consistently.
    
    Single-phase pipeline: Try position → Validate → Compute action probs → Save → Next position
    """
    assert os.path.exists(barn_dataset_dir), f"BARN dataset directory not found: {barn_dataset_dir}"
    assert num_environments > 0, f"num_environments must be positive: {num_environments}"
    assert num_positions_per_env > 0, f"num_positions_per_env must be positive: {num_positions_per_env}"
    assert start_env_id >= 0, f"start_env_id must be non-negative: {start_env_id}"
    assert resolution > 0, f"resolution must be positive: {resolution}"
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    total_valid_positions = 0
    target_total = num_environments * num_positions_per_env
    
    print(f"🏗  Creating BARN supervised dataset (Single-Phase Pipeline)")
    print(f"     Input: {barn_dataset_dir}")
    print(f"     Output: {output_base_dir}")
    print(f"     Target: {target_total} positions ({num_environments} envs × {num_positions_per_env} positions)")
    print(f"     Grid: 121×121@{resolution}m (consistent)")
    print(f"     Model: {config['model_name']}")
    print(f"     BARN→90×90@{resolution}m → LiDAR → 121×121@{resolution}m")
    
    for i in range(num_environments):
        env_id = start_env_id + i
        print(f"\n   Environment {i+1}/{num_environments}: BARN {env_id}")
        
        try:
            # Load BARN environment once
            barn_occupancy_grid, barn_resolution = load_barn_environment(barn_dataset_dir, env_id)
            print(f"     Loaded BARN {env_id}: {barn_occupancy_grid.shape}@{barn_resolution}m")
            
            # Convert to target resolution once
            barn_grid_converted = convert_barn_to_target_resolution(
                barn_occupancy_grid, barn_resolution, resolution
            )
            
            # Single-phase pipeline: Try positions one by one until we have enough valid ones
            valid_positions_found = 0
            max_attempts = num_positions_per_env * 50  # Reasonable limit
            attempts = 0
            
            while valid_positions_found < num_positions_per_env and attempts < max_attempts:
                attempts += 1
                
                # Sample one robot position
                robot_positions = sample_robot_positions_in_barn(
                    barn_grid_converted, num_positions=1, min_clearance=0.2, resolution=resolution
                )
                
                if len(robot_positions) == 0:
                    continue  # Try another position
                
                robot_pos = robot_positions[0]
                env_name = f"env_{env_id:04d}_pos_{valid_positions_found:02d}"
                env_output_dir = os.path.join(output_base_dir, env_name)
                
                print(f"     Trying position {valid_positions_found+1}: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})")
                
                # Skip if already exists and skip_existing is True
                if skip_existing and os.path.exists(env_output_dir):
                    required_files = ['costmap.npy', 'sdf.npy', 'scene.png', 'action_probs.pkl', 
                                    'ground_truth_trajectories.png', 'level_set_representatives.png']
                    if all(os.path.exists(os.path.join(env_output_dir, f)) for f in required_files):
                        print(f"      ⏭  {env_name} already exists, skipping")
                        valid_positions_found += 1
                        continue
                
                try:
                    # Process this single position
                    env_data = process_single_barn_position(barn_grid_converted, robot_pos, resolution, debug_dir=None)
                    
                    # Test if action probability generation works
                    from supervised_dataset_generation import compute_action_probabilities_for_environment
                    action_prob_config = config.copy()
                    action_prob_config["initial_state_set"] = np.array([[0.0, 0.0, 0.0]])
                    
                    action_prob_data, _ = compute_action_probabilities_for_environment(
                        action_prob_config, env_data['obstacles'], env_data['sdf'], 
                        resolution, initial_state=np.array([0.0, 0.0, 0.0])
                    )
                    
                    if action_prob_data is None:
                        print(f"       Action probability generation failed, trying next position...")
                        continue
                    
                    # Success! Save everything immediately
                    print(f"       Position valid, saving {env_name}...")
                    os.makedirs(env_output_dir, exist_ok=True)
                    
                    # Save basic files
                    np.save(os.path.join(env_output_dir, "costmap.npy"), env_data['costmap'])
                    np.save(os.path.join(env_output_dir, "sdf.npy"), env_data['sdf'])
                    
                    # Save scene visualization 
                    from utility_helper_map import visualize_sdf
                    scene_path = os.path.join(env_output_dir, "scene.png")
                    visualize_sdf(env_data['sdf'], resolution, scene_path)
                    
                    # Save action probabilities
                    with open(os.path.join(env_output_dir, "action_probs.pkl"), "wb") as f:
                        import pickle
                        pickle.dump(action_prob_data, f)
                    
                    # Save boundary states if any
                    if len(action_prob_data['boundary_states_with_probs']) > 0:
                        with open(os.path.join(env_output_dir, "boundary_states_with_probs.pkl"), "wb") as f:
                            pickle.dump(action_prob_data['boundary_states_with_probs'], f)
                    
                    # Generate and save trajectories
                    from utility_helper_map import parallelized_trajectory_sampling_cuda, visualize_trajectories_background
                    trajectories = parallelized_trajectory_sampling_cuda(action_prob_data, 10000)
                    
                    if len(trajectories) > 0:
                        # Save trajectory visualization
                        vis_filepath = os.path.join(env_output_dir, "ground_truth_trajectories.png")
                        visualize_trajectories_background(
                            trajectories=trajectories,
                            costmap=env_data['costmap'],
                            resolution=resolution,
                            show_vis=False,
                            save_vis=True,
                            vis_filepath=vis_filepath,
                            alpha=0.2,
                            marker_size=1,
                        )
                    
                    # Save level set visualization
                    from utility_helper_map import save_level_set_visualization
                    level_set_vis_path = os.path.join(env_output_dir, "level_set_representatives.png")
                    save_level_set_visualization(
                        action_prob_data['level_set_representatives'], 
                        level_set_vis_path,
                        config,
                        action_prob_data['boundary_states']['boundary_states']
                    )
                    
                    # Save Global + localrobot area visualization for coordinate verification
                    robot_pose = env_data['robot_pose']
                    barn_area_vis_path = os.path.join(env_output_dir, "barn_robot_area_verification.png")
                    try:
                        visualize_barn_with_robot_area(barn_grid_converted, robot_pose, env_data['costmap'], barn_area_vis_path, resolution)
                    except Exception as e:
                        print(f"          Could not save BARN area visualization: {e}")
                    
                    print(f"       {env_name} saved successfully ({attempts} attempts)")
                    valid_positions_found += 1
                    total_valid_positions += 1
                    
                except Exception as e:
                    print(f"       Position processing failed: {e}, trying next position...")
                    continue
            
            if valid_positions_found == num_positions_per_env:
                print(f"     Environment {env_id}: {valid_positions_found}/{num_positions_per_env} positions found")
            else:
                print(f"      Environment {env_id}: Only {valid_positions_found}/{num_positions_per_env} positions found after {attempts} attempts")
                
        except Exception as e:
            print(f"     Environment {env_id} failed: {type(e).__name__}: {str(e)}")
            continue
    
    print(f"\n BARN DATASET SUMMARY")
    print(f" Valid positions: {total_valid_positions}/{target_total}")
    print(f" Output: {output_base_dir}")
    print("="*80)

def convert_barn_to_target_resolution(occupancy_grid: np.ndarray, 
                                     barn_resolution: float, 
                                     target_resolution: float) -> np.ndarray:
    """Convert BARN occupancy grid from barn_resolution to target_resolution with exact scaling.
    
    Args:
        occupancy_grid: Original BARN grid (typically 30×30@0.15m)
        barn_resolution: Original resolution (typically 0.15m)  
        target_resolution: Target resolution (typically 0.05m)
        
    Returns:
        Converted grid (typically 90×90@0.05m) with same world coverage, no boundary padding
    """
    assert barn_resolution > 0, f"barn_resolution must be positive: {barn_resolution}"
    assert target_resolution > 0, f"target_resolution must be positive: {target_resolution}"
    
    if abs(barn_resolution - target_resolution) < 1e-6:
        # Resolutions are the same, return as-is
        return occupancy_grid.copy()
    
    # Calculate exact scaling factor and dimensions
    scale_factor = barn_resolution / target_resolution
    old_height, old_width = occupancy_grid.shape
    
    # Calculate exact new dimensions (no truncation)
    new_height = int(round(old_height * scale_factor))
    new_width = int(round(old_width * scale_factor))
    
    print(f"     BARN Resolution conversion: {barn_resolution}m → {target_resolution}m")
    print(f"       Grid size: ({old_height}, {old_width}) → ({new_height}, {new_width})")
    print(f"       Scale factor: {scale_factor:.3f}")
    
    # Verify exact conversion maintains world coverage
    expected_world_size = old_height * barn_resolution
    actual_world_size = new_height * target_resolution
    assert abs(expected_world_size - actual_world_size) < 1e-6, f"World size mismatch: {expected_world_size} vs {actual_world_size}"
    
    # Create new grid with converted resolution using nearest neighbor interpolation
    converted_grid = np.zeros((new_height, new_width))
    
    for new_y in range(new_height):
        for new_x in range(new_width):
            # Map new coordinates back to old coordinates
            old_x = new_x / scale_factor
            old_y = new_y / scale_factor
            
            # Use nearest neighbor
            old_x_int = int(round(old_x))
            old_y_int = int(round(old_y))
            
            # Bounds checking
            if 0 <= old_x_int < old_width and 0 <= old_y_int < old_height:
                converted_grid[new_y, new_x] = occupancy_grid[old_y_int, old_x_int]
    
    print(f"       Final BARN grid: ({converted_grid.shape[0]}, {converted_grid.shape[1]}) @ {target_resolution}m/cell")
    print(f"       World coverage: {converted_grid.shape[0] * target_resolution:.1f}m × {converted_grid.shape[1] * target_resolution:.1f}m")
    
    return converted_grid