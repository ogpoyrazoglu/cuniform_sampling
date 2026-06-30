"""Visualization utilities for experiment results."""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt display issues
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List
import os
from utils.geometry_utils import get_vehicle_boundary_points


def visualize_trajectory(trajectory: np.ndarray, controls: np.ndarray, 
                        occupancy_grid: np.ndarray, goal: np.ndarray,
                        start: np.ndarray, success: bool, termination_reason: str,
                        save_path: str, title: str = "Trajectory Visualization",
                        experiment_config: dict = None) -> None:
    """Create and save trajectory visualization.
    
    Args:
        trajectory: Array of [x, y, theta, v] states
        controls: Array of [v_cmd, steer_angle] controls
        occupancy_grid: Environment occupancy grid
        goal: [x, y] goal position
        start: [x, y] start position
        success: Whether the trial was successful
        termination_reason: Reason for termination
        save_path: Path to save the visualization
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot occupancy grid with red obstacles
    from matplotlib.colors import ListedColormap
    colors = ['white', 'red']
    cmap = ListedColormap(colors)
    
    # Plot map with proper extent
    resolution = experiment_config['global_map_resolution']
    origin = experiment_config['global_origin']
    grid_height, grid_width = occupancy_grid.shape
    x_coords = np.arange(grid_width) * resolution + origin[0]
    y_coords = np.arange(grid_height) * resolution + origin[1]
    
    ax.imshow(occupancy_grid, cmap=cmap, origin='lower', 
              extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    
    # Plot trajectory (blue)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=3, label='Robot Path')
    
    # Plot start (blue circle) and goal (yellow star)
    ax.plot(start[0], start[1], 'o', color='blue', markersize=10, 
            markeredgecolor='black', markeredgewidth=1, label='Start')
    ax.plot(goal[0], goal[1], '*', color='yellow', markersize=15, 
            markeredgecolor='black', markeredgewidth=1, label='Goal')
    
    # Add obstacles legend entry (using red square)
    ax.plot([], [], 's', color='red', markersize=8, label='Obstacles')
    
    # Plot vehicle at final position
    if len(trajectory) > 0 and experiment_config is not None:
        final_state = trajectory[-1]
        plot_vehicle(ax, final_state, experiment_config, color='blue')
    
    status_text = "Success" if success else termination_reason.replace('_', ' ').title()
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f"{title} - {status_text}", fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    padding = 1.0  # meters
    grid_height, grid_width = occupancy_grid.shape
    map_x_min = origin[0]
    map_x_max = origin[0] + grid_width * resolution
    map_y_min = origin[1]
    map_y_max = origin[1] + grid_height * resolution

    ax.set_xlim(map_x_min - padding, map_x_max + padding)
    ax.set_ylim(map_y_min - padding, map_y_max + padding)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {save_path}")


def create_trajectory_animation(trajectory: np.ndarray, controls: np.ndarray,
                               occupancy_grid: np.ndarray, goal: np.ndarray,
                               start: np.ndarray, success: bool,
                               termination_reason: str,
                               sampled_trajectories: List[np.ndarray],
                               costmaps_history: List[np.ndarray],
                               save_path: str, dt: float,
                               experiment_config: dict = None,
                               visualize_costmap: bool = False,
                               visualize_all_trajectories: bool = True,
                               title: str = "Trajectory Animation") -> None:
    """Create animated visualization showing trajectory sampling and robot movement.
    
    Args:
        trajectory: Array of [x, y, theta, v] states
        controls: Array of [v_cmd, steer_angle] controls  
        occupancy_grid: Environment occupancy grid
        goal: [x, y] goal position
        start: [x, y] start position
        success: Whether the trial was successful
        termination_reason: Reason for termination
        sampled_trajectories: List of sampled trajectory arrays for each step
        costmap_history: List of costmaps for each step
        save_path: Path to save the MP4 animation
        dt: Time step in seconds (for real-time animation timing)
        experiment_config: Experiment configuration dictionary (required)
        visualize_costmap: Whether to show costmap overlay
        visualize_all_trajectories: Whether to show all sampled trajectories
        title: Animation title
    """
    # Require experiment_config - no fallback values
    if experiment_config is None:
        raise ValueError("experiment_config is required - no fallback values allowed")
    
    # Get parameters from config instead of hardcoding
    required_keys = ['global_map_resolution', 'global_origin', 'vehicle_length', 'vehicle_width']
    for key in required_keys:
        if key not in experiment_config:
            raise ValueError(f"Missing required experiment_config key: {key}")
    
    resolution = experiment_config['global_map_resolution']
    origin = experiment_config['global_origin']
    vehicle_length = experiment_config['vehicle_length']
    vehicle_width = experiment_config['vehicle_width']
    use_point_robot = experiment_config['point_robot_mode']
    
    # Create figure and axis with white background
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot occupancy grid with red obstacles
    # Create custom colormap: white for free space, red for obstacles
    from matplotlib.colors import ListedColormap
    colors = ['white', 'red']
    cmap = ListedColormap(colors)
    
    # Plot map with proper extent
    grid_height, grid_width = occupancy_grid.shape
    x_coords = np.arange(grid_width) * resolution + origin[0]
    y_coords = np.arange(grid_height) * resolution + origin[1]
    
    ax.imshow(occupancy_grid, cmap=cmap, origin='lower', 
              extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    
    # Plot goal as yellow star
    ax.plot(goal[0], goal[1], '*', color='yellow', markersize=15, 
            markeredgecolor='black', markeredgewidth=1, label='Goal')
    
    # Plot start as blue circle
    ax.plot(start[0], start[1], 'o', color='blue', markersize=10, 
            markeredgecolor='black', markeredgewidth=1, label='Start')
    
    # Add obstacles legend entry (using red square)
    ax.plot([], [], 's', color='red', markersize=8, label='Obstacles')
    
    # Initialize plot elements
    robot_path_line, = ax.plot([], [], 'b-', linewidth=3, label='Robot Path')

    # Use a single list to manage all dynamically created elements (lines, patches, scatters)
    dynamic_elements = []
    # Create vehicle representation (Polygon or Circle)
    if use_point_robot:
        # Use a Circle patch. Radius slightly larger than resolution for visibility.
        vis_radius = max(0.1, resolution * 1.5)
        vehicle_patch = patches.Circle((0, 0), radius=vis_radius, color='blue', alpha=0.7, zorder=5)
    else: # Use a Polygon patch
        vehicle_patch = patches.Polygon(np.zeros((4, 2)), closed=True, color='blue', alpha=0.7, zorder=5)
    ax.add_patch(vehicle_patch)
    
    # Add costmap legend entry if costmap visualization is enabled
    if visualize_costmap:
        # Use a Patch for the circular range area
        lidar_range_patch = patches.Patch(color='lightblue', alpha=0.3, label='LiDAR Range')
        # Use a line with a marker for the detection points
        lidar_hits_line, = ax.plot([], [], 'o', color='darkblue', markersize=4, label='LiDAR Detections')
        # Manually add the patch to the legend handles
        handles, labels = ax.get_legend_handles_labels()
        handles.append(lidar_range_patch)
        ax.legend(handles=handles, loc='upper right', fontsize=10)
    else:
        ax.legend(loc='upper right', fontsize=10)
    
    # Initialize dynamic costmap overlay (will be updated each frame)
    costmap_overlay = None
    
    # Set plot properties
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    padding = 1.0  # meters
    grid_height, grid_width = occupancy_grid.shape
    map_x_min = origin[0]
    map_x_max = origin[0] + grid_width * resolution
    map_y_min = origin[1]
    map_y_max = origin[1] + grid_height * resolution

    ax.set_xlim(map_x_min - padding, map_x_max + padding)
    ax.set_ylim(map_y_min - padding, map_y_max + padding)
    
    def animate(step):
        nonlocal costmap_overlay
        for element in dynamic_elements:
            # Check if the element has a remove method (handles lines, patches, scatters)
            if hasattr(element, 'remove'):
                try:
                    element.remove()
                except Exception:
                    pass # Handle cases where element might already be detached
        dynamic_elements.clear()

        # Draw new LiDAR visualization for the current step
        if visualize_costmap and costmaps_history and step < len(costmaps_history):
            robot_state = trajectory[step]
            robot_pos = robot_state[:2]
            robot_yaw = robot_state[2]
            
            # 1. Draw the circular LiDAR range
            lidar_range = experiment_config['lidar_scan_range']
            lidar_circle = patches.Circle(
                robot_pos, lidar_range,
                facecolor='lightblue', alpha=0.3, zorder=1.5
            )
            ax.add_patch(lidar_circle)
            dynamic_elements.append(lidar_circle)
            
            # 2. Draw the individual LiDAR hit points
            costmap = costmaps_history[step]
            if np.any(costmap > 0):
                # Find the grid coordinates [row, col] of all hits
                rows, cols = np.where(costmap > 0.5)
                
                # Convert grid indices back to robot-frame coordinates (in meters)
                center = costmap.shape[0] // 2
                x_robot = (cols - center) * resolution
                y_robot = (center - rows) * resolution # Reverse the y-flip
                
                # Transform robot-frame points to world-frame
                cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
                x_world = robot_pos[0] + x_robot * cos_yaw - y_robot * sin_yaw
                y_world = robot_pos[1] + x_robot * sin_yaw + y_robot * cos_yaw
                
                # Plot all hits with a single, efficient scatter call
                hit_scatter = ax.scatter(x_world, y_world, c='darkblue', s=8, zorder=4)
                dynamic_elements.append(hit_scatter)
        current_step_data = sampled_trajectories[step] if sampled_trajectories and step < len(sampled_trajectories) else None

        # Check if this step uses the hybrid visualization structure
        if isinstance(current_step_data, dict) and current_step_data.get('is_hybrid', False):
            # --- Hybrid CU-MPPI Visualization ---
            
            # 1. C-Uniform Samples (Gray)
            if visualize_all_trajectories and current_step_data['cu_samples'].size > 0:
                for traj in current_step_data['cu_samples']:
                    if len(traj) > 1:
                        # Color: Gray, Alpha: 0.3, Z-order: 2.5
                        line, = ax.plot(traj[:, 0], traj[:, 1], color='gray',
                                        alpha=0.3, linewidth=0.5, zorder=2.5)
                        dynamic_elements.append(line)

            # 2. Best C-Uniform Trajectory (Initialization) (Cyan)
            traj = current_step_data['cu_best']
            if traj.size > 1:
                # Color: Cyan (or Orange), Alpha: 0.8, Z-order: 3.0
                line, = ax.plot(traj[:, 0], traj[:, 1], color='cyan', linestyle='--',
                                alpha=0.8, linewidth=1.5, zorder=3.0)
                dynamic_elements.append(line)

            # 3. MPPI Samples (Magenta)
            if visualize_all_trajectories and current_step_data['mppi_samples'].size > 0:
                for traj in current_step_data['mppi_samples']:
                    if len(traj) > 1:
                        # Color: Magenta, Alpha: 0.4, Z-order: 3.5
                        line, = ax.plot(traj[:, 0], traj[:, 1], color='magenta',
                                        alpha=0.4, linewidth=1.0, zorder=3.5)
                        dynamic_elements.append(line)

            # 4. Final MPPI Nominal Trajectory (Execution) (Green)
            traj = current_step_data['mppi_nominal']
            if traj.size > 1:
                # Color: Green, Alpha: 0.9, Z-order: 4.0
                line, = ax.plot(traj[:, 0], traj[:, 1], color='green',
                                alpha=0.9, linewidth=2.0, zorder=4.0)
                dynamic_elements.append(line)

        elif current_step_data is not None:
            # --- Standard Visualization (C-Uniform or MPPI) ---
            # This maintains compatibility with existing controllers.

            if visualize_all_trajectories:
                # Plot all samples (Black)
                for traj in current_step_data:
                    if len(traj) > 1:
                        line, = ax.plot(traj[:, 0], traj[:, 1], 'k-',
                                        alpha=0.3, linewidth=1, zorder=3)
                        dynamic_elements.append(line)
            
            # Plot minimum cost / nominal trajectory (Green)
            # In standard visualization, the best/nominal is always at index 0.
            if len(current_step_data) > 0:
                min_cost_traj = current_step_data[0]
                if len(min_cost_traj) > 1:
                    min_line, = ax.plot(min_cost_traj[:, 0], min_cost_traj[:, 1],
                                        'g-', linewidth=2, alpha=0.8, zorder=4)
                    dynamic_elements.append(min_line)
        
        # Update robot path (blue)
        if step < len(trajectory):
            # Plot robot path so far
            robot_path_line.set_data(trajectory[:step+1, 0], trajectory[:step+1, 1])
            
            # Update current robot position
            current_pos = trajectory[step]
            if use_point_robot: # Update center of the Circle patch
                vehicle_patch.center = (current_pos[0], current_pos[1])
            else: # Update vertices of the Polygon patch
                boundary_points = get_vehicle_boundary_points(current_pos, vehicle_length, vehicle_width)
                vehicle_patch.set_xy(boundary_points)
        
        # Update title with current step and add costmap info
        status_text = "Success" if success else termination_reason.replace('_', ' ').title()
        title_suffix = f" - Step {step+1}/{len(trajectory)} - {status_text}"
        if visualize_costmap and costmap_overlay is not None:
            title_suffix += " (Costmap: ON)"
        ax.set_title(f"{title}{title_suffix}", fontsize=14, fontweight='bold')
        
        elements_to_return = ([robot_path_line, vehicle_patch] + dynamic_elements)
        return elements_to_return
    
    # Calculate real-time animation parameters
    interval_ms = int(dt * 1000)  # Convert dt to milliseconds
    fps = 1.0 / dt  # Frames per second for real-time playback
    
    # Create animation (one frame per step, real-time timing)
    anim = FuncAnimation(fig, animate, frames=len(trajectory), 
                        interval=interval_ms, blit=False, repeat=False)
    
    # Save as MP4 with real-time fps
    print(f"Creating real-time animation: {save_path} (dt={dt}s, fps={fps:.1f})")
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=150)
    plt.close()
    print(f"Animation saved: {save_path}")


def plot_vehicle(ax, state: np.ndarray, experiment_config: dict, color: str = 'blue') -> None:
    """ Plots the vehicle as a polygon using the ground-truth boundary points
    Args:
        ax: Matplotlib axis
        state: [x, y, theta, v] vehicle state
        experiment_config: Experiment configuration dictionary
        color: Vehicle color
    """
    use_point_robot = experiment_config['point_robot_mode']
    if use_point_robot: # Draw as a circle (point)
        resolution = experiment_config['global_map_resolution']
        vis_radius = max(0.1, resolution * 1.5)
        vehicle_circle = patches.Circle((state[0], state[1]), radius=vis_radius, color=color, alpha=0.7, zorder=5)
        ax.add_patch(vehicle_circle)
    else: # Draw as a polygon 
        vehicle_length = experiment_config['vehicle_length']
        vehicle_width = experiment_config['vehicle_width']
        boundary_points = get_vehicle_boundary_points(state, vehicle_length, vehicle_width)
        vehicle_poly = patches.Polygon(boundary_points, closed=True, color=color, alpha=0.7)
        ax.add_patch(vehicle_poly)

def save_enhanced_visualization(trajectory: np.ndarray, controls: np.ndarray,
                               occupancy_grid: np.ndarray, goal: np.ndarray,
                               start: np.ndarray, success: bool,
                               termination_reason: str,
                               sampled_trajectories: List[np.ndarray],
                               save_dir: str, controller_name: str, 
                               environment_id: int, dt: float,
                               costmaps_history: List[np.ndarray],
                               experiment_config: dict = None,
                               visualize_costmap: bool = False,
                               visualize_all_trajectories: bool = True,
                               save_animations: bool = True) -> None:
    """Save static trajectory plot and animated visualization.
    
    Args:
        trajectory: Array of [x, y, theta, v] states
        controls: Array of [v_cmd, steer_angle] controls
        occupancy_grid: Environment occupancy grid
        goal: [x, y] goal position
        start: [x, y] start position
        success: Whether the trial was successful
        termination_reason: Reason for termination
        sampled_trajectories: List of sampled trajectory arrays for each step
        save_dir: Directory to save visualizations
        controller_name: Name of controller for filename
        environment_id: Environment ID for filename
        dt: Time step for animation timing
        experiment_config: Experiment configuration dictionary (required)
        local_costmap: Local costmap for visualization (optional)
        robot_position: Robot position for costmap centering (optional)
        visualize_costmap: Whether to show costmap overlay
        visualize_all_trajectories: Whether to show all sampled trajectories
    """
    # BRITTLE: Require experiment_config - no fallback values
    if experiment_config is None:
        raise ValueError("experiment_config is required - no fallback values")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Static trajectory visualization
    static_path = os.path.join(save_dir, f"{controller_name}_env{environment_id}_trajectory.png")
    visualize_trajectory(trajectory, controls, occupancy_grid, goal, start, success, 
                        termination_reason, static_path, f"{controller_name.upper()} - Environment {environment_id}",
                        experiment_config=experiment_config)
    
    if save_animations:
        # Animated trajectory visualization
        animation_path = os.path.join(save_dir, f"{controller_name}_env{environment_id}_animation.mp4")
        create_trajectory_animation(
            trajectory, controls, occupancy_grid, goal, start, success,
            termination_reason,
            sampled_trajectories, costmaps_history, animation_path, dt,
            experiment_config=experiment_config,
            visualize_costmap=visualize_costmap,
            visualize_all_trajectories=visualize_all_trajectories,
            title=f"{controller_name.upper()}"
        )