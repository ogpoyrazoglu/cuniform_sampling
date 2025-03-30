#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import numpy as np
import yaml
import glob
from PIL import Image
import pickle
import os

class MapVisualizer:
    def __init__(self, yaml_path, manual_obstacles=None):
        self.manual_obstacles = manual_obstacles or []
        self.load_map_data(yaml_path)
        
    def load_map_data(self, yaml_path):
        with open(yaml_path) as f: # Load YAML file
            self.map_metadata = yaml.safe_load(f)
        
        # Load and process PGM image
        map_dir = os.path.dirname(yaml_path)
        pgm_path = os.path.join(map_dir, self.map_metadata['image'])
        
        # Flip image vertically to match ROS coordinate convention
        self.map_image = np.flipud(np.array(Image.open(pgm_path)))
        
        # Extract map parameters
        self.resolution = self.map_metadata['resolution']
        self.origin = np.array(self.map_metadata['origin'][:2])  # [x, y] in meters
        self.height, self.width = self.map_image.shape

    def plot_map(self, ax):
        """Plot the occupancy grid map in world coordinates"""
        # Calculate map bounds in meters
        x_min = self.origin[0]
        x_max = self.origin[0] + self.width * self.resolution
        y_min = self.origin[1]
        y_max = self.origin[1] + self.height * self.resolution

        # Define the cropping range (restrict visualization to -10 <= x <= 5)
        crop_x_min = -10
        crop_x_max = 7.5

        # Convert world coordinates to pixel indices
        pixel_x_min = int((crop_x_min - self.origin[0]) / self.resolution)
        pixel_x_max = int((crop_x_max - self.origin[0]) / self.resolution)

        # Ensure valid cropping indices within the image bounds
        pixel_x_min = max(0, pixel_x_min)
        pixel_x_max = min(self.width, pixel_x_max)

        # Crop the image along the X-axis
        cropped_image = self.map_image[:, pixel_x_min:pixel_x_max]

        # Update new world extents after cropping
        new_x_min = crop_x_min
        new_x_max = crop_x_max

        ax.imshow(cropped_image,
                cmap='gray',
                extent=[new_x_min, new_x_max, y_min, y_max],  # New extent
                origin='lower',
                vmin=0, vmax=100,
                aspect='equal')

        # Set new limits to enforce cropping
        ax.set_xlim(new_x_min, new_x_max)

    def detect_crash(self, traj):
        # Assume that ending within a certain boundary is a success
        goal_x, goal_y = -1, -15  # Change to actual goal coordinates
        distance_to_goal = np.hypot(traj[-1][0] - goal_x, traj[-1][1] - goal_y)
        return distance_to_goal > 1.5  # Mark as crash if far from goal

    def plot_trajectories(self, ax, trajectories, color):
        """Plot robot trajectories in world coordinates"""
        for traj in trajectories:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, linewidth=0.9, alpha=0.85, color=color)
            is_crash = self.detect_crash(traj)
            if is_crash:
                ax.scatter(xs[-1], ys[-1], alpha=0.98, marker='X', color=color, s=90)
            # else:
            #     ax.scatter(xs[-1], ys[-1], marker='^', color=color, s=50)

    def plot_obstacles(self, ax):
        """Plot manual obstacles in world coordinates"""
        for obs in self.manual_obstacles:
            ax.add_patch(plt.Rectangle(
                (obs['x'], obs['y']),
                obs['width'],
                obs['height'],
                angle=obs['angle'],
                color='black',
                alpha= 1.0,
                linewidth=0.1
            ))
        ax.scatter([], [], marker='s', color='black', s=100, label='Obstacle')

    def visualize(self, trajectories):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot map background
        self.plot_map(ax)
        
        self.plot_trajectories(ax, trajectories)
        
        # Plot manual obstacles
        self.plot_obstacles(ax)
        
        ax.scatter(-1, -15, color='gold', marker='*', s=200, zorder=5, label='Goal')

        # Configure axes
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        # ax.set_title('Robot Trajectories', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio and tight bounds
        ax.set_aspect('equal')
        ax.autoscale(tight=True)
        
        # Add legend (only once)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))  # Deduplicate
        ax.legend(unique.values(), unique.keys(), loc='upper right')

        plt.show()

    def analyze_trajectories(self, all_trajectories):
        """Analyze trajectories that both start and end within y ∈ [-14, -2]"""
        contained_trajs = []
        for traj in all_trajectories:
            is_crash = self.detect_crash(traj)
            if not is_crash:
                find_initial_y = False
                find_last_y = False
                initial_index = 0
                last_index = 0
                if not traj or len(traj) < 2:  # Skip invalid trajectories
                    continue
                # Get first and last points in world coordinates
                for i in range(len(traj)):
                    y = traj[i][1]
                    if (not find_initial_y and -14 <= y <= -2.0):
                        find_initial_y = True
                        initial_index = i   
                for neg_i in range(len(traj)-1, -1, -1):
                    y = traj[neg_i][1]
                    if (not find_last_y and -14 <= y <= -2.0):
                        find_last_y = True
                        last_index = neg_i  
                filtered_traj = traj[initial_index:last_index+1]
                filtered_traj[0] = (traj[initial_index][0], -2.0)
                filtered_traj[-1] = (traj[last_index][0], -14.0)
                contained_trajs.append(filtered_traj)
        # Calculate trajectory lengths within zone
        lengths = []
        for traj in contained_trajs:
            path_length = 0.0
            for i in range(len(traj)-1):
                dx = traj[i+1][0] - traj[i][0]
                dy = traj[i+1][1] - traj[i][1]
                path_length += np.hypot(dx, dy)
            lengths.append(path_length)

        total_attempts = len(all_trajectories)
        success_rate = len(contained_trajs)/total_attempts if total_attempts > 0 else 0
        stats = {
            'total_attempts': total_attempts,
            'successful_trajectories': len(contained_trajs),
            'success_rate': success_rate,
            'avg_length': np.mean(lengths) if lengths else 0,
            'median_length': np.median(lengths) if lengths else 0,
            'max_length': np.max(lengths) if lengths else 0,
            'min_length': np.min(lengths) if lengths else 0
        }
        
        print(f"\nContained Trajectory Analysis (y ∈ [-14, -2])")
        print(f"Attempts: {stats['total_attempts']} | Successes: {stats['successful_trajectories']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Path Lengths [m] - Avg: {stats['avg_length']:.2f} | Median: {stats['median_length']:.2f}")
        print(f"Range: {stats['min_length']:.2f}-{stats['max_length']:.2f}")

        return stats

def main(args=None):
    base_paths = [
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_289/cuniform_mppi/mppi_0.05",
        "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_289/cuniform_mppi/mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_289/cuniform_mppi/log_mppi_0.05",
        "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_289/cuniform_mppi/log_mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_289/mppi_0.05",
        "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_289/mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_289/log_mppi_0.05",
        "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_289/log_mppi_0.1",
        # env110 below
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_110/cuniform_mppi_0.05",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_110/cuniform_mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_110/cuniform_log_mppi_0.05",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_110/cuniform_log_mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_110/mppi_0.05",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_110/mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_110/log_mppi_0.05",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_110/log_mppi_0.1",
        # env21 below
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/cuniform_mppi_0.05",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/cuniform_mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/cuniform_log_mppi_0.05",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/cuniform_log_mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/mppi_0.05",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/log_mppi_0.05",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/log_mppi_0.1",
        # "/home/nvidia/f1tenth_ws/experiments_data/iros/env_barn_21/stein_0.1_2iter",
    ]
    controller_names = [
        "CU-MPPI",
        "CU-LogMPPI",
        "MPPI",
        "Log-MPPI",
    ]  # Adjust names as desired

    # Dictionary to store trajectories for each controller setting
    trajectories_dict = {}
    for base_path, name in zip(base_paths, controller_names):
        traj_files = sorted(glob.glob(os.path.join(base_path, "trajectory_*.pkl")))
        trajectories = []
        for file in traj_files:
            with open(file, 'rb') as f:
                trajectories.append(pickle.load(f))
        trajectories_dict[name] = trajectories[7:]
        print(f"Name: {name}")

    all_trajectories = []
    traj_files = sorted(glob.glob(os.path.join(base_path, "trajectory_*.pkl")))

    for file in traj_files:
        with open(file, 'rb') as f:
            all_trajectories.append(pickle.load(f))  # Append each trajectory set

    manual_obstacles = [
        {'x': -0.5, 'y': -7.3, 'width': 0.31, 'height': 0.15, 'angle':45.0}, # Wooden Square 1
        {'x': -1.37, 'y': -9.2, 'width': 0.31, 'height': 0.15, 'angle':20.0}, # Wooden Square 2
        {'x': -2.07, 'y': -9.55, 'width': 0.31, 'height': 0.15, 'angle':0.0}, # Wooden Square 3
        {'x': -2.88, 'y': -10.3, 'width': 0.31, 'height': 0.15, 'angle':-50.0}, # Wooden Square 4
        {'x': -1.75, 'y': -12.7, 'width': 0.31, 'height': 0.15, 'angle':0.0}, # (FAR)Wooden Square 5
        {'x': -2.5, 'y': -8.6, 'width': 0.72, 'height': 0.25, 'angle':90.0}, # Long Right Box
        {'x': -1.0, 'y': -7.8, 'width': 0.57, 'height': 0.18, 'angle':0.0},  # Skinny Front Left Box
        {'x': -1.37, 'y': -10.5, 'width': 0.60, 'height': 0.35, 'angle':90.0}, # Square Middle Box
        {'x': -1.01, 'y': -10.0, 'width': 0.50, 'height': 0.18, 'angle':0.0}, # Long middle left Box
        {'x': -0.75, 'y': -12.2, 'width': 0.11,	 'height': 0.11, 'angle':0.0}, # Small Back Box
        {'x': -2.2, 'y': -12.1, 'width': 0.25, 'height': 0.25, 'angle':0.0} # Back right square Box
    ]
    visualizer = MapVisualizer(
        yaml_path="/home/nvidia/f1tenth_ws/maps/shepherd_lab_map_cleaned.yaml",
        manual_obstacles=manual_obstacles
    )
    # analysis_results = visualizer.analyze_trajectories(all_trajectories)
    # visualizer.visualize(all_trajectories)  # Pass all trajectories together

    # return
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    visualizer.plot_map(ax)
    
    # Assign colors
    # colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(trajectories_dict)))
    colors = ['#e6194B', '#ffe119', '#3cb44b', '#4363d8', '#f032e6']
    # colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    for (name, trajs), color in zip(trajectories_dict.items(), colors):
        visualizer.plot_trajectories(ax, trajs, color=color)
        ax.plot([], [], color=color, linewidth=2.5, label=name)
    
    # Add legend markers
    # ax.scatter([], [], marker='o', color='green', s=50, label='Start')
    # ax.scatter([], [], marker='^', color='blue', s=50, label='Success')
    # ax.scatter(-1, -15, color='gold', marker='*', s=200, zorder=5, label='Goal')
    from matplotlib.patches import Circle
    circle = Circle((-1.29, -1.05), 0.4, edgecolor='olive', facecolor='none', linewidth=2, zorder=10)
    ax.add_patch(circle)
    # ax.scatter([], [], marker='o', color='olive', s=50, label='Start')

    circle_goal = Circle((-1, -15), 0.3, edgecolor='Magenta', facecolor='none', linewidth=2, zorder=10)
    ax.add_patch(circle_goal)
    ax.scatter([], [], facecolors='none', edgecolors='Magenta', marker='o', s=50, label='Goal')
    visualizer.plot_obstacles(ax)
    # ax.scatter([], [], marker='X', color='black', s=100, label='Crash\n(color = controller)')
    ax.scatter([], [], marker='X', color='black', s=100, label='Crash')

    # generic_handles = [
    #     Line2D([0], [0], color='black', lw=2, label='Trajectory (color = controller)'),
    #     Line2D([0], [0], marker='X', color='black', lw=0, markersize=8, label='Crash'),
    #     Line2D([0], [0], marker='o', color='Magenta', lw=0, markersize=8, label='Goal'),
    #     Line2D([0], [0], marker='s', color='black', lw=0, markersize=8, label='Obstacle'),
    # ]
    # ax.legend(handles=generic_handles, loc='lower left', ncol=1, fontsize='small')
    
    # Configure plot
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    # ax.set_title('Robot Trajectories for Different Controllers', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.autoscale(tight=True)
    # ax.legend(loc='upper right')
    ax.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
    main()
