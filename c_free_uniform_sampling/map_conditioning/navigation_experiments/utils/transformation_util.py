import torch
import numpy as np

def transform_trajectories_to_world_frame(robot_frame_trajectories: torch.Tensor, 
                                            current_world_pose: np.ndarray) -> torch.Tensor:
    """Transform trajectories from robot frame to world frame.
    
    Args:
        robot_frame_trajectories: [trajectory_length+1, num_trajectories, 3] tensor in robot frame
        x_curr: [3] array with current robot pose in world frame [x, y, theta]
        
    Returns:
        world_frame_trajectories: [trajectory_length+1, num_trajectories, 3] tensor in world frame
    """
    # Extract current world pose
    world_x, world_y, world_theta = current_world_pose
    
    # Get trajectory positions in robot frame
    robot_x = robot_frame_trajectories[:, :, 0]  # [trajectory_length+1, num_trajectories]
    robot_y = robot_frame_trajectories[:, :, 1]  # [trajectory_length+1, num_trajectories]
    robot_theta = robot_frame_trajectories[:, :, 2]  # [trajectory_length+1, num_trajectories]
    
    # Transform positions to world frame using rotation and translation
    cos_theta = torch.cos(torch.tensor(world_theta, device=robot_frame_trajectories.device))
    sin_theta = torch.sin(torch.tensor(world_theta, device=robot_frame_trajectories.device))
    
    # Rotate and translate positions
    world_x_traj = world_x + robot_x * cos_theta - robot_y * sin_theta
    world_y_traj = world_y + robot_x * sin_theta + robot_y * cos_theta
    
    # Transform orientations (add current world orientation)
    world_theta_traj = robot_theta + world_theta
    
    # Normalize angles to [-pi, pi]
    world_theta_traj = torch.atan2(torch.sin(world_theta_traj), torch.cos(world_theta_traj))
    
    # Stack back into trajectory format
    world_frame_trajectories = torch.stack([world_x_traj, world_y_traj, world_theta_traj], dim=2)
    
    return world_frame_trajectories