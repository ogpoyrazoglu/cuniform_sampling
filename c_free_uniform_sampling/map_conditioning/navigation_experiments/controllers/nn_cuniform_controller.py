"""
Neural C-Uniform controllers using PyTorch.
   0: Unsupervised NN C-Uniform, 
   1: Supervised Map conditioned NN C-Uniform
Relies on TorchPlannerBase for standardized dynamics and cost evaluation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import sys
import time
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from .torch_planner_base import TorchPlannerBase
from utils.transformation_util import transform_trajectories_to_world_frame

# Add path for BARN_navigation_experiments utils
barn_utils_path = os.path.join(os.path.dirname(__file__), '..')
if barn_utils_path not in sys.path:
    sys.path.append(barn_utils_path)
from utils.dynamics import cuda_dynamics_KS_3d_scalar_v_batched as dynamics_scalar_torch


# Add parent directory to path to import model classes
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from map_conditioning.model_parts import MapPixelFeature, MapAct_PixelInterpolated, MapAct

# Debug timestamp counter
_debug_timestamp = 0

def load_compiled_state_dict(model, path):
    """Loads a state_dict saved from a torch.compile'd model."""
    state_dict = torch.load(path, map_location='cuda', weights_only=True)
    
    # Use a dictionary comprehension to strip the '_orig_mod.' prefix
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    print(f"Successfully loaded compiled weights into {model.__class__.__name__} from {path}")

class CUniformController(TorchPlannerBase):
    """C-Uniform controller with neural network guidance."""
    
    def __init__(self, controller_config: dict, experiment_config: dict = None, type_override: int = None, seed: int = None):
        """Initialize C-Uniform controller."""
        if seed is None:
            seed = experiment_config.get('seed', 2025) if experiment_config else 2025
        super().__init__(controller_config, experiment_config, seed)

        # 0: Unsupervised, 1: Supervised and map conditioned
        # Allow controller_type to be overridden, otherwise use the value from the config file
        self.controller_type = type_override if type_override is not None else self.config['controller_type']
        
        self.num_a = self.config['num_a']
        self.num_steering_angle = self.config['num_steering_angle']
        self.arange = self.config['arange'] 

        self.feature_extractor_path = self.config['feature_extractor_path'] if self.controller_type == 1 else None
        self.model_path = (self.config['map_conditioned_model_path'] if self.controller_type == 1 
                          else self.config['unsupervised_cuniform_model_path'])

        self.rng = np.random.RandomState(self.seed)
        self.actions = self.generate_actions(
            self.arange, self.num_a, self.wrange, self.num_steering_angle, False
        )
        
        # Load models if paths are provided - instantiate classes first, then load state dicts
        self.model = None
        self.feature_extractor = None
        
        if self.model_path and os.path.exists(self.model_path):
            # Load model based on controller type
            if self.controller_type == 0:
                # Unsupervised model - load full model object
                # Add path for model_parts module
                map_conditioning_path = os.path.join(os.path.dirname(__file__), '../..')
                if map_conditioning_path not in sys.path:
                    sys.path.append(map_conditioning_path)
                self.model = torch.load(self.model_path, map_location='cuda', weights_only=False)
            else:
                # Map-conditioned model - instantiate and load state dict
                num_actions = len(self.actions)
                self.model = MapAct_PixelInterpolated(num_actions=num_actions) # if the saved model is compiled, use this
                # self.model.load_state_dict(torch.load(self.model_path, map_location='cuda', weights_only=True))
                load_compiled_state_dict(self.model, self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded {'unsupervised' if self.controller_type == 0 else 'map-conditioned'} model from: {self.model_path}")
        
        # Only load feature extractor if controller_type == 1 (supervised mode)
        if self.controller_type == 1 and self.feature_extractor_path and os.path.exists(self.feature_extractor_path):
            # Instantiate feature extractor
            self.feature_extractor = MapPixelFeature()
            # Load state dictionary
            # self.feature_extractor.load_state_dict(torch.load(self.feature_extractor_path, map_location='cuda', weights_only=True))
            load_compiled_state_dict(self.feature_extractor, self.feature_extractor_path) # if the saved model is compiled, use this

            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            print(f"Loaded feature extractor from: {self.feature_extractor_path}")
            
        # Planning parameters - get from experiment config
        self.num_trajectories = self.num_rollouts  # Use shared parameter from experiment config
        self.trajectory_length = self.num_steps     # Use shared parameter from experiment config
        self.num_vis_trajectories = min(self.num_vis_rollouts, self.num_trajectories)  # Limit visualization like MPPI
        
        # Storage for visualization
        self.sampled_trajectories_history = []
        self.current_state = np.zeros(3, dtype=np.float32)  # [x, y, theta]
        
        print(f"C-Uniform Controller initialized (type: {self.controller_type})")
    
    def reset(self):
        """Reset controller state for new environment."""
        super().reset()
        self.sampled_trajectories_history = []
        self.current_state = np.zeros(3, dtype=np.float32)  # reset current state
        self.rng = np.random.RandomState(self.seed)  # Reset generator for reproducibility
    
    def generate_actions(self, arange, num_a, steering_angle_range, num_steering_angle, deg2rad_conversion):
        """
        Generates a NumPy array of all possible (steering, velocity) action pairs
        based on the given ranges and discretization parameters.
        Returns: np.ndarray: Array of shape (num_steering_angle * num_v, 2) where each row is [steering, acceleration]
        """
        if deg2rad_conversion:
            steering_values = np.deg2rad(np.linspace(steering_angle_range[0], steering_angle_range[1], num_steering_angle))
        else:
            steering_values = np.linspace(steering_angle_range[0], steering_angle_range[1], num_steering_angle)
        a_value = np.linspace(arange[0], arange[1], num_a)
        actions = np.array([[s, a] for s in steering_values for a in a_value], dtype=np.float32)
        return actions
    
    def get_control_action(self, current_state: np.ndarray, goal_state: np.ndarray, 
                          global_occupancy_grid: np.ndarray, dt: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute control action using C-Uniform sampling.
        
        Args:
            current_state: [x, y, theta, v] robot state
            goal_state: [x, y] target position
            global_occupancy_grid: 2D global environment occupancy grid
            dt: Time step for planning
            
        Returns:
            Tuple of:
                - Control action [velocity, angular_velocity]
                - Info dictionary for visualization and debugging
        """
        # Store current robot state for proper trajectory visualization
        self.current_state = current_state[:3].astype(np.float32)  # [x, y, theta]
        
        # Prepare model inputs based on controller type (this will generate the LiDAR-based costmap)
        sdf_tensor, feature_extractor = self._prepare_model_inputs(global_occupancy_grid, current_state)
        
        robot_frame_trajectories, trajectory_actions = self._sample_trajectories_robot_frame(
            current_state, sdf_tensor, feature_extractor
        )
        
        control_action, best_trajectory_idx = self._evaluate_robot_frame_trajectories(
            robot_frame_trajectories, trajectory_actions, goal_state, current_state[:3]
        )
        
        # Quick robot frame visualization for debugging
        # self._debug_robot_frame_trajectories(robot_frame_trajectories, best_trajectory_idx, goal_state, current_state[:3])
        
        world_frame_trajectories = transform_trajectories_to_world_frame(
            robot_frame_trajectories, current_state[:3]
        )
        
        # Prepare visualization data
        info = self._prepare_visualization_info(world_frame_trajectories, trajectory_actions, best_trajectory_idx)
        return control_action, info
    
    def _prepare_model_inputs(self, global_occupancy_grid: np.ndarray, current_state: np.ndarray) -> Tuple:
        """
        Prepare model inputs by calling the base perception module and storing the results.
        """
        # Call the base method to handle perception generation and storage of costmaps
        perception_data = super()._prepare_perception_inputs(global_occupancy_grid, current_state)

        # For supervised mode, return the SDF tensor and feature extractor
        if self.controller_type == 1:
            if 'sdf_tensor' not in perception_data:
                 raise ValueError("SDF tensor missing from perception data for supervised mode.")
            return perception_data['sdf_tensor'], self.feature_extractor

        # For unsupervised mode, no specific model inputs are needed
        assert self.controller_type == 0, "at this point, we should only be in unsupervised mode"
        return None, None
    
    def _sample_trajectories_robot_frame(self, current_state: np.ndarray, 
                                        sdf_tensor: torch.Tensor, feature_extractor,
                                        velocity_override: float = None) -> Tuple:
        """Sample trajectories in robot frame."""
        # Validate models are loaded
        if self.model is None:
            mode = "unsupervised" if self.controller_type == 0 else "supervised map-conditioned"
            raise ValueError(f"Model not loaded for {mode} mode")
        
        if self.controller_type == 1 and feature_extractor is None:
            raise ValueError("Feature extractor not loaded for supervised mode")
        
        # Generate action space
        steering_actions = self.actions[:, 0]  # Extract steering angles from pre-generated actions
        
        # Sample trajectories (always start from robot frame origin)
        robot_frame_initial_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Robot frame origin
        robot_frame_trajectories, trajectory_actions = self.sample_trajectories_cuniform(
            initial_state=robot_frame_initial_state,
            actions=steering_actions,
            dynamics_cuda=dynamics_scalar_torch,
            num_trajectories=self.num_trajectories,
            trajectory_length=self.trajectory_length,
            model=self.model,
            feature_extractor=feature_extractor,
            sdf_tensor=sdf_tensor,
            wheelbase=self.wheelbase,
            uniform_sampling=False,
            velocity_override=velocity_override, 
        )
        
        return robot_frame_trajectories, trajectory_actions
    
    def _evaluate_robot_frame_trajectories(self, robot_frame_trajectories: torch.Tensor, 
                                          trajectory_actions: torch.Tensor, 
                                          goal_state: np.ndarray, current_world_pose: np.ndarray) -> Tuple[np.ndarray, int]:
        """Evaluate robot frame trajectories against robot frame costmap and select best action.
        Cost calculation exits early when goal is reached - we don't care about what happens after goal."""
        robot_frame_goal = self._transform_goal_to_robot_frame(goal_state, current_world_pose)
        goal_tensor = torch.from_numpy(robot_frame_goal).float().to(self.device)

        total_costs = self._calculate_trajectory_costs(robot_frame_trajectories, goal_tensor)
        # Find best trajectory (minimum cost)
        best_idx = torch.argmin(total_costs).item()

        # Get the first action from the best trajectory
        steering_actions = self.actions[:, 0]
        best_action_idx = trajectory_actions[0, best_idx].item()
        best_action = steering_actions[best_action_idx]

        return np.array([self.vrange[0], best_action], dtype=np.float32), best_idx
    
    def _prepare_visualization_info(self, trajectories: torch.Tensor, trajectory_actions: torch.Tensor, best_trajectory_idx: int) -> Dict[str, Any]:
        """Prepare visualization data for the info dictionary."""
        assert trajectories is not None, "trajectories should not be None"
        
        # Convert trajectories to MPPI format: (num_trajectories, timesteps, 3)
        all_trajectories = trajectories.cpu().numpy()  # Shape: (timesteps, num_trajectories, 3)
        all_trajectories_mppi_format = all_trajectories.transpose(1, 0, 2)
        
        # Create visualization array with best trajectory at index 0
        vis_rollouts = np.zeros((self.num_vis_trajectories, self.trajectory_length + 1, 3), dtype=np.float32)
        
        # Put best trajectory at index 0 (green line in visualization)
        vis_rollouts[0] = all_trajectories_mppi_format[best_trajectory_idx]
        
        # Add remaining trajectories (black lines in visualization)
        remaining_indices = [i for i in range(min(self.num_vis_trajectories-1, all_trajectories_mppi_format.shape[0])) 
                            if i != best_trajectory_idx][:self.num_vis_trajectories-1]
        for i, idx in enumerate(remaining_indices):
            vis_rollouts[i+1] = all_trajectories_mppi_format[idx]
        
        # Apply angle wrapping to prevent visualization artifacts
        vis_rollouts[:, :, 2] = np.arctan2(np.sin(vis_rollouts[:, :, 2]), np.cos(vis_rollouts[:, :, 2]))
        
        self.sampled_trajectories_history.append(vis_rollouts)
        state_rollouts = vis_rollouts
        
        return {
            'local_costmap': self.local_costmap_map if hasattr(self, 'local_costmap_map') else None,
            'visualize_costmap': True,
            'visualize_all_trajectories': True,
            'state_rollouts': state_rollouts
        }
    
    def get_sampled_trajectories_history(self):
        """Return complete history of sampled trajectories for post-experiment analysis."""
        return self.sampled_trajectories_history
    
    def _debug_robot_frame_trajectories(self, robot_frame_trajectories: torch.Tensor,
                                        best_trajectory_idx: int, goal_state: np.ndarray,
                                        current_world_pose: np.ndarray):
        """Quick visualization of robot frame trajectories with costmap for debugging."""
        global _debug_timestamp # Declare that we are using the global counter

        # Create a single, timestamped debug directory for the entire experiment run
        if not hasattr(self, 'debug_dir') or self.debug_dir is None:
            self.debug_dir = f"debug_output/{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.debug_dir, exist_ok=True)
        
        # Convert to numpy for visualization
        trajectories_np = robot_frame_trajectories.cpu().numpy()
        
        # Transform goal to robot frame
        robot_frame_goal = self._transform_goal_to_robot_frame(goal_state, current_world_pose)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Show costmap if available
        if hasattr(self, 'local_costmap_map') and self.local_costmap_map is not None:
            # Calculate the extent of the map in meters for correct plotting
            map_extent = self.local_costmap_size * self.local_costmap_resolution / 2.0
            ax.imshow(self.local_costmap_map, cmap='gray_r', origin='upper', 
                    extent=[-map_extent, map_extent, -map_extent, map_extent])
        
        # Plot all trajectories (black)
        for traj_idx in range(trajectories_np.shape[1]):
            x_traj = trajectories_np[:, traj_idx, 0]
            y_traj = trajectories_np[:, traj_idx, 1]
            ax.plot(x_traj, y_traj, 'k-', alpha=0.2, linewidth=0.5)
        
        # Plot best trajectory (red)
        if best_trajectory_idx < trajectories_np.shape[1]:
            x_best = trajectories_np[:, best_trajectory_idx, 0]
            y_best = trajectories_np[:, best_trajectory_idx, 1]
            ax.plot(x_best, y_best, 'r-', linewidth=2, label='Best trajectory')
        
        # Plot robot position (blue circle at origin)
        ax.plot(0, 0, 'bo', markersize=8, label='Robot')
        
        # Plot goal (green star)
        ax.plot(robot_frame_goal[0], robot_frame_goal[1], 'g*', markersize=12, label='Goal')
        
        # Set limits and labels
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('Robot Frame X (m)')
        ax.set_ylabel('Robot Frame Y (m)')
        ax.set_title(f'Robot Frame Trajectories (Step: {_debug_timestamp})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Save the plot to the timestamped directory
        filename = f"{self.debug_dir}/robot_frame_step_{_debug_timestamp:04d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        _debug_timestamp += 1 # Increment the global counter for the next step
        
        print(f"      Robot frame debug saved: {filename}")

    def batch_states_to_grid(self, states, sdf_shape, resolution):
        """
        Converts robot states to normalized grid coordinates for F.grid_sample.

        Args:
            states (np.ndarray or torch.Tensor): Array of shape (N, D) where D>=2.
                Only the first two dimensions (x,y) are used.
            sdf_shape (tuple): The shape (H, W) of the SDF grid.
            resolution (float): The size of each grid cell (meters per cell).

        Returns:
            torch.Tensor: A grid of shape (1, N, 1, 2) with values in [-1, 1],
                        ready to be used as the grid input for F.grid_sample.
        """
        if isinstance(states, np.ndarray): # If input is numpy, convert to torch tensor.
            states = torch.from_numpy(states).float()
        xy = states[:, :2]  # shape: (N, 2)
        H, W = sdf_shape
        center_x = (W - 1) / 2.0
        center_y = (H - 1) / 2.0

        # Convert world coordinates (in meters) to pixel indices.
        # Column index: center_x + (x / resolution)
        # Row index: center_y - (y / resolution)  (assuming y increases upward in world frame)
        cols = center_x + xy[:, 0] / resolution
        rows = center_y - xy[:, 1] / resolution

        # Normalize to the interval [0, 1] relative to grid dimensions.
        cols_norm = cols / (W - 1)
        rows_norm = rows / (H - 1)

        # Convert to grid_sample coordinates in the interval [-1, 1].
        x_grid = 2 * cols_norm - 1
        y_grid = 2 * rows_norm - 1

        grid = torch.stack([x_grid, y_grid], axis=1) # Stack into (N, 2) 
        grid = grid.unsqueeze(0).unsqueeze(2) # reshape to (1, N, 1, 2) required by grid_sample.
        return grid

    def bilinear_sample_sdf_features(self, sdf_features, states, resolution):
        """
        sdf_features: (B, C, H, W)
        states (np.ndarray or torch.Tensor): Robot states with shape (B, N, D)
                (only the first two coordinates are used).
        Returns: (B, N, C)
        """
        B, C, H, W = sdf_features.shape
        assert B == 1, "lets use multiple states to query the same feature map, batch operation not supported"
        grid = self.batch_states_to_grid(states, (H, W), resolution)   # → [1, N, 1, 2]

        # grid_sample expects (1, H_out, W_out, 2), so this is (1, N, 1, 2)
        sampled = F.grid_sample(sdf_features, grid, align_corners=False, mode='bilinear')  # returns (1, C, N, 1)
        sampled = sampled.squeeze(-1).squeeze(0)  # → [C, N]
        sampled = sampled.transpose(0, 1)         # → [N, C]
        return sampled

    def sample_trajectories_cuniform(
            self, initial_state, actions, dynamics_cuda,
            num_trajectories, trajectory_length,
            model, feature_extractor, sdf_tensor, wheelbase,
            uniform_sampling=False, velocity_override=None,
        ):
        """Sample trajectories using C-Uniform approach.
        
        This function only handles trajectory sampling and returns the raw trajectories.
        Evaluation should be done separately to keep the function modular.
        """
        start_time = time.time()
        t_step = self.dt
        if velocity_override is not None:
            v = velocity_override
        else: # Default behavior for standalone C-Uniform
            v = self.vrange[0]

        # Pre-convert actions to GPU tensor for efficient indexing
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device="cuda")

        # Shape after repeat: (num_trajectories, state_dim)
        batch_current_states = (
            torch.tensor(initial_state, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(num_trajectories, 1)
            .cuda()
        )
        state_dim = batch_current_states.shape[1]

        # Preallocate a tensor to store states for each time step.
        # Shape: (trajectory_length+1, num_trajectories, state_dim)
        trajectory_states = torch.empty((trajectory_length + 1, num_trajectories, state_dim),
                                        dtype=torch.float32, device="cuda")
        trajectory_states[0] = batch_current_states

        # Preallocate a tensor to record the chosen action indices at each step.
        # Shape: (trajectory_length, num_trajectories)
        trajectory_actions = torch.empty((trajectory_length, num_trajectories),
                                        dtype=torch.int64, device="cuda")
        model_name = model.__class__.__name__
        if model_name == "MapAct_PixelInterpolated" and feature_extractor is not None:
            # For pixel-interpolated model, use the dense feature map and bilinear interpolation
            dense_features = feature_extractor(sdf_tensor)  # [1, feature_dim, H, W]
        for step in range(trajectory_length):
            theta = batch_current_states[:, 2]
            theta_sin = torch.sin(theta)
            theta_cos = torch.cos(theta)

            network_current_states = torch.hstack((
                batch_current_states[:, 0:2],
                theta_sin.unsqueeze(1),
                theta_cos.unsqueeze(1)
            ))

            # Predict action probabilities in batch using the model
            if self.controller_type == 1: # model_name should be "MapAct_PixelInterpolated"
                # dense_features = feature_extractor(costmap_tensor)  # [1, feature_dim, H, W]
                interpolated_features = self.bilinear_sample_sdf_features(
                    sdf_features=dense_features, 
                    states=network_current_states, 
                    resolution=self.experiment_config['local_costmap_resolution']
                )
                current_probabilities = model(network_current_states, interpolated_features).detach()
            elif self.controller_type == 0: # model_name should be "MapAct"
                if uniform_sampling:
                    # Use uniform probabilities instead of model predictions
                    batch_size = network_current_states.shape[0]
                    num_actions = len(actions)
                    current_probabilities = torch.ones((batch_size, num_actions), 
                                                    device=network_current_states.device, 
                                                    dtype=torch.float32) / num_actions
                else:
                    current_probabilities = model(network_current_states).detach()
            else:
                raise ValueError(f"Unsupported controller type: {self.controller_type}")
            
            # if self.controller_type == 0 and step == 0: # For c-uniform sampler, set initial probabilities to uniform
            #     current_probabilities = torch.full_like(current_probabilities, 1.0 / current_probabilities.size(1))
            # Use torch.multinomial to efficiently sample one action index per trajectory.
            chosen_action_indices = torch.multinomial(current_probabilities, num_samples=1).squeeze(1)

            # Save the chosen action indices into the preallocated tensor.
            trajectory_actions[step] = chosen_action_indices.to("cuda", torch.int64)

            # Convert the chosen actions using efficient tensor indexing (OPTIMIZED & Verified)
            chosen_actions_tensor = actions_tensor[chosen_action_indices]

            # add action jitter to compensate for discrete action space when compared against mppi's continuous action space
            ACTION_PERTURBATION = True
            if ACTION_PERTURBATION:
                steer_min = float(np.min(actions))
                steer_max = float(np.max(actions))
                # small zero‑mean Gaussian noise
                noise = torch.randn_like(chosen_actions_tensor) * 0.00436 # about 0.25 degrees, just tiny bit of jitter
                chosen_actions_tensor = chosen_actions_tensor + noise
                chosen_actions_tensor = torch.clamp(chosen_actions_tensor, min=steer_min, max=steer_max)

            # Propagate states in parallel.
            batch_current_states = dynamics_cuda(batch_current_states, chosen_actions_tensor, t_step, v, wheelbase)

            STATE_PERTURBATION = False
            if STATE_PERTURBATION:
                sigma_xy    = 0.05
                sigma_theta = 0.05
                noise_xy    = torch.randn_like(batch_current_states[:, 0:2]) * sigma_xy # xy noise
                noise_th    = torch.randn_like(batch_current_states[:, 2:3]) * sigma_theta # θ noise
                batch_current_states = torch.cat([
                    batch_current_states[:, 0:2] + noise_xy,
                    batch_current_states[:, 2:3] + noise_th,
                ], dim=1)

            # Record the newly updated states.
            trajectory_states[step + 1] = batch_current_states
        # print(f"    Trajectory sampling process took {time.time() - start_time} seconds")
        return trajectory_states, trajectory_actions