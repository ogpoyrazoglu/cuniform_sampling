"""
MPPI Controller (Pure PyTorch Implementation).
Provides both standard MPPI and Log-MPPI variants.
Relies on TorchPlannerBase for standardized dynamics and cost evaluation.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any
from .torch_planner_base import TorchPlannerBase
import torch.distributions as dist
from utils.transformation_util import transform_trajectories_to_world_frame

def Normal2LogN(m, v):
    """ m: mean, v: variance (of the underlying Normal distribution)
    Return: mu: mean, sigma: standard deviation (of the resulting Log-Normal distribution)
    """
    # Add small epsilon for numerical stability if variance is zero
    v = max(v, 1e-9)
    mu = np.exp(m + 0.5 * v)
    var = np.exp(2 * m + v) * (np.exp(v) - 1)
    sigma = np.sqrt(var)
    return mu, sigma

class MPPIPyTorchController(TorchPlannerBase):
    """
    Model Predictive Path Integral (MPPI) controller implemented fully in PyTorch.
    """
    def __init__(self, controller_config: dict, experiment_config: dict = None,
                 type_override: int = None, seed: int = None):

        if seed is None:
            seed = experiment_config.get('seed', 2025) if experiment_config else 2025
        super().__init__(controller_config, experiment_config, seed)

        # MPPI specific parameters
        self.K = self.num_rollouts # Number of samples (N)
        self.T = self.T_horizon    # Trajectory length (T)
        self.nu = 2                # Control dimension [v, w]
        
        self.lambda_weight = self.config['lambda_weight']
        self.mppi_iterations = self.config['num_opt']

        # Determine MPPI type (0: Standard Gaussian, 1: Log-Normal/NLN)
        if type_override is not None:
            self.mppi_type = type_override
        else:
            self.mppi_type = self.config.get('mppi_type', 0)

        self._init_noise_params()

        # Control sequence (Nominal trajectory) initialization (T, nu)
        self.U_nominal = torch.zeros((self.T, self.nu), dtype=torch.float32, device=self.device)
        # Initialize with minimum velocity
        self.U_nominal[:, 0] = float(self.vrange[0])

        # Storage for visualization
        self.sampled_trajectories_history = []
        self.current_state = np.zeros(3, dtype=np.float32)  # [x, y, theta]

        print(f"MPPIPyTorch Controller initialized. MPPI-Type: {self.mppi_type}. Samples: {self.K}")

    def _init_noise_params(self):
        """Initialize noise parameters, handling fixed/variable velocity modes."""
        # Noise standard deviations [velocity_std, steering_std]
        u_std_config = np.array(self.config['u_std'], dtype=np.float32)
        u_std = u_std_config.copy()

        if not self.variable_velocity_mode:
            # Fixed velocity mode: velocity noise is zero
            u_std[0] = 0.0
        elif u_std[0] <= 0:
            print(f"WARNING: Variable velocity mode is enabled, but velocity noise sigma (u_std[0]) is {u_std[0]}.")


        # Shape (2,) for [vel, steer]
        self.noise_sigma = torch.tensor(u_std, device=self.device, dtype=torch.float32)

        if self.mppi_type == 1:
            # Calculate based on the mean of the original u_std from the config
            mean_u_std = np.mean(u_std_config)

            # follow the original log-normal MPPI convention where
            # mean_u_std is treated as the variance input 'v' for Normal2LogN
            self.mu_LogN, self.std_LogN = Normal2LogN(0, mean_u_std)

            # mu_LogN (M) and std_LogN (S) are passed directly to the sampler,
            # which interprets them as the underlying parameters (mu, sigma)
            underlying_m = torch.tensor(self.mu_LogN, device=self.device, dtype=torch.float32)
            underlying_s = torch.tensor(self.std_LogN, device=self.device, dtype=torch.float32)

            # Initialize the PyTorch distribution object (loc=mu, scale=sigma)
            self.log_normal_dist = dist.LogNormal(loc=underlying_m, scale=underlying_s)

    def reset(self):
        """Reset controller state for new environment."""
        super().reset()
        # Reset nominal control sequence
        self.U_nominal = torch.zeros((self.T, self.nu), dtype=torch.float32, device=self.device)
        self.U_nominal[:, 0] = float(self.vrange[0])

        self.sampled_trajectories_history = []
        self.current_state = np.zeros(3, dtype=np.float32)

    def get_control_action(self, current_state: np.ndarray, goal_state: np.ndarray,
                          global_occupancy_grid: np.ndarray, dt: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal control action using GPU-accelerated MPPI algorithm."""
        self.current_state = current_state[:3].astype(np.float32)

        # Perception
        self._prepare_perception_inputs(global_occupancy_grid, current_state)

        # Warm-start: Shift the nominal trajectory forward
        self._shift_nominal_trajectory()

        # Run MPPI Optimization
        U_optimal, mppi_trajectories = self._run_mppi_optimization(
            self.U_nominal, current_state, goal_state
        )

        # Update the internal nominal trajectory with the optimized one
        self.U_nominal = U_optimal

        # Get current control action (first step)
        control_action_np = U_optimal[0].cpu().numpy()

        # Visualization
        info = self._prepare_visualization_info(
             mppi_trajectories, U_optimal, current_state[:3]
        )
        return control_action_np, info

    def _shift_nominal_trajectory(self):
        """Shift the nominal trajectory forward one step (Warm-starting)."""
        # Roll the sequence forward. The element at index T-2 moves to T-1.
        self.U_nominal = torch.roll(self.U_nominal, shifts=-1, dims=0)

    # =================================================================================
    # Core MPPI Logic
    # =================================================================================
    def _run_mppi_optimization(self, U_nominal, current_state, goal_state):
        """
        Runs the MPPI optimization loop in PyTorch.
        Implements the update rule matching reference_mppi.py (Weighted Effective Noise).
        """
        # Setup for robot frame calculations
        robot_frame_initial_state = torch.zeros(3, device=self.device, dtype=torch.float32)
        robot_frame_goal = self._transform_goal_to_robot_frame(goal_state, current_state[:3])
        goal_tensor = torch.from_numpy(robot_frame_goal).float().to(self.device)

        # Control limits for clamping (Tensors for broadcasting)
        # Shape (1, 1, 2)
        min_ctrl = torch.tensor([float(self.vrange[0]), float(self.wrange[0])], device=self.device).view(1, 1, 2)
        max_ctrl = torch.tensor([float(self.vrange[1]), float(self.wrange[1])], device=self.device).view(1, 1, 2)

        # MPPI Iteration Loop
        for iteration in range(self.mppi_iterations):
            # 1. Generate Noise (T, K, 2)
            noise = self._generate_noise_torch()

            # 2. Perturb nominal controls (T, K, 2)
            perturbed_controls = U_nominal.unsqueeze(1) + noise

            # Clamp controls to physical limits (Vectorized)
            perturbed_controls_clamped = torch.max(torch.min(perturbed_controls, max_ctrl), min_ctrl)

            # 3. Rollout trajectories (using clamped controls) (T+1, K, 3)
            mppi_trajectories = self._rollout_full_controls_torch(perturbed_controls_clamped, robot_frame_initial_state)

            # 4. Calculate costs (K,)
            costs = self._calculate_trajectory_costs(mppi_trajectories, goal_tensor)

            # 5. Weight trajectories (Importance Sampling)
            weights = torch.softmax(-(1.0 / self.lambda_weight) * costs, dim=0)

            # 6. Update Nominal Control (Standard MPPI Update)
            # Use the raw sampled noise for the update, ignoring clamping effects.
            noise_to_use = noise

            # Reshape weights: (K,) -> (1, K, 1)
            weights_reshaped = weights.unsqueeze(0).unsqueeze(2)

            # Calculate weighted perturbations
            # Sum((1, K, 1) * (T, K, 2)) along dim 1 -> (T, 2)
            perturbations = torch.sum(weights_reshaped * noise_to_use, dim=1)

            # Update nominal control
            U_nominal = U_nominal + perturbations

            # Final clamping (safety measure)
            U_nominal = torch.max(torch.min(U_nominal, max_ctrl.view(1, 2)), min_ctrl.view(1, 2))

        # Return the final optimized trajectory and the trajectories from the LAST iteration
        return U_nominal, mppi_trajectories

    def _generate_noise_torch(self):
        """
        Generates noise samples on the GPU. Supports Standard and Log-MPPI.
        Returns noise shape: (T, K, 2)
        """
        T, K = self.T, self.K

        # Sigma shape: (2,) -> (1, 1, 2) for broadcasting
        sigma = self.noise_sigma.view(1, 1, 2)

        if self.mppi_type == 0:
            # Standard MPPI: Gaussian Noise
            std_normal = torch.randn(T, K, 2, device=self.device)
            noise = std_normal * sigma

        elif self.mppi_type == 1:
            # Log-MPPI (NLN - Normal * Log-Normal)
            # 1. Generate Log-Normal Noise by sampling from the initialized distribution
            log_normal_noise = self.log_normal_dist.sample((T, K, 2))

            # Generate Standard Normal Noise N(0, 1)
            std_normal_noise = torch.randn(T, K, 2, device=self.device)

            # Multiply: NLN = LogNormal * Normal
            noise = log_normal_noise * std_normal_noise

            # Scaling by sigma (Standard deviation of the control inputs)
            noise = noise * sigma
        else:
            raise ValueError(f"Unknown MPPI type: {self.mppi_type}")

        return noise

    # =================================================================================
    # Visualization
    # =================================================================================

    def _prepare_visualization_info(self, mppi_trajectories_robot, U_nominal_final, current_pose):
        """
        Prepare visualization data.
        Transforms trajectories to the world frame and packages them.
        """
        # 1. MPPI Samples (Last iteration)
        # Transform (T+1, K, 3) -> (K, T+1, 3)
        mppi_traj_world = transform_trajectories_to_world_frame(mppi_trajectories_robot, current_pose)
        mppi_traj_world = mppi_traj_world.permute(1, 0, 2)

        # Limit visualization count
        num_vis = min(self.num_vis_rollouts, self.K)

        # 2. MPPI Final (Nominal/Optimal trajectory)
        # Rollout the final nominal control sequence (Ensure it's visualized correctly)
        robot_frame_initial_state = torch.zeros(3, device=self.device, dtype=torch.float32)
        U_nominal_reshaped = U_nominal_final.unsqueeze(1) # (T, 1, 2)
        nominal_traj_robot = self._rollout_full_controls_torch(U_nominal_reshaped, robot_frame_initial_state)

        # Transform (T+1, 1, 3) -> (T+1, 3)
        nominal_traj_world = transform_trajectories_to_world_frame(nominal_traj_robot, current_pose)
        nominal_traj_world = nominal_traj_world.squeeze(1)

        # 3. Assemble visualization array (Best trajectory at index 0)
        vis_rollouts = torch.zeros((num_vis, self.T + 1, 3), dtype=torch.float32, device=self.device)

        # Index 0: The optimal trajectory (Green line)
        vis_rollouts[0] = nominal_traj_world

        # Indices 1 onwards: Sampled trajectories (Black lines)
        if num_vis > 1:
            # We select the first N-1 samples from the last iteration.
            num_samples_to_copy = min(num_vis - 1, mppi_traj_world.shape[0])
            if num_samples_to_copy > 0:
                vis_rollouts[1:1+num_samples_to_copy] = mppi_traj_world[:num_samples_to_copy]

        # Convert to numpy
        vis_rollouts_np = vis_rollouts.cpu().numpy()

        # Apply angle wrapping (dynamics should handle this, but required for visualization)
        vis_rollouts_np[:, :, 2] = np.arctan2(np.sin(vis_rollouts_np[:, :, 2]), np.cos(vis_rollouts_np[:, :, 2]))

        # Store history
        self.sampled_trajectories_history.append(vis_rollouts_np)

        return {
            'local_costmap': self.local_costmap_map if hasattr(self, 'local_costmap_map') else None,
            'visualize_costmap': True,
            'visualize_all_trajectories': True,
            'state_rollouts': vis_rollouts_np
        }

    def get_sampled_trajectories_history(self):
        """Return complete history of sampled trajectories for post-experiment analysis."""
        return self.sampled_trajectories_history