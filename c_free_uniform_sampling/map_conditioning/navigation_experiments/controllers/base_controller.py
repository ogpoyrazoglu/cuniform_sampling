"""Abstract base controller class."""
from abc import ABC, abstractmethod
import numpy as np
import yaml
from typing import Tuple, Dict, Any


class BaseController(ABC):
    """Abstract base class for all navigation controllers."""
    
    def __init__(self, controller_config: dict, experiment_config: dict, seed: int):
        """Initialize controller with provided configuration and shared parameters."""
        # Store controller-specific config
        self.config = controller_config
        
        # Store experiment config
        self.experiment_config = experiment_config
        self.seed = seed
        
        # Initialize shared parameters from experiment config
        self._init_shared_parameters()
    
    def _init_shared_parameters(self):
        """Initialize parameters shared across all controllers from experiment config."""
        # Planning parameters
        self.T = self.experiment_config['horizon_T']
        self.dt = self.experiment_config['dt']
        self.num_steps = int(self.T / self.dt)
        
        # Vehicle parameters
        self.vehicle_length = self.experiment_config['vehicle_length']
        self.vehicle_width = self.experiment_config['vehicle_width']
        self.wheelbase = self.experiment_config['vehicle_wheelbase']
        
        # Control constraints
        self.vrange = np.array(self.experiment_config['vrange'], dtype=np.float32)
        self.wrange = np.array(self.experiment_config['wrange'], dtype=np.float32)
        
        # Cost weights
        self.obs_penalty = self.experiment_config['obs_penalty']
        self.dist_weight = self.experiment_config['dist_weight']
        self.terminal_weight = self.experiment_config['terminal_weight']
        self.goal_tolerance = self.experiment_config['goal_tolerance']

        self.collision_occupancy_ratio = self.experiment_config['collision_occupancy_ratio']
        self.robot_footprint_size = self.experiment_config['robot_footprint_size']
        self.robot_footprint_area = self.robot_footprint_size * self.robot_footprint_size

        # LIDAR parameters
        self.lidar_scan_range = self.experiment_config['lidar_scan_range']
        self.lidar_num_beams = self.experiment_config['lidar_num_beams']

        # Perception Mode
        self.perception_mode = self.experiment_config['perception_mode']

        # Local costmap parameters
        self.local_costmap_size = self.experiment_config['local_costmap_size']
        self.local_costmap_resolution = self.experiment_config['local_costmap_resolution']
        assert self.local_costmap_size % 2 == 1, "local_costmap_size must be an odd number to have a true center."
        
        # Obstacle inflation parameters - Select profile based on mode
        inflation_profiles = self.experiment_config.get('inflation_profiles', {})

        # Use the profile for the active perception mode
        active_profile = inflation_profiles[self.perception_mode]
        self.inflation_radius = active_profile['inflation_radius']
        self.max_inflation_value = active_profile['max_inflation_value']
        self.sdf_inflation_cells = active_profile['sdf_inflation_cells']
        self.extract_boundaries = active_profile['extract_boundaries']

        # Update the main config dictionary so the perception module sees the active values
        self.experiment_config['inflation_radius'] = self.inflation_radius
        self.experiment_config['max_inflation_value'] = self.max_inflation_value
        self.experiment_config['sdf_inflation_cells'] = self.sdf_inflation_cells
        self.experiment_config['extract_boundaries'] = self.extract_boundaries
        
        # Sampling parameters (shared across controllers)
        self.num_rollouts = self.experiment_config['num_rollouts']
        self.num_vis_rollouts = self.experiment_config['num_vis_rollouts']
        self.initialization_budget_ratio = self.experiment_config.get('initialization_budget_ratio', 0.5)
    
    @abstractmethod
    def get_control_action(self, current_state: np.ndarray, goal_state: np.ndarray, 
                          global_occupancy_grid: np.ndarray, dt: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute control action.
        
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
        pass
    
    @abstractmethod
    def reset(self):
        """Reset controller state for new environment."""
        pass 