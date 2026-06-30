"""Experiment runner for BARN navigation experiments."""
import numpy as np
import time
from typing import Dict, Any
from map_conditioning.navigation_experiments.core.environments_manager import EnvironmentsManager
from controllers.base_controller import BaseController
from utils.dynamics import bicycle_model_step
from utils.geometry_utils import calculate_traveled_distance, calculate_control_smoothness, check_collision_along_path

class ExperimentRunner:
    """Runs navigation experiments with given controller and environment."""
    
    def __init__(self, experiment_config: dict):
        """Initialize experiment runner.
        
        Args:
            experiment_config: Configuration dictionary with all experiment parameters
        """
        self.max_steps = experiment_config['max_steps']
        self.goal_tolerance = experiment_config['goal_tolerance']
        self.dt = experiment_config['dt']
        self.point_robot_mode = experiment_config['point_robot_mode']
        
    def run_trial(self, controller: BaseController, environment: EnvironmentsManager,
                  start_state: np.ndarray, goal_pos: np.ndarray) -> Dict[str, Any]:
        """Run a single navigation trial.
        
        Args:
            controller: Controller to use
            environment: EnvironmentsManager to navigate in
            start_state: [x, y, theta, v] start state
            goal_pos: [x, y] goal position
            
        Returns:
            Dictionary containing trial results
        """
        # Initialize robot state
        current_state = start_state.copy() # [x, y, theta, v]
        prev_state = current_state.copy()  # Track previous state for path collision checking
        
        # Storage for trajectory and controls
        trajectory = [current_state.copy()]
        controls = []

        # Lists to store historical data for visualization
        all_sampled_trajectories = []
        costmaps_history = []
        
        # Progress tracking
        step_count = 0
        last_progress_time = time.time()
        
        print(f"    Start: {start_state} → Goal: {goal_pos}")
        if self.point_robot_mode:
             print("    [INFO] Running trial in POINT ROBOT mode.")
        while step_count < self.max_steps:
            # Get global occupancy grid from environment
            global_occupancy_grid = environment.current_map
            
            # Get control action from controller
            control, info = controller.get_control_action(
                current_state, goal_pos, global_occupancy_grid, self.dt
            )
            
            # Collect data for visualization from the current step's info dictionary
            if 'state_rollouts' in info:
                all_sampled_trajectories.append(info['state_rollouts'])
            if 'local_costmap' in info:
                costmaps_history.append(info['local_costmap'])
            
            # Apply control and update state
            prev_state = current_state.copy()  # Store previous state before update
            current_state = bicycle_model_step(current_state, control, self.dt, controller.wheelbase)
            
            # Store trajectory and control
            trajectory.append(current_state.copy())
            controls.append(control.copy())
            
            step_count += 1
            
            # Progress indicator
            if step_count % 50 == 0:
                current_time = time.time()
                if current_time - last_progress_time > 1.0:  # Print every second
                    dist_to_goal = np.linalg.norm(current_state[:2] - goal_pos)
                    print(f"    Step {step_count}: Distance to goal = {dist_to_goal:.2f}m")
                    last_progress_time = current_time
            
            # Check termination conditions
            dist_to_goal = np.linalg.norm(current_state[:2] - goal_pos)
            
            # Create a dictionary to pass all collected data to the result function
            trial_data = {
                "trajectory": trajectory,
                "controls": controls,
                "sampled_trajectories": all_sampled_trajectories,
                "costmaps_history": costmaps_history
            }

            if dist_to_goal <= self.goal_tolerance:
                print(f"     GOAL REACHED at step {step_count}")
                return self._create_result(True, trial_data, step_count, dist_to_goal, "goal_reached")
            
            # Check collision along the path (prevents tunneling through walls)
            collision_detected = False
            if step_count == 1:
                # For first step, only check current position
                collision_detected = environment.check_collision(current_state)
            else:
                # For subsequent steps, check along the path between previous and current positions
                collision_detected = check_collision_along_path(
                    environment.current_map, 
                    prev_state, 
                    current_state,
                    environment.vehicle_length,
                    environment.vehicle_width,
                    environment.resolution,
                    environment.origin,
                    use_point_robot=self.point_robot_mode,
                )
            
            if collision_detected:
                print(f"     COLLISION detected at step {step_count}")
                return self._create_result(False, trial_data, step_count, dist_to_goal, "collision")
        
        # Max steps reached
        print(f"     MAX STEPS reached ({self.max_steps})")
        # Create a dictionary to pass all collected data to the result function
        trial_data = {
            "trajectory": trajectory, # trajectory is a list of robot's past states
            "controls": controls,
            "sampled_trajectories": all_sampled_trajectories,
            "costmaps_history": costmaps_history
        }
        return self._create_result(False, trial_data, step_count, dist_to_goal, "max_steps")
    
    def _create_result(self, success: bool, trial_data: Dict[str, Any], steps: int, 
                      final_distance: float, termination_reason: str) -> Dict[str, Any]:
        """Create result dictionary from trial data."""
        trajectory_array = np.array(trial_data['trajectory'])
        controls_array = np.array(trial_data['controls'])
        
        # Calculate additional metrics
        traveled_distance = calculate_traveled_distance(trajectory_array)
        control_smoothness = calculate_control_smoothness(controls_array)
        
        result = {
            'success': success,
            'trajectory': trajectory_array,
            'controls': controls_array,
            'steps': steps,
            'final_distance': final_distance,
            'termination_reason': termination_reason,
            'traveled_distance': traveled_distance,
            'control_smoothness': control_smoothness,
            'sampled_trajectories': trial_data['sampled_trajectories'],
            'costmaps_history': trial_data['costmaps_history'] # <-- Pass history to final result
        }
        return result