"""Environment management for BARN navigation experiments."""
import numpy as np
import pickle
import os
from typing import Tuple, Dict, Any, Iterator
from utils.geometry_utils import (
    check_collision_polygon, 
    get_vehicle_boundary_points, 
    rasterize_polygon_interior, 
    check_collision_point
)
from data.visualize_converted_maps import visualize_map

class EnvironmentsManager:
    """Manages experiment suites and acts as an iterator over test configurations."""
    
    def __init__(self, experiment_config: dict):
        """Initialize experiment manager.
        
        Args:
            experiment_config: Configuration dictionary with all experiment parameters
        """
        self.experiment_config = experiment_config.copy()
        self.suite_config = experiment_config['experiment_suite']
        
        # Vehicle and map parameters
        self.resolution = experiment_config['global_map_resolution']
        self.origin = experiment_config['global_origin']
        self.vehicle_length = experiment_config['vehicle_length']
        self.vehicle_width = experiment_config['vehicle_width']
        self.vehicle_wheelbase = experiment_config['vehicle_wheelbase']
        
        # Current state
        self.current_map = None 
        self.current_config = None
        self.test_plan = []
        self.current_index = 0
        
        # For on-the-fly dumbbell generation
        self.results_base_dir = None  # Will be set by run_experiments.py
        self.dumbbell_maps_dir = None
        self.dumbbell_viz_dir = None
        
        # Build the test plan based on experiment type
        self._build_test_plan()
        
        # Set up random number generator with seed
        self.rng = np.random.RandomState(experiment_config['seed'])
        
    def set_results_directory(self, results_dir: str) -> None:
        """Set the results directory and create dumbbell folders if needed.
        
        Args:
            results_dir: Base results directory path
        """
        self.results_base_dir = results_dir
        
        # Only create DUMBELL_dataset folder if using dumbbell-related experiments
        experiment_type = self.suite_config['type']
        if experiment_type in ["DumbbellSweep", "BudgetSweep"]:
            dumbbell_dataset_dir = os.path.join(results_dir, "DUMBELL_dataset")
            self.dumbbell_maps_dir = os.path.join(dumbbell_dataset_dir, "generated_dumbbell_maps")
            self.dumbbell_viz_dir = os.path.join(dumbbell_dataset_dir, "generated_dumbbell_visualizations")
            os.makedirs(self.dumbbell_maps_dir, exist_ok=True)
            os.makedirs(self.dumbbell_viz_dir, exist_ok=True)
        else:
            # Set to None for non-dumbbell experiments
            self.dumbbell_maps_dir = None
            self.dumbbell_viz_dir = None
        
    def _build_test_plan(self) -> None:
        """Build complete test plan based on experiment suite configuration."""
        experiment_type = self.suite_config['type']
        trials_per_env = self.suite_config['trials_per_environment']
        
        if experiment_type == "BARN":
            self._build_barn_test_plan(trials_per_env)
        elif experiment_type == "DumbbellSweep":
            self._build_dumbbell_test_plan(trials_per_env)
        elif experiment_type == "BudgetSweep":
            self._build_budget_sweep_test_plan(trials_per_env)
        elif experiment_type == "Polygon":
            self._build_polygon_test_plan(trials_per_env)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
            
        print(f"Built test plan with {len(self.test_plan)} total test configurations")
    
    def _update_map_parameters(self) -> None:
        """Dynamically updates map resolution and origin based on the experiment suite type."""
        experiment_type = self.suite_config['type']
        if experiment_type == "Polygon":
            polygon_config = self.suite_config['polygon_config']
            self.resolution = polygon_config['map_resolution']
            self.origin = polygon_config['map_origin']
            
            # Update the main config dict so other modules (perception, visualizer) use these updated values during the experiment run
            self.experiment_config['global_map_resolution'] = self.resolution
            self.experiment_config['global_origin'] = self.origin
            print(f"INFO: Updated global map parameters for {experiment_type} suite.")
    
    def _build_polygon_test_plan(self, trials_per_env: int) -> None:
        """Build test plan for Polygon experiments."""
        poly_config = self.suite_config['polygon_config']
        dataset_path = poly_config['dataset_path']
        test_environments = poly_config['test_environments']
        prefix = poly_config['file_prefix']
        suffix = poly_config['file_suffix']

        for env_id in test_environments:
            map_file = f"{prefix}{env_id}{suffix}"
            for trial_num in range(trials_per_env):
                test_config = {
                    'type': 'Polygon',
                    'env_id': env_id,
                    'trial_num': trial_num,
                    'dataset_path': dataset_path,
                    'map_file': map_file
                }
                self.test_plan.append(test_config)
        
    def _build_barn_test_plan(self, trials_per_env: int) -> None:
        """Build test plan for BARN experiments."""
        barn_config = self.suite_config['barn_config']
        dataset_path = barn_config['dataset_path']
        test_environments = barn_config['test_environments']
        
        for env_id in test_environments:
            for trial_num in range(trials_per_env):
                test_config = {
                    'type': 'BARN',
                    'env_id': env_id,
                    'trial_num': trial_num,
                    'dataset_path': dataset_path,
                    'map_file': f"BARN_{env_id:04d}.npy"
                }
                self.test_plan.append(test_config)
                
    def _build_dumbbell_test_plan(self, trials_per_env: int) -> None:
        """Build test plan for DumbbellSweep experiments."""
        base_dumbbell_params = self.suite_config['base_dumbbell_params']
        dumbbell_config = self.suite_config['dumbbell_config']

        bottleneck_widths = base_dumbbell_params['bottleneck_widths']
        bottleneck_lengths = dumbbell_config['bottleneck_lengths']
        
        for width in bottleneck_widths:
            for length in bottleneck_lengths:
                for trial_num in range(trials_per_env):
                    test_config = {
                        # No dataset path because maps are generated on the fly
                        'type': 'DumbbellSweep',
                        'width': width,
                        'length': length,
                        'trial_num': trial_num,
                        'map_file': f"dumbbell_w{width:02d}_l{length:02d}.npy"
                    }
                    self.test_plan.append(test_config)
    
    def _build_budget_sweep_test_plan(self, trials_per_env: int) -> None:
        """Builds a test plan for sweeping over bottleneck widths and sampling budgets."""
        base_dumbbell_params = self.suite_config['base_dumbbell_params']
        budget_config = self.suite_config['budget_config']

        bottleneck_widths = base_dumbbell_params['bottleneck_widths']

        sampling_budgets = budget_config['sampling_budgets_to_test']
        fixed_length = budget_config['fixed_bottleneck_length']
        
        for width in bottleneck_widths:
            for budget in sampling_budgets:
                for trial_num in range(trials_per_env):
                    test_config = {
                        'type': 'BudgetSweep',
                        'width': width,
                        'length': fixed_length,
                        'sampling_budget': budget,
                        'trial_num': trial_num,
                        'map_file': f"dumbbell_w{width:02d}_l{fixed_length:02d}.npy"
                    }
                    self.test_plan.append(test_config)
    
    def __iter__(self) -> Iterator['EnvironmentsManager']:
        """Make Environment iterable."""
        self.current_index = 0
        return self
        
    def __next__(self) -> 'EnvironmentsManager':
        """Get next test configuration and load corresponding map."""
        if self.current_index >= len(self.test_plan):
            raise StopIteration
            
        # Load the current test configuration
        self.current_config = self.test_plan[self.current_index]
        self._load_current_map()
        
        self.current_index += 1
        return self
        
    def _load_current_map(self) -> None:
        """Load the map for the current test configuration."""
        config = self.current_config
        
        if config['type'] == 'BARN':
            self._load_barn_map(config)
        elif config['type'] in ['DumbbellSweep', 'BudgetSweep']:
            self._load_dumbbell_map(config)
        elif config['type'] == 'Polygon':
            self._load_polygon_map(config)
        else:
            raise ValueError(f"Unknown test type: {config['type']}")
    
    def _load_barn_map(self, config: Dict[str, Any]) -> None:
        """Load BARN dataset map."""
        map_path = os.path.join(config['dataset_path'], config['map_file'])
        
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"BARN map file not found: {map_path}")
            
        self.current_map = np.load(map_path)
        
    def _load_dumbbell_map(self, config: Dict[str, Any]) -> None:
        """Generate Dumbbell map on-the-fly and save to results directory."""
        # Generate map on-the-fly using parameters from config
        base_dumbbell_params = self.suite_config['base_dumbbell_params']
        
        # Get parameters for map generation
        map_size = tuple(base_dumbbell_params['map_size_cells'])  # (width, height)
        room_width = base_dumbbell_params['room_width_cells']
        bottleneck_width = config['width'] + 2  # Add 2 for boundary walls
        bottleneck_length = config['length']    # Length as specified
        
        # Generate the map
        # self.current_map = create_dumbbell_map(
        self.current_map = create_bottom_bottleneck_dumbbell_map(
            map_size=map_size,
            room_width=room_width,
            bottleneck_width=bottleneck_width,
            bottleneck_length=bottleneck_length
        )
        
        # Save map and visualization if results directory is set
        if self.results_base_dir is not None:
            # Create filenames
            map_filename = f"dumbbell_w{config['width']:02d}_l{config['length']:02d}.npy"
            viz_filename = f"dumbbell_w{config['width']:02d}_l{config['length']:02d}.png"
            
            # Save map
            map_path = os.path.join(self.dumbbell_maps_dir, map_filename)
            np.save(map_path, self.current_map)
            
            # Save visualization
            viz_path = os.path.join(self.dumbbell_viz_dir, viz_filename)
            title = f"Dumbbell Environment (w={config['width']}, l={config['length']})"
            visualize_map(
                map_data=self.current_map,
                title=title,
                save_path=viz_path,
                resolution=self.resolution,
                origin=self.origin
            )
    
    def _load_polygon_map(self, config: Dict[str, Any]) -> None:
        """Load Polygon pickle file and generate the occupancy grid."""
        map_path = os.path.join(config['dataset_path'], config['map_file'])
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Polygon map file not found: {map_path}")

        with open(map_path, 'rb') as f: # Load the pickle file
            polygon_dict = pickle.load(f)

        # raw dictionary for later use (getting start/goal)
        self.current_polygon_data = polygon_dict

        # Extract and scale vertices
        poly_config = self.suite_config['polygon_config']
        scale_factor = poly_config['polygon_scale_factor']
        try:
            vertices = polygon_dict["level_2_polygon"] * scale_factor
        except KeyError:
            raise KeyError(f"Key 'level_2_polygon' not found in {config['map_file']}.")

        # Generate the occupancy grid
        map_size = tuple(poly_config['map_size_cells'])

        # Rasterize using the robust utility (handles coordinate conversions) 
        # Uses the dynamically updated self.resolution and self.origin
        self.current_map = rasterize_polygon_interior(
            vertices, map_size=map_size, resolution=self.resolution,
            origin=self.origin
        )
        
        # Ensure boundary cells are always marked as obstacles (enclosed environment)
        self.current_map[0, :] = 1   # Top boundary
        self.current_map[-1, :] = 1  # Bottom boundary  
        self.current_map[:, 0] = 1   # Left boundary
        self.current_map[:, -1] = 1  # Right boundary
    
    def get_trial_params(self, trial_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get start and goal positions for the current environment and trial.
        
        Args:
            trial_num: Trial number (0-based)
            
        Returns:
            (start_state, goal_position) - start as [x,y,theta,v], goal as [x,y]
        """
        if self.current_config is None:
            raise RuntimeError("No current configuration loaded. Use iterator protocol first.")
            
        config = self.current_config
        
        if config['type'] == 'BARN':
            start_state, goal_pos = self._get_barn_trial_params()
        elif config['type'] in ['DumbbellSweep', 'BudgetSweep']:
            start_state, goal_pos = self._get_dumbbell_trial_params(trial_num)
        elif config['type'] == 'Polygon':
            start_state, goal_pos = self._get_polygon_trial_params(trial_num)
        else:
            raise ValueError(f"Unknown test type: {config['type']}")

        return start_state, goal_pos
        
    def _get_barn_trial_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get start/goal for BARN experiments."""
        # Fixed positions for BARN environments  
        initial_velocity = self.experiment_config['vrange'][0]
        start_state = np.array([0.0, 0.0, 0.0, initial_velocity])  # [x, y, theta, v]
        goal_position = np.array([6.5, 0.0])          # [x, y]
        return start_state, goal_position
        
    def _get_dumbbell_trial_params(self, trial_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get start/goal for Dumbbell experiments.
        also handles multi-trial start positions by spreading them vertically across the free space of the room."""
        base_dumbbell_params = self.suite_config['base_dumbbell_params']
        trials_per_env = self.suite_config['trials_per_environment']
        
        map_w_cells, map_h_cells = base_dumbbell_params['map_size_cells']
        room_w_cells = base_dumbbell_params['room_width_cells']
        
        # Get the specific bottleneck length for THIS environment from the current config
        # Add 2 to get the total length, matching the map generation logic
        bottleneck_l_cells = self.current_config['length'] + 2

        # Get the top and bottom wall coordinates of the room in the grid
        room_height_cells = int(map_h_cells * 0.8)
        top_wall_y_grid = (map_h_cells - room_height_cells) // 2
        bottom_wall_y_grid = top_wall_y_grid + room_height_cells - 1
        
        # Define the padding in meters and convert to grid cells
        padding_m = self.suite_config['start_lin_space_padding_meters']
        padding_cells = padding_m / self.resolution
        
        # Define the padded segment for the start positions
        y_max_padded_grid = top_wall_y_grid + padding_cells
        y_min_padded_grid = bottom_wall_y_grid - padding_cells

        possible_y_coords_grid = np.linspace(y_min_padded_grid, y_max_padded_grid, num=trials_per_env)
        start_y_grid = possible_y_coords_grid[trial_num]
        
        # Calculate the start and end of the passage in grid coordinates
        passage_x_start = (map_w_cells - bottleneck_l_cells) // 2
        passage_x_end = passage_x_start + bottleneck_l_cells
        
        # Calculate room centers based on the actual room boundaries
        start_x_grid = (passage_x_start - room_w_cells + passage_x_start) / 2
        goal_x_grid = (passage_x_end + (passage_x_end + room_w_cells)) / 2

        room_height_cells = int(map_h_cells * 0.8)
        room_y_start_cells = (map_h_cells - room_height_cells) // 2
        bottleneck_w_cells = self.current_config['width'] + 2
        # Place the goal y-coordinate in the middle of the passage.
        goal_y_grid = room_y_start_cells + (bottleneck_w_cells / 2.0)
        # goal_y_grid = map_h_cells // 2

        # Convert grid coordinates to METERS relative to the map's (0,0) cell
        start_x_map_m = start_x_grid * self.resolution
        start_y_map_m = start_y_grid * self.resolution
        goal_x_map_m = goal_x_grid * self.resolution
        goal_y_map_m = goal_y_grid * self.resolution

        # Add the global origin offset to get the final WORLD coordinates
        start_x_world = start_x_map_m + self.origin[0]
        start_y_world = start_y_map_m + self.origin[1]
        goal_x_world = goal_x_map_m + self.origin[0]
        goal_y_world = goal_y_map_m + self.origin[1]

        initial_velocity = self.experiment_config['vrange'][0]
        start_state = np.array([start_x_world, start_y_world, 0.0, initial_velocity])
        goal_position = np.array([goal_x_world, goal_y_world])
        return start_state, goal_position
    
    def _get_polygon_trial_params(self, trial_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get start/goal for Polygon experiments from the loaded data.
        Handles repeating base configurations (e.g., A->B, B->A) multiple times.
        """

        if self.current_polygon_data is None:
             raise RuntimeError("Polygon data not loaded before getting trial params.")
        data = self.current_polygon_data

        try:
            if "trials" not in data or not isinstance(data["trials"], list):
                 raise KeyError("Key 'trials' not found or is not a list. Ensure you are using the A* Generated dataset.")
            
            available_configurations = data["trials"]
            num_configurations = len(available_configurations)

            if num_configurations == 0:
                raise ValueError("No trial configurations found in the polygon data file.")

            # Handle multiple trials by repeating configurations sequentially.
            # E.g., 6 trials, 2 configs -> 3 repeats per config.
            # Trials 0, 1, 2 use config 0. Trials 3, 4, 5 use config 1.
            trials_per_env = self.suite_config['trials_per_environment']

            # Calculate repeats per configuration (using float division for potentially uneven distribution).
            repeats_per_configuration = trials_per_env / num_configurations

            # Calculate the configuration index: floor(trial_num / repeats_per_configuration)
            config_index = int(np.floor(trial_num / repeats_per_configuration))

            # Ensure index is within bounds (important for edge cases/floating point).
            config_index = min(config_index, num_configurations - 1)

            # Access the specific trial (e.g., 0 for Trial A, 1 for Trial B)
            current_trial = available_configurations[config_index]
            start_pose = current_trial["start_pose"] # [x, y, theta]
            goal_pos = current_trial["goal_pos"]     # [x, y]
        except KeyError:
            raise KeyError(f"Keys 'xstart' or 'xgoal' not found in the polygon data file.")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Error accessing trial data for trial {trial_num}: {e}")

        if start_pose is None or goal_pos is None:
            raise ValueError(f"Start or goal position is None in trial {trial_num} of the current polygon data file.")

        # these coordinates are already in world frame (meters) and do NOT require the scaling factor applied to vertices.
        start_x, start_y, start_theta = start_pose
        initial_velocity = self.experiment_config['vrange'][0]
        start_state = np.array([start_x, start_y, start_theta, initial_velocity])
        goal_position = np.array([goal_pos[0], goal_pos[1]])
        
        return start_state, goal_position
        
    def check_collision(self, state: np.ndarray) -> bool:
        """Check if vehicle state collides with obstacles.
        Args: state: [x, y, theta, v] vehicle state
        Returns: True if collision detected
        """
        if self.current_map is None:
            return False

        use_point_robot = self.experiment_config['point_robot_mode']
        if use_point_robot:
            return check_collision_point(
                self.current_map, state[:2], self.resolution, self.origin)
        else:
            boundary_points = get_vehicle_boundary_points(
                state, self.vehicle_length, self.vehicle_width)
            return check_collision_polygon(
                self.current_map, boundary_points, self.resolution, self.origin)
    
    def check_goal_reached(self, state: np.ndarray, goal: np.ndarray, tolerance: float) -> bool:
        """Check if goal is reached.
        Args:
            state: [x, y, theta, v] vehicle state
            goal: [x, y] goal position
            tolerance: Goal tolerance radius
        Returns: True if goal reached
        """
        distance = np.linalg.norm(state[:2] - goal)
        return distance <= tolerance or state[0] > goal[0]  # Success if past goal x
        
    def get_current_env_info(self) -> Dict[str, Any]:
        """Get information about the current environment."""
        if self.current_config is None:
            return {}
            
        info = {
            'config': self.current_config.copy(),
            'map_shape': self.current_map.shape if self.current_map is not None else None,
            'total_tests': len(self.test_plan),
            'current_index': self.current_index - 1  # Adjust for 0-based indexing
        }
        return info 


''' Helper functions for generating maps '''

def create_dumbbell_map(map_size, room_width, bottleneck_width, bottleneck_length):
    """
    Generates a dumbbell-shaped occupancy grid map with 1-pixel thick walls.

    This function draws the boundary of the dumbbell shape as a series of
    connected line segments, ensuring a correct and continuous 1-pixel wall.

    Args:
        map_size (tuple): The overall (width, height) of the map in cells.
        room_width (int): The width of the left and right rectangular rooms.
        bottleneck_width (int): The width (height dimension) of the connecting passage.
        bottleneck_length (int): The length (width dimension) of the connecting passage.

    Returns:
        numpy.ndarray: A 2D numpy array representing the generated map.
    """
    # Add an assertion to ensure the total width can accommodate two walls.
    assert bottleneck_width > 2, "Total bottleneck_width must be > 2 to allow for two 1-pixel walls."

    # 1. Start with a map of all free space (value 0)
    grid_map = np.zeros(shape=(map_size[1], map_size[0]), dtype=np.int8)
    map_h, map_w = grid_map.shape

    # 2. Calculate the key coordinates for the dumbbell shape
    passage_y_start = (map_h - bottleneck_width) // 2
    passage_y_end = passage_y_start + bottleneck_width
    passage_x_start = (map_w - bottleneck_length) // 2
    passage_x_end = passage_x_start + bottleneck_length
    left_room_x_start = passage_x_start - room_width
    right_room_x_end = passage_x_end + room_width
    room_height = int(map_h * 0.8)
    room_y_start = (map_h - room_height) // 2
    room_y_end = room_y_start + room_height

    # 3. Draw the entire boundary piece-by-piece
    
    # --- Left Room Walls ---
    grid_map[room_y_start, left_room_x_start:passage_x_start] = 1
    grid_map[room_y_end - 1, left_room_x_start:passage_x_start] = 1
    grid_map[room_y_start:room_y_end, left_room_x_start] = 1
    
    # --- Right Room Walls ---
    grid_map[room_y_start, passage_x_end:right_room_x_end] = 1
    grid_map[room_y_end - 1, passage_x_end:right_room_x_end] = 1
    grid_map[room_y_start:room_y_end, right_room_x_end - 1] = 1

    # --- Passage Walls (Connects the two rooms) ---
    # Top wall of the passage
    grid_map[passage_y_start, passage_x_start:passage_x_end] = 1
    # Bottom wall of the passage
    grid_map[passage_y_end - 1, passage_x_start:passage_x_end] = 1
    
    # --- Vertical Connectors ---
    # Top connectors
    grid_map[room_y_start:passage_y_start, passage_x_start] = 1
    grid_map[room_y_start:passage_y_start, passage_x_end - 1] = 1
    # Bottom connectors
    grid_map[passage_y_end:room_y_end, passage_x_start] = 1
    grid_map[passage_y_end:room_y_end, passage_x_end - 1] = 1
    return grid_map

def create_bottom_bottleneck_dumbbell_map(map_size, room_width, bottleneck_width, bottleneck_length):
    """
    Generates a dumbbell-shaped occupancy grid map with the bottleneck entrance
    at the bottom of the two main rooms.

    Args:
        map_size (tuple): The overall (width, height) of the map in cells.
        room_width (int): The width of the left and right rectangular rooms.
        bottleneck_width (int): The width (height dimension) of the connecting passage.
        bottleneck_length (int): The length (width dimension) of the connecting passage.

    Returns:
        numpy.ndarray: A 2D numpy array representing the generated map where 1s are obstacles.
    """
    # Ensure the bottleneck is wide enough for two walls.
    assert bottleneck_width > 2, "Total bottleneck_width must be > 2 to allow for two 1-pixel walls."

    # 1. Start with a map of all free space (value 0)
    grid_map = np.zeros(shape=(map_size[1], map_size[0]), dtype=np.int8)
    map_h, map_w = grid_map.shape

    # 2. Calculate the key coordinates for the dumbbell shape.
    # The horizontal layout remains centered.
    passage_x_start = (map_w - bottleneck_length) // 2
    passage_x_end = passage_x_start + bottleneck_length
    left_room_x_start = passage_x_start - room_width
    right_room_x_end = passage_x_end + room_width

    # The main rooms are still vertically centered in the map.
    room_height = int(map_h * 0.8)
    room_y_start = (map_h - room_height) // 2
    room_y_end = room_y_start + room_height

    # Position the passage at the top of the main rooms.
    passage_y_start = room_y_start
    passage_y_end = passage_y_start + bottleneck_width

    # 3. Draw the entire boundary of the shape piece-by-piece.
    # --- Left Room Walls ---
    grid_map[room_y_start, left_room_x_start:passage_x_start] = 1
    # Top wall of the left room
    grid_map[room_y_end - 1, left_room_x_start:passage_x_start] = 1
    grid_map[room_y_start:room_y_end, left_room_x_start] = 1

    # --- Right Room Walls ---
    grid_map[room_y_start, passage_x_end:right_room_x_end] = 1
    # Top wall of the right room
    grid_map[room_y_end - 1, passage_x_end:right_room_x_end] = 1

    # Right-most wall of the right room
    grid_map[room_y_start:room_y_end, right_room_x_end - 1] = 1

    # --- Passage Walls (Connecting the rooms at the bottom) ---
    grid_map[passage_y_start, passage_x_start:passage_x_end] = 1
    grid_map[passage_y_end - 1, passage_x_start:passage_x_end] = 1

    # vertical connectors connecting the passage to the top of the rooms
    grid_map[passage_y_end:room_y_end, passage_x_start] = 1
    grid_map[passage_y_end:room_y_end, passage_x_end - 1] = 1

    return grid_map