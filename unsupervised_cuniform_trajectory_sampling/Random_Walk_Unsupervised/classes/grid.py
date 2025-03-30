import numpy as np

class Grid:
    def __init__(self, thresholds):
        """ 
        Args:
            thresholds (list): A list of threshold values [Dx, Dy, Dtheta] for each dimension.
        """
        self.thresholds = np.array(thresholds)
        self.dims = len(thresholds)

    def get_index(self, state):
        """
        Computes the index of the grid cell for a given state by dividing the state value by thresholds and rounding.

        Args:
            state (list or np.ndarray): The state values [x, y, theta].

        Returns:
            tuple: The index of the grid cell in each dimension.
        """
        state = np.array(state)
        indices = np.round(state / self.thresholds).astype(int)
        # print("state: ", state)
        # print("indices: ", indices)
        return tuple(indices)

    def get_grid_center(self, index):
        """
        Computes the center point of a grid cell given its index.

        Args:
            index (tuple): The grid index of the cell.

        Returns:
            np.ndarray: The center point of the grid cell.
        """
        index = np.array(index)
        center = index * self.thresholds
        return center

    def perturb_state(self, center_point, division_factor=2, num_samples=5):
        """
        Generates perturbed points around the center point within half the cell dimensions.

        Args:
            center_point (tuple): The center point of the grid cell.
            grid (Grid): The grid object.
            division_factor (int): Factor to determine the perturbation range within the cell dimensions.
            num_samples (int): Number of perturbed points to generate.

        Returns:
            list: List of perturbed points as tuples.
        """
        # Convert center point to NumPy array for vectorized operations
        center_point_array = np.array(center_point).astype(np.float32)

        # Calculate perturbations for all dimensions at once using broadcasting
        perturbations = np.random.uniform(
            -self.thresholds / division_factor,  # Lower bound for each dimension
            self.thresholds / division_factor,   # Upper bound for each dimension
            (num_samples, self.dims)             # Generate an array of size (num_samples, dims)
        )

        # Generate perturbed points by adding perturbations to the center point
        perturbed_points_array = center_point_array + perturbations
        # Convert perturbed points to tuples
        perturbed_points = [tuple(point) for point in perturbed_points_array]

        return perturbed_points

    def perturb_state_deterministic(self, center_point, division_factor=4):
        """
        Deterministic version of the perturbation that generates 27 samples around the center point.
        The operation is done in three steps:
        1. Perturb the y-coordinate by adding, subtracting, and using no perturbation (generate 3 points).
        2. For each of the 3 points from step 1, perturb the theta-coordinate by adding, subtracting, and using no perturbation (generate 9 points).
        3. For each of the 9 points from step 2, perturb the x-coordinate by adding, subtracting, and using no perturbation (generate 27 points).
        
        Args:
            center_point (tuple): The center point of the grid cell (x, y, theta).
            division_factor (int): Factor to determine the perturbation range within the cell dimensions.
                                By default, we use 1/4th of the threshold for perturbation.

        Returns:
            np.ndarray: Array of shape (27, 3) containing the 27 perturbed points.
        """
        # Convert center point to NumPy array for easy manipulation
        center_point_array = np.array(center_point).astype(np.float32)

        # Calculate the step sizes for each dimension (x, y, theta)
        step_x = self.thresholds[0] / division_factor
        step_y = self.thresholds[1] / division_factor
        step_theta = self.thresholds[2] / division_factor

        # Step 1: Perturb the y-coordinate (generate 3 points by adding, subtracting, and no perturbation for y)
        y_offsets = np.array([-step_y, 0, step_y])
        points_y = center_point_array + np.array([[0, offset, 0] for offset in y_offsets])

        # Step 2: Perturb the theta-coordinate for the 3 y-perturbed points (generate 9 points)
        theta_offsets = np.array([-step_theta, 0, step_theta])
        points_theta = np.array([point + np.array([0, 0, offset]) for point in points_y for offset in theta_offsets])

        # Step 3: Perturb the x-coordinate for the 9 theta-perturbed points (generate 27 points)
        x_offsets = np.array([-step_x, 0, step_x])
        perturbed_points = np.array([point + np.array([offset, 0, 0]) for point in points_theta for offset in x_offsets])

        return perturbed_points