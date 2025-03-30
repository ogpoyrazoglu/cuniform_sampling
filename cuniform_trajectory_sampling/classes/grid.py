import numpy as np
from numba import njit
from pprint import pprint 

@njit(parallel=True) #TODO: can apply numba optimization to other functions for further optimization
def get_grid_centers_vectorized(indices, thresholds):
    return indices * thresholds
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
        state_arr = np.array(state)
        raw_indices = np.round(state_arr / self.thresholds).astype(int)

        if self.dims >= 3:  # Assume the third dimension is theta - yaw angle
            max_theta_index = int(np.pi / self.thresholds[2])  # Calculate max index dynamically
            # Normalize to [-max_theta_index, max_theta_index], unifying -max_theta_index and max_theta_index
            raw_indices[-1] = ((raw_indices[-1] + max_theta_index) % (2 * max_theta_index)) - max_theta_index
            if raw_indices[-1] == -max_theta_index:
                raw_indices[-1] = max_theta_index

        return tuple(raw_indices)
    
    def get_index_vectorized(self, states):
        """
        Vectorized version, now inputs need to be np array, shape (n, 3)
        Returns:
            np.ndarray: Array of shape (n, 3) containing the grid indices for each state.
        """
        raw_indices = np.round(states / self.thresholds).astype(int)
        if self.dims >= 3:  # Only wrap the last column (theta-index)
            # NOTE: it's best to have theta threshold divides by pi evenly for symmetric binning.
            max_theta_index = int(np.pi / self.thresholds[2])
            ''' Line below normalize theta to [-max_theta_index, max_theta_index], unifying -max_theta_index and max_theta_index
                An example to understand, if max_theta_index = 40 (every cell span 4.5 degrees)
                    mod (2 * max_theta_index) is the length of desired circular range. collapses any integer into range [0,79].
                    minus 40 at the end to shifts everything down by 40. Now [0,79] becomes [-40,39].
                    plus 40 beforehand to ensure -40 lands on 0 before modulo, which then after modulo and subtracting 40 becomes -40.
            '''
            raw_indices[:, -1] = ((raw_indices[:, -1] + max_theta_index) % (2 * max_theta_index)) - max_theta_index
            raw_indices[raw_indices[:, -1] == -max_theta_index, -1] = max_theta_index
        return raw_indices

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
    
    def get_grid_centers_vectorized(self, indices):
        """
        Vectorized version, now input need to be np array
        Returns:
            np.ndarray: Array of shape (n, 3) containing the grid centers for each index.
        """
        assert np.issubdtype(indices.dtype, np.integer), "Input indices must be of integer type"
        centers = indices * self.thresholds
        return centers

    def get_adjacent_cells(self, index):
        """
        Computes the adjacent grid cells for a given index.
        Args:
            index (tuple): The grid index of the cell.
        Returns:
            np.ndarray: Array of shape (n_adj, dims) containing the indices of adjacent cells.(including the original given center index)
        """
        index = np.array(index)
        deltas = [-1, 0, 1]  # Possible offsets in each dimension
        shifts = np.array(np.meshgrid(*[deltas] * self.dims)).T.reshape(-1, self.dims) # generate all combinations of offsets for each dimension
        adjacent_cells = index + shifts # Apply offsets to the original index
        return adjacent_cells

    def perturb_state(self, center_point, division_factor=4, num_samples=5):
        """
        Generates perturbed points around the center point within half the cell dimensions.

        Args:
            center_point (tuple): The center point of the grid cell.
            grid (Grid): The grid object.
            division_factor (int): Factor to determine the perturbation range within the cell dimensions. Assume it's not 0
            num_samples (int): Number of perturbed points to generate.

        Returns:
            np.ndarray: Array of perturbed points of shape (num_samples, dims).
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
        return perturbed_points_array

    def perturb_state_deterministic_vectorized(self, points, division_factor=4): #NOTE: this version generalize to different dimensions
        """
        Generalized version of the deterministic perturbation function that generates perturbed samples
        around each input point. This function handles an array of points with arbitrary dimensions simultaneously.

        Args:
            points (np.ndarray): An array of shape (N, D), where N is the number of points, and
                                D is the dimensionality of the configuration space.
            division_factor (int): Factor to determine the perturbation range within the cell dimensions.
                                By default, we use 1/4th of the thresholds for perturbation.

        Returns:
            np.ndarray: An array of shape (N * num_perturbations, D) containing the perturbed points
                        for each input point, where `num_perturbations` = 3^D.
        """
        if abs(division_factor) <= 0.000001:
            return points  # return points without perturbation
        num_dims = points.shape[1] # Number of dimensions (D)
        # Calculate the step sizes for each dimension
        step_sizes = self.thresholds[:num_dims] / division_factor
        offsets = np.array([-1, 0, 1]) # generate offsets for each dimension (-1, 0, 1)

        # Create perturbations grid dynamically for the given dimensionality
        perturbations = np.array(np.meshgrid(*[step_sizes[d] * offsets for d in range(num_dims)]))
        perturbations = perturbations.T.reshape(-1, num_dims)

        # Add perturbations to each point
        perturbed_points = points[:, None, :] + perturbations[None, :, :]
        # Reshape to (N * 27, 3)
        perturbed_points = perturbed_points.reshape(-1, num_dims)
        return perturbed_points

    def perturb_state_deterministic_no_diagonal(self, center_point, division_factor=4): #NOTE: assume 3d configuration space
        """
        Generates perturbations for the given state, excluding diagonal perturbations.

        The function only perturbs along the x, y, and theta axes (±1) and excludes
        combinations of multiple axes.
        Args:
            center_point (tuple): The center point of the grid cell (x, y, theta).
            division_factor (int): Factor to determine the perturbation range within
                                the cell dimensions.

        Returns:
            np.ndarray: Array of shape (7, 3) containing the 7 perturbed points.
        """
        if abs(division_factor) <= 0.000001:
            return np.array([center_point])  # Don't do perturbation
        
        # Convert center point to NumPy array for easy manipulation
        center_point_array = np.array(center_point).astype(np.float32)

        # Calculate the step sizes for each dimension (x, y, theta)
        step_x = self.thresholds[0] / division_factor
        step_y = self.thresholds[1] / division_factor
        step_theta = self.thresholds[2] / division_factor

        # Generate the 7 perturbed points
        perturbed_points = [
            center_point_array,  # Center point
            center_point_array + np.array([step_x, 0, 0]),  # +1 in x
            center_point_array - np.array([step_x, 0, 0]),  # -1 in x
            center_point_array + np.array([0, step_y, 0]),  # +1 in y
            center_point_array - np.array([0, step_y, 0]),  # -1 in y
            center_point_array + np.array([0, 0, step_theta]),  # +1 in theta
            center_point_array - np.array([0, 0, step_theta])  # -1 in theta
        ]
        return np.array(perturbed_points)
