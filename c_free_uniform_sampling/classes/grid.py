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
        if states.size == 0:
            return states

        if states.ndim != 2 or states.shape[1] != self.dims:
            print(f"Input {states} has unexpected shape: ", states.shape, " it should be (-1, self.dims) - get_index_vectorized()")
            states = np.array(states).reshape(-1, self.dims)

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
        if indices.ndim != 2 or indices.shape[1] != self.dims:
            print("Input has unexpected shape: ", indices.shape, " it should be (-1, self.dims) - get_grid_centers_vectorized()")
            indices= np.array(indices).reshape(-1, self.dims)
        centers = indices * self.thresholds
        return centers

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
        if num_dims > 2:
            # NOTE: ASSUME third axis (theta) is periodic with period 2π → wrap back into [−π,π]
            perturbed_points[:, 2] = (perturbed_points[:, 2] + np.pi) % (2*np.pi) - np.pi
        return perturbed_points
    
class AdaptiveGrid(Grid):
    """
    On demand, build a Grid whose thresholds grow linearly per level:
    """
    def __init__(self, base_thresholds, growth=0.5, dt=0.1, max_velocity=4.0, max_acceleration=6.0, max_steering_deg=30.0, wheelbase=0.324):
        """
        base_thresholds: [Δx₀, Δy₀, Δθ₀, Δv₀]
        growth:          per-level fractional increase (e.g. 0.1 = +10% per level)
        dt:              time step for dynamics
        max_velocity:    maximum velocity (m/s)
        max_acceleration: maximum acceleration magnitude (m/s²)
        max_steering_deg: maximum steering angle (degrees)
        wheelbase:       vehicle wheelbase (m)
        """
        self.base   = np.array(base_thresholds, dtype=np.float32)
        self.growth = float(growth)
        
        # Calculate upper bounds based on dynamics to ensure cell escape
        # For a state to escape a cell, the cell radius must be smaller than 
        # the maximum state change possible in one time step
        # Use conservative bounds: divide by 2 so max change moves you cleanly to next cell
        
        # Position bounds: max distance traveled at max velocity
        max_pos_change = max_velocity * dt
        pos_upper_bound = max_pos_change
        
        # Velocity bound: max velocity change due to acceleration  
        max_vel_change = max_acceleration * dt  
        vel_upper_bound = max_vel_change  
        
        # Angular bound: max angular change at max velocity and max steering
        import math
        max_angular_vel = (max_velocity / wheelbase) * math.tan(math.radians(max_steering_deg))
        max_theta_change = max_angular_vel * dt
        theta_upper_bound = max_theta_change  # Conservative: threshold = max_change (radius = max_change/2)
        
        # Set upper bounds for each dimension
        if len(base_thresholds) == 2:  # 2D case
            self.upper_bounds = np.array([pos_upper_bound, pos_upper_bound])
        elif len(base_thresholds) == 3:  # 3D case (x, y, theta)
            self.upper_bounds = np.array([pos_upper_bound, pos_upper_bound, theta_upper_bound])
        elif len(base_thresholds) == 4:  # 4D case (x, y, theta, v)
            self.upper_bounds = np.array([pos_upper_bound, pos_upper_bound, theta_upper_bound, vel_upper_bound])
        else:
            # For other dimensions, use a conservative bound
            self.upper_bounds = np.full(len(base_thresholds), max(pos_upper_bound, vel_upper_bound))
    
    def print_upper_bounds(self):
        print(f"AdaptiveGrid upper bounds: {self.upper_bounds}")

    def get_thresholds(self, level, zero_index):
        ''' sanity check flag
        zero_index: must explicitly choose 0-based (True) or 1-based (False) indexing  
            (this explicit flag forces double-check level convention and avoid off-by-one errors)
        '''
        # piecewise growth: 50% per level for the first 6, then 10% thereafter
        l = level if zero_index else level - 1
        FAST_LVLS = 6
        if l < FAST_LVLS:
            scale = 1.0 + self.growth * l
        else:
            scale = 1.0 + self.growth * FAST_LVLS + self.growth/2 * (l - FAST_LVLS)
        
        # Apply scaling but enforce upper bounds
        scaled_thresholds = self.base * scale
        clamped_thresholds = np.minimum(scaled_thresholds, self.upper_bounds)
        
        return clamped_thresholds

    def get_index(self, state, level, zero_index):
        return Grid(self.get_thresholds(level, zero_index)).get_index(state)

    def get_index_vectorized(self, states, level, zero_index):
        return Grid(self.get_thresholds(level, zero_index)).get_index_vectorized(states)

    def get_grid_center(self, index, level, zero_index):
        return Grid(self.get_thresholds(level, zero_index)).get_grid_center(index)

    def get_grid_centers_vectorized(self, indices, level, zero_index):
        return Grid(self.get_thresholds(level, zero_index)).get_grid_centers_vectorized(indices)

    def perturb_state_deterministic_vectorized(self, points, division_factor, level, zero_index): 
        return Grid(self.get_thresholds(level, zero_index)).perturb_state_deterministic_vectorized(points, division_factor)