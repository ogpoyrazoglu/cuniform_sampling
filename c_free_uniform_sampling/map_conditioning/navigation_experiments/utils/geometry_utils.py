""" geometry utilities for getting vehicle boundary points and collision checking """
import numpy as np
from typing import List, Tuple
import cv2

def check_collision_point(occupancy_grid: np.ndarray, position: np.ndarray,
                          resolution: float, origin: List[float]) -> bool:
    """Check if a single point (x, y) is in collision."""
    # Convert world coordinates to grid coordinates
    grid_x = int((position[0] - origin[0]) / resolution)
    grid_y = int((position[1] - origin[1]) / resolution)

    grid_height, grid_width = occupancy_grid.shape

    # Check boundaries (Treating out of bounds as collision)
    if not (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
        return True

    # Check occupancy
    return occupancy_grid[grid_y, grid_x] > 0.5

def get_vehicle_boundary_points(state: np.ndarray, vehicle_length: float, vehicle_width: float) -> np.ndarray:
    """Get vehicle boundary points for collision checking.
    
    Args:
        state: [x, y, theta, v] vehicle state
        vehicle_length: Vehicle length
        vehicle_width: Vehicle width
        
    Returns:
        Array of boundary points [[x1, y1], [x2, y2], ...]
    """
    x, y, theta = state[:3]
    
    # Define vehicle corners relative to center
    corners = np.array([
        [vehicle_length / 2, 0],                    # 1. Front-center
        [vehicle_length / 2, -vehicle_width / 2],   # 2. Front-left corner
        [0, -vehicle_width / 2],                    # 3. Left-center
        [-vehicle_length / 2, -vehicle_width / 2],  # 4. Rear-left corner
        [-vehicle_length / 2, 0],                   # 5. Rear-center
        [-vehicle_length / 2, vehicle_width / 2],   # 6. Rear-right corner
        [0, vehicle_width / 2],                     # 7. Right-center
        [vehicle_length / 2, vehicle_width / 2]     # 8. Front-right corner
    ])
    
    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    
    # Transform corners
    transformed_corners = corners @ R.T + np.array([x, y])
    
    return transformed_corners


def check_collision_polygon(occupancy_grid: np.ndarray, boundary_points: np.ndarray,
                           resolution: float, origin: List[float]) -> bool:
    """
    Check collision by rasterizing the vehicle polygon and checking for overlap with occupied cells.
    """
    # Create a blank mask with the same dimensions as the map
    mask = np.zeros_like(occupancy_grid, dtype=np.uint8)

    # Convert world boundary points to grid pixel coordinates
    pixel_coords = np.array(
        [(point - np.array(origin)) / resolution for point in boundary_points],
        dtype=np.int32
    )
    
    # Draw the robot's filled-in polygon on the mask
    cv2.fillPoly(mask, [pixel_coords.reshape((-1, 1, 2))], 1)

    # Check for overlap: If any cell is 1 in BOTH the robot mask AND the obstacle grid, a collision exists.
    return np.any(np.logical_and(mask, occupancy_grid))

def check_collision_along_path(occupancy_grid: np.ndarray, prev_state: np.ndarray, 
                              current_state: np.ndarray, vehicle_length: float, 
                              vehicle_width: float, resolution: float, 
                              origin: List[float], num_interpolation_steps: int = 5,
                              use_point_robot: bool = False) -> bool:
    """Check collision along the path between two robot states.
    
    This function prevents tunneling through walls by checking intermediate positions
    along the robot's path between consecutive time steps.
    
    Args:
        occupancy_grid: Environment occupancy grid
        prev_state: Previous robot state [x, y, theta, v]
        current_state: Current robot state [x, y, theta, v]
        vehicle_length: Vehicle length
        vehicle_width: Vehicle width
        resolution: Grid resolution
        origin: Grid origin [x, y]
        num_interpolation_steps: Number of intermediate positions to check
        use_point_robot: Whether to use point robot mode
        
    Returns:
        True if collision detected along the path
    """
    # Check collision at current position first
    if use_point_robot:
        if check_collision_point(occupancy_grid, current_state[:2], resolution, origin):
            return True
    else:
        current_boundary = get_vehicle_boundary_points(current_state, vehicle_length, vehicle_width)
        if check_collision_polygon(occupancy_grid, current_boundary, resolution, origin):
            return True
    
    # If positions are very close, no need to interpolate
    position_diff = np.linalg.norm(current_state[:2] - prev_state[:2])
    if position_diff < resolution * 0.5:  # Less than half a grid cell
        return False
    
    # Interpolate between previous and current positions
    for i in range(1, num_interpolation_steps + 1):
        alpha = i / (num_interpolation_steps + 1)
        
        # Linear interpolation of position
        interp_x = prev_state[0] + alpha * (current_state[0] - prev_state[0])
        interp_y = prev_state[1] + alpha * (current_state[1] - prev_state[1])
        
        # Linear interpolation of orientation (with proper angle wrapping)
        theta_diff = current_state[2] - prev_state[2]
        # Handle angle wrapping
        if theta_diff > np.pi:
            theta_diff -= 2 * np.pi
        elif theta_diff < -np.pi:
            theta_diff += 2 * np.pi
        interp_theta = prev_state[2] + alpha * theta_diff
        
        # Create interpolated state
        interp_state = np.array([interp_x, interp_y, interp_theta, current_state[3]])
        if use_point_robot:
            if check_collision_point(occupancy_grid, interp_state[:2], resolution, origin):
                 return True
        else:
            interp_boundary = get_vehicle_boundary_points(interp_state, vehicle_length, vehicle_width)
            if check_collision_polygon(occupancy_grid, interp_boundary, resolution, origin):
                return True
    return False

def calculate_traveled_distance(trajectory: np.ndarray) -> float:
    """Calculate the total distance traveled along a trajectory.
    Args: trajectory: Array of robot states, shape (N, 4), where the first two columns are [x, y] positions.
    Returns: The total distance traveled (a non-negative float).
    """
    if len(trajectory) < 2:
        return 0.0
    positions = trajectory[:, :2]  # [x, y], the position coordinates
    # get Euclidean distances between consecutive positions
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_distance = np.sum(distances)
    return float(total_distance)

def calculate_control_smoothness(controls: np.ndarray) -> float:
    """Calculates the control smoothness as the sum of absolute steering changes.
    Args: controls: Array of control inputs, shape (N, 2), where the second column is the steering angle.
    Returns: The total steering effort (a non-negative float).
    """
    if len(controls) < 2:
        return 0.0
    steering_angles = controls[:, 1] # steering angle column
    # sum of absolute differences between consecutive steering commands
    steering_effort = np.sum(np.abs(np.diff(steering_angles)))
    
    return float(steering_effort)

def rasterize_polygon_interior(vertices: np.ndarray,
                               map_size: Tuple[int, int],
                               resolution: float,
                               origin: List[float]) -> np.ndarray:
    """
    Rasterizes a polygon into an occupancy grid.
    Interior is FREE (0), Exterior/Boundary is OCCUPIED (1).
    Handles conversion between world coordinates (origin='lower') and CV2 image coordinates (origin='upper').
    """
    width, height = map_size
    # Start with a fully occupied grid (1s) in image coordinate format
    image_grid = np.ones((height, width), dtype=np.uint8)

    # Helper to convert world coordinates (bottom-left origin) to CV2 image coordinates (top-left origin)
    def world_to_image_coords(pt):
        x, y = pt
        # 1. Convert world meters to grid indices (Cartesian)
        gx = (x - origin[0]) / resolution
        gy = (y - origin[1]) / resolution
        # 2. Convert Cartesian grid indices to Image indices (Flip Y-axis)
        ix = int(gx)
        iy = int(height - 1 - gy)
        return ix, iy

    # Convert vertices to pixel coordinates suitable for CV2
    image_coords = np.array([world_to_image_coords(v) for v in vertices], dtype=np.int32)

    # Use cv2.fillPoly to fill the interior with FREE space (0) on the image grid
    cv2.fillPoly(image_grid, [image_coords.reshape((-1, 1, 2))], 0)

    # The 'image_grid' is in image coordinates. The stack expects Cartesian coordinates (origin='lower').
    # Flip it vertically to convert back.
    cartesian_grid = np.flipud(image_grid)

    # Return as int8 for consistency with other map types (BARN/Dumbbell use int8)
    return cartesian_grid.astype(np.int8)