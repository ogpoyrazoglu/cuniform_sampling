"""Create simple test BARN environment files."""
import numpy as np
import os


def create_test_barn_environment(env_id: int, width: int, height: int) -> np.ndarray:
    """Create a simple test BARN environment.
    
    Args:
        env_id: Environment ID
        width: Grid width 
        height: Grid height
        
    Returns:
        Occupancy grid (1=obstacle, 0=free)
    """
    # Create base free space
    grid = np.zeros((height, width), dtype=int)
    
    # Add some simple obstacles based on env_id
    if env_id == 0:
        # Simple corridor with 3 obstacles - KEEP UNCHANGED
        grid[20:30, 50:60] = 1  # Obstacle 1
        grid[60:70, 100:110] = 1  # Obstacle 2
        grid[40:50, 150:160] = 1  # Obstacle 3

    elif env_id == 1:
        # Serpentine maze to force extreme right/left turns.
        # This version is self-contained with boundaries the planner can see.
        
        # Add 5-pixel thick top and bottom boundaries
        grid[0:5, :] = 1  # Top wall
        grid[-5:, :] = 1 # Bottom wall
        
        # Wall 1: Forces an initial hard right turn by blocking the upper path.
        # Extends from y=30 to the bottom boundary (y=90).
        grid[30:90, 60:70] = 1
        
        # Wall 2: Forces a hard left turn by blocking the lower path.
        # Extends from the top boundary (y=0) to y=60.
        grid[0:60, 120:130] = 1
        
        # Wall 3: Forces another hard right turn.
        grid[30:90, 180:190] = 1
        
        # Wall 4: Forces a final hard left turn to reach the goal area.
        grid[0:60, 240:250] = 1
        
    elif env_id == 121:
        # Add 5-pixel thick top and bottom boundaries
        grid[0:5, :] = 1  # Top wall
        grid[-5:, :] = 1 # Bottom wall
        # 15 small square obstacles scattered throughout
        grid[15:25, 40:50] = 1    # Obstacle 1
        grid[35:45, 70:80] = 1    # Obstacle 2
        grid[55:65, 30:40] = 1    # Obstacle 3
        grid[25:35, 110:120] = 1  # Obstacle 4
        grid[45:55, 140:150] = 1  # Obstacle 5
        grid[65:75, 90:100] = 1   # Obstacle 6
        grid[10:20, 170:180] = 1  # Obstacle 7
        grid[30:40, 200:210] = 1  # Obstacle 8
        grid[50:60, 230:240] = 1  # Obstacle 9
        grid[70:80, 180:190] = 1  # Obstacle 10
        grid[20:30, 260:270] = 1  # Obstacle 11
        grid[40:50, 160:170] = 1  # Obstacle 12
        grid[60:70, 120:130] = 1  # Obstacle 13
        grid[8:18, 210:220] = 1   # Obstacle 14
        grid[75:85, 60:70] = 1    # Obstacle 15
    
    return grid


def main():
    """Create test BARN files."""
    os.makedirs("test_barn_files", exist_ok=True)
    
    for env_id in [0, 1, 121]:
        grid = create_test_barn_environment(env_id, 280, 90)
        filename = f"test_barn_files/BARN_{env_id:04d}.npy"
        np.save(filename, grid)
        print(f"Created {filename} with shape {grid.shape}")


if __name__ == "__main__":
    main() 