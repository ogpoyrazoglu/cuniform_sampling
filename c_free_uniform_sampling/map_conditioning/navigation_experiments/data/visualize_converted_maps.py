"""
Script to visualize converted BARN maps.

This script reads all converted .npy files from BARN_npy_files_0.05_padded directory
and creates visualization images saved to BARN_npy_files_0.05_padded_visualization directory.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

def visualize_map(map_data, title, save_path, resolution=0.05, origin=None):
    """Visualize a single map.
    Args:
        map_data: 2D numpy array representing the occupancy grid
        title: Title for the plot
        save_path: Path to save the visualization
        resolution: Map resolution in meters/cell
        origin: Map origin [x, y] in meters
    """
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Create colormap: white for free space, red for obstacles
    colors = ['white', 'red']
    cmap = ListedColormap(colors)
    
    # Calculate extent for proper coordinate display
    if origin is None:
        origin = [0, 0]  # Default origin
    
    grid_height, grid_width = map_data.shape
    x_coords = np.arange(grid_width) * resolution + origin[0]
    y_coords = np.arange(grid_height) * resolution + origin[1]
    
    # Plot the map
    plt.imshow(map_data, cmap=cmap, origin='lower', 
               extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    
    # Add grid lines to show cell boundaries (optional, for clarity)
    for i in range(0, grid_width, 10):  # Every 10 cells
        x_line = x_coords[i]
        plt.axvline(x=x_line, color='gray', alpha=0.3, linewidth=0.5)
    for i in range(0, grid_height, 10):  # Every 10 cells
        y_line = y_coords[i]
        plt.axhline(y=y_line, color='gray', alpha=0.3, linewidth=0.5)
    
    # Add labels and title
    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color='white', label='Free Space'),
        patches.Patch(color='red', label='Obstacles')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add map info as text
    info_text = f"Map Size: {grid_width}×{grid_height} cells\n"
    info_text += f"Resolution: {resolution:.3f} m/cell\n"
    info_text += f"Physical Size: {grid_width*resolution:.1f}×{grid_height*resolution:.1f} m"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Set equal aspect ratio and tight layout
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main visualization function."""
    input_dir = "BARN_npy_files_0.05_padded"
    output_dir = "BARN_npy_files_0.05_padded_visualization"
    
    if not os.path.exists(input_dir):
        print(f" Input directory '{input_dir}' does not exist!")
        print("Please run convert_barn_maps.py first.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    
    # Get all .npy files in input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if not input_files:
        print(f" No .npy files found in '{input_dir}'!")
        return
    
    print(f"Found {len(input_files)} .npy files to visualize")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Process each file
    for i, filename in enumerate(input_files):
        input_path = os.path.join(input_dir, filename)
        
        # Create output filename (replace .npy with .png)
        output_filename = filename.replace('.npy', '.png')
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            map_data = np.load(input_path) # Load the map
            # Extract map ID from filename (e.g., "grid_125.npy" -> "125")
            map_id = filename.replace("grid_", "").replace(".npy", "")
            title = f"BARN Environment {map_id} (Converted & Padded)"
            visualize_map(map_data, title, output_path, resolution=0.05, origin=[0, 0])
            print(f"[{i+1:3d}/{len(input_files)}] {filename} -> {output_filename}")
            print(f"    Map size: {map_data.shape}")
        except Exception as e:
            print(f"[{i+1:3d}/{len(input_files)}] ERROR processing {filename}: {e}")
    print("-" * 60)
    print(f" Visualization complete! All images saved to {output_dir}")
    print(f"Generated {len(input_files)} visualization images")

if __name__ == "__main__":
    main() 