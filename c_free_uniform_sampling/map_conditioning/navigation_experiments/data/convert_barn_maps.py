"""
A concise script to convert BARN maps. It rescales them from 0.15 to 
0.05 resolution and adds horizontal padding for start/goal areas.
"""
import os
import glob
import numpy as np
from scipy.ndimage import zoom

# --- Configuration ---
INPUT_DIR = "BARN_npy_files"
OUTPUT_DIR = "BARN_npy_files_0.05_padded"
OLD_RESOLUTION = 0.15
NEW_RESOLUTION = 0.05
HORIZONTAL_PADDING = 40 # Cells to add to the left and right

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all .npy files in the input directory
    input_files = glob.glob(os.path.join(INPUT_DIR, '*.npy'))
    zoom_factor = OLD_RESOLUTION / NEW_RESOLUTION

    print(f"Found {len(input_files)} maps. Converting and padding...")
    print("-" * 60)

    for i, in_path in enumerate(input_files):
        original_filename = os.path.basename(in_path)
        # Extract the numeric ID from the original filename (e.g., "grid_0.npy" -> 0)
        map_id_str = original_filename.replace("grid_", "").replace(".npy", "")
        map_id = int(map_id_str)

        # Construct the new filename in the "BARN_xxxx.npy" format
        new_filename = f"BARN_{map_id:04d}.npy"
        out_path = os.path.join(OUTPUT_DIR, new_filename)

        # Load the original map
        original_map = np.load(in_path)
        # Convert resolution using nearest-neighbor interpolation
        converted_map = zoom(original_map, zoom_factor, order=0)
        
        # The padding format is ((top, bottom), (left, right))
        padded_map = np.pad(
            converted_map, 
            ((0, 0), (HORIZONTAL_PADDING, HORIZONTAL_PADDING)), 
            mode='constant', 
            constant_values=0
        ) # Add horizontal padding (left and right)

        # Add a 1-pixel obstacle boundary around the entire map
        padded_map[0, :] = 1   # Top edge
        padded_map[-1, :] = 1  # Bottom edge
        padded_map[:, 0] = 1   # Left edge
        padded_map[:, -1] = 1  # Right edge

        np.save(out_path, padded_map)
        print(f"[{i+1:3d}/{len(input_files)}] Converted {original_filename} to {new_filename}: {original_map.shape} -> {padded_map.shape}")
    print("-" * 60)
    print(f" Conversion complete! All maps saved to {OUTPUT_DIR}")
