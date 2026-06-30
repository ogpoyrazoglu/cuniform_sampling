import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import time
import datetime

# Add the parent directory to the path to import classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.grid import Grid

# Import model parts
from model_parts import MapAct

# Import the visualization function
from utility_helper_map import(
    visualize_trajectories_background, 
    sample_trajectories_feasible, 
    cuda_dynamics_KS_3d_steering_angle_batched
)

def create_results_folder():
    """Create a timestamped folder for storing all visualizations."""
    now = datetime.datetime.now()
    folder_name = f"trajectory_uniformity_{now.strftime('%b')}{now.day}_{now.strftime('%H%M%S')}"
    folder_path = os.path.join(os.path.dirname(__file__), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Results will be saved to: {folder_path}")
    return folder_path

def load_level_set_representatives(file_path):
    """Load level set representatives from the pickle file."""
    assert os.path.exists(file_path), f"File not found: {file_path}"
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    assert 'pruned_level_set_representatives_across_LS' in data, \
        "Key 'pruned_level_set_representatives_across_LS' not found in data"
    
    representatives_raw = data['pruned_level_set_representatives_across_LS']
    config = data['config']
    
    # Convert sets to lists for easier handling
    representatives = []
    for level_set in representatives_raw:
        if isinstance(level_set, set):
            representatives.append(list(level_set))
        else:
            representatives.append(level_set)
    
    print(f"Loaded {len(representatives)} level sets")
    for i, level_set in enumerate(representatives):
        print(f"  Level set {i}: {len(level_set)} representatives")
    
    return representatives, config

def load_baseline_model(model_path, device):
    """Load the baseline model trained on open space."""
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    
    # Initialize model architecture using MapAct
    model = MapAct(state_dim=4, num_actions=31, hidden_dim=1024)
    
    # Load the trained weights
    model_checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(model_checkpoint)
    
    model.eval()
    model = model.to(device)
    print(f"Loaded baseline model from {model_path} on {device}")
    return model

def sample_trajectories_with_model(config, device, model=None, num_trajectories=100000):
    """Sample trajectories using either baseline model or uniform actions."""
    initial_state = np.array([0.0, 0.0, 0.0])
    
    # Create dummy tensors for SDF and costmap (empty environment)
    grid_size = 121
    dummy_sdf = np.zeros((grid_size, grid_size), dtype=np.float32)
    dummy_costmap = np.zeros((grid_size, grid_size), dtype=bool)
    
    sdf_tensor = torch.tensor(dummy_sdf, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    costmap_tensor = torch.tensor(dummy_costmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Determine trajectory length from config
    trajectory_length = (int(config['total_t'] / config['dt'])) + 1
    
    if model is not None:
        # Use the baseline model for sampling
        print(f"Sampling {num_trajectories} trajectories with baseline model...")
        trajectories = sample_trajectories_feasible(
            initial_state=initial_state,
            actions=config['actions'][:, 0],  # Extract steering angles
            dynamics_cuda=cuda_dynamics_KS_3d_steering_angle_batched,
            num_trajectories=num_trajectories,
            trajectory_length=trajectory_length,
            model=model,
            feature_extractor=None,
            config=config,
            sdf_tensor=sdf_tensor,
            costmap_tensor=costmap_tensor,
            map_embedding=None,
            uniform_sampling=False,
        )
    else:
        # Create uniform action model
        class UniformMapAct(MapAct):
            def __init__(self, num_actions):
                super().__init__(state_dim=4, num_actions=num_actions, hidden_dim=1024)
                self.num_actions = num_actions
                self.__class__.__name__ = "MapAct"
            
            def forward(self, states):
                batch_size = states.shape[0]
                uniform_probs = torch.ones((batch_size, self.num_actions), device=states.device) / self.num_actions
                return uniform_probs
        
        uniform_model = UniformMapAct(len(config['actions'])).to(device)
        
        print(f"Sampling {num_trajectories} trajectories with uniform actions...")
        trajectories = sample_trajectories_feasible(
            initial_state=initial_state,
            actions=config['actions'][:, 0],
            dynamics_cuda=cuda_dynamics_KS_3d_steering_angle_batched,
            num_trajectories=num_trajectories,
            trajectory_length=trajectory_length,
            model=uniform_model,
            feature_extractor=None,
            config=config,
            sdf_tensor=sdf_tensor,
            costmap_tensor=costmap_tensor,
            map_embedding=None,
            uniform_sampling=False,
        )
    
    return trajectories

def extract_all_states_from_trajectories(trajectories):
    """Extract all states from trajectories (excluding final state with None action)."""
    all_states = []
    for traj in trajectories:
        for state, action in traj:
            if action is not None:  # Skip the final state
                all_states.append(state)
    return np.array(all_states)

def compute_level_set_entropy(trajectory_states, representatives, grid):
    """
    Compute entropy for each level set by counting ALL representatives.
    This is the corrected version that includes zero-occurrence representatives.
    """
    entropies = []
    
    for level_idx, level_reps in enumerate(representatives):
        # Skip level 0 (single representative at origin)
        if level_idx == 0:
            continue
            
        if len(level_reps) == 0:
            entropies.append(0.0)
            continue
        
        print(f"Computing entropy for Level Set {level_idx}")
        print(f"  Number of representatives: {len(level_reps)}")
        
        # Convert representatives to numpy array
        reps_array = np.array(level_reps)
        if reps_array.ndim == 1 and len(reps_array) == 3:
            reps_array = reps_array.reshape(1, -1)
        elif reps_array.ndim == 1 and len(reps_array) % 3 == 0:
            reps_array = reps_array.reshape(-1, 3)
        
        if reps_array.ndim != 2 or reps_array.shape[1] < 3:
            entropies.append(0.0)
            continue
        
        # Get grid indices for all representatives
        rep_indices = grid.get_index_vectorized(reps_array)
        rep_indices_tuples = [tuple(idx) for idx in rep_indices]
        
        # Initialize counts for ALL representatives to zero
        # This ensures we include ALL representatives in entropy calculation,
        # even those that never get visited by trajectories
        cell_counts = {}
        for cell in rep_indices_tuples:
            cell_counts[cell] = 0
        
        print(f"  Initialized {len(cell_counts)} representative cells")
        
        # Get grid indices for trajectory states
        traj_indices = grid.get_index_vectorized(trajectory_states)
        
        # Count occurrences in representative cells
        total_hits = 0
        for idx_tuple in map(tuple, traj_indices):
            if idx_tuple in cell_counts:
                cell_counts[idx_tuple] += 1
                total_hits += 1
        
        print(f"  Total trajectory states hitting this level: {total_hits}")
        print(f"  Cells with non-zero counts: {sum(1 for count in cell_counts.values() if count > 0)}")
        
        # Convert to probability distribution
        if total_hits == 0:
            # No states hit this level set
            entropies.append(0.0)
            print(f"  No states hit level set {level_idx}, entropy = 0.0")
            continue
        
        # Normalize counts to probabilities
        counts_array = np.array(list(cell_counts.values()))
        probs = counts_array / total_hits
        
        # Add small epsilon to avoid log(0) for zero probabilities
        probs_with_epsilon = probs + 1e-8
        
        # Compute entropy: -sum(p * log(p))
        entropy = -np.sum(probs_with_epsilon * np.log(probs_with_epsilon))
        
        # Compute maximum possible entropy for this level set
        max_entropy = np.log(len(cell_counts))
        uniformity_percentage = (entropy / max_entropy * 100) if max_entropy > 0 else 0
        
        entropies.append(entropy)
        
        print(f"  Entropy: {entropy:.4f}")
        print(f"  Max possible entropy: {max_entropy:.4f}")
        print(f"  Uniformity: {uniformity_percentage:.1f}%")
        
        # Debug info for first few level sets
        if level_idx < 3:
            print(f"  Count distribution: min={np.min(counts_array)}, max={np.max(counts_array)}, mean={np.mean(counts_array):.2f}")
    
    return entropies

def save_trajectory_visualization(trajectories, method_name, results_folder):
    """Save trajectory visualization."""
    print(f"Creating trajectory visualization for {method_name}...")
    
    # Create a dummy costmap (empty environment)
    grid_size = 121
    costmap = np.zeros((grid_size, grid_size), dtype=bool)
    resolution = 0.05
    
    # Convert trajectories to the format expected by visualize_trajectories_background
    formatted_trajectories = []
    for traj in trajectories:
        formatted_traj = []
        for state, action in traj:
            if action is not None:
                formatted_traj.append((state, action))
        if len(formatted_traj) > 0:
            formatted_trajectories.append(formatted_traj)
    
    print(f"  Formatted {len(formatted_trajectories)} trajectories for visualization")
    
    # Create visualization filename
    vis_filename = f"trajectory_visualization_{method_name.replace(' ', '_').lower()}.png"
    vis_path = os.path.join(results_folder, vis_filename)
    
    # Create title
    title = f"Trajectory Visualization - {method_name}\n{len(formatted_trajectories)} trajectories sampled"
    
    # Use the existing visualization function
    visualize_trajectories_background(
        trajectories=formatted_trajectories,
        costmap=costmap,
        resolution=resolution,
        show_vis=False,
        save_vis=True,
        vis_filepath=vis_path,
        alpha=0.2,
        marker_size=0.5,
        title=title
    )
    
    print(f"  Trajectory visualization saved: {os.path.basename(vis_path)}")

def plot_entropy_comparison(results_dict, representatives, save_path):
    """Plot entropy comparison between methods with maximum entropy reference."""
    methods = list(results_dict.keys())
    
    # Skip maximum entropy if present
    if "Maximum Entropy" in methods:
        methods.remove("Maximum Entropy")
    
    num_levels = len(results_dict[methods[0]])
    # Start from level 1 since we skip level 0
    x_indices = np.arange(1, num_levels + 1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods) + 1))
    markers = ['o', 's', 'D', '^']
    
    # Plot maximum possible entropy line
    max_entropies = []
    for level_idx in range(1, len(representatives)):  # Skip level 0
        num_reps = len(representatives[level_idx])
        max_entropy = np.log(num_reps) if num_reps > 0 else 0
        max_entropies.append(max_entropy)
    
    ax.plot(
        x_indices,
        max_entropies,
        'k--',
        linewidth=2,
        label='Maximum Possible Entropy',
        alpha=0.7
    )
    
    # Plot method results
    for idx, method_name in enumerate(methods):
        entropies = results_dict[method_name]
        ax.plot(
            x_indices,
            entropies,
            marker=markers[idx % len(markers)],
            label=method_name,
            linewidth=2,
            markersize=8,
            color=colors[idx]
        )
    
    ax.set_xlabel("Level Set")
    ax.set_ylabel("Entropy")
    ax.set_title("Trajectory-Based Uniformity Analysis: Entropy by Level Set\n(Level 0 excluded - single representative)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Level {i}" for i in x_indices])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Entropy comparison plot saved: {os.path.basename(save_path)}")

def save_analysis_report(results_dict, representatives, results_folder):
    """Save detailed analysis report."""
    report_path = os.path.join(results_folder, "trajectory_uniformity_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("TRAJECTORY-BASED UNIFORMITY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ANALYSIS: How uniformly do trajectory states cover each level set?\n")
        f.write("Higher percentage = more uniform coverage within level set representatives\n")
        f.write("(Level 0 excluded - single representative at origin)\n\n")
        
        for method_name, entropies in results_dict.items():
            if method_name == "Maximum Entropy":
                continue
                
            f.write(f"{method_name}:\n")
            uniformity_percentages = []
            
            for entropy_idx, entropy in enumerate(entropies):
                level_idx = entropy_idx + 1  # Since we skip level 0
                num_reps = len(representatives[level_idx])
                max_entropy = np.log(num_reps) if num_reps > 0 else 0
                uniformity_pct = (entropy / max_entropy * 100) if max_entropy > 0 else 0
                uniformity_percentages.append(uniformity_pct)
                
                f.write(f"  Level {level_idx}: {uniformity_pct:.1f}% uniformity ")
                f.write(f"(entropy: {entropy:.4f}, max: {max_entropy:.4f}, reps: {num_reps})\n")
            
            avg_uniformity = np.mean(uniformity_percentages)
            f.write(f"  → Average: {avg_uniformity:.1f}%\n\n")
    
    print(f"Analysis report saved: {os.path.basename(report_path)}")

def main():
    # Create results folder
    results_folder = create_results_folder()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # File paths
    data_file_path = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset_supervised/saved_pickles/C_Uniform_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_seed_2025_grid_0.100_0.100_6.000deg_t1.41_ts0.2_vrange_2.0_2.0_steer_range_-30.0_30.0_steering_31.pkl"
    model_path = "/home/mikasa/RSN/traj_sampling/map_conditioning/baseline_models/baseline_openspace_model.pth"
    
    # Load data
    print("Loading level set representatives...")
    representatives, config = load_level_set_representatives(data_file_path)
    
    # Create grid for analysis
    thresholds = config['thresholds']
    grid = Grid(thresholds=thresholds)
    print(f"Grid thresholds: {thresholds}")
    
    # Set random seed for reproducibility
    np.random.seed(2025)
    torch.manual_seed(2025)
    
    # Number of trajectories to sample
    num_trajectories = 100000
    
    results_dict = {}
    
    # 1. Analyze with uniform actions
    print("\n" + "="*80)
    print("ANALYZING UNIFORM ACTIONS")
    print("="*80)
    
    uniform_trajectories = sample_trajectories_with_model(config, device, model=None, num_trajectories=num_trajectories)
    uniform_states = extract_all_states_from_trajectories(uniform_trajectories)
    print(f"Extracted {len(uniform_states)} states from uniform trajectories")
    
    uniform_entropies = compute_level_set_entropy(uniform_states, representatives, grid)
    results_dict["Uniform Actions"] = uniform_entropies
    
    # Save uniform trajectory visualization
    save_trajectory_visualization(uniform_trajectories, "Uniform Actions", results_folder)
    
    # 2. Analyze with baseline model (if available)
    try:
        print("\n" + "="*80)
        print("ANALYZING BASELINE MODEL")
        print("="*80)
        
        model = load_baseline_model(model_path, device)
        
        baseline_trajectories = sample_trajectories_with_model(config, device, model=model, num_trajectories=num_trajectories)
        baseline_states = extract_all_states_from_trajectories(baseline_trajectories)
        print(f"Extracted {len(baseline_states)} states from baseline trajectories")
        
        baseline_entropies = compute_level_set_entropy(baseline_states, representatives, grid)
        results_dict["Baseline Model"] = baseline_entropies
        
        # Save baseline trajectory visualization
        save_trajectory_visualization(baseline_trajectories, "Baseline Model", results_folder)
        
    except Exception as e:
        print(f"Error loading/analyzing baseline model: {e}")
        print("Continuing without baseline model analysis...")
    
    # 3. Print summary results
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    for method_name, entropies in results_dict.items():
        print(f"\n{method_name}:")
        uniformity_percentages = []
        
        for entropy_idx, entropy in enumerate(entropies):
            level_idx = entropy_idx + 1  # Since we skip level 0
            num_reps = len(representatives[level_idx])
            max_entropy = np.log(num_reps) if num_reps > 0 else 0
            uniformity_pct = (entropy / max_entropy * 100) if max_entropy > 0 else 0
            uniformity_percentages.append(uniformity_pct)
            print(f"  Level {level_idx}: {uniformity_pct:.1f}% uniformity (reps: {num_reps})")
        
        avg_uniformity = np.mean(uniformity_percentages)
        print(f"  → Average: {avg_uniformity:.1f}%")
    
    # 4. Create plots and save report
    plot_path = os.path.join(results_folder, "trajectory_entropy_comparison.png")
    plot_entropy_comparison(results_dict, representatives, plot_path)
    
    save_analysis_report(results_dict, representatives, results_folder)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to folder: {os.path.basename(results_folder)}")

if __name__ == "__main__":
    main() 