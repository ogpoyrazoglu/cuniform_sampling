"""
BASELINE MODEL EVALUATION SCRIPT
================================

This script evaluates the baseline open space model by computing collision metrics
on the same test environments used for the map-conditioned model comparison.

Since the baseline model doesn't respond to different environments (open space only),
we use the pre-generated trajectories and test them against different environment costmaps.
"""

import os
import pickle
import numpy as np
import datetime
from dataset_utils import get_dataset_files, compute_collision_metrics

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_baseline_trajectories():
    """Load the pre-generated baseline trajectories."""
    traj_path = os.path.join(SCRIPT_DIR, "baseline_models", "baseline_trajectories.pkl")
    
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Baseline trajectories not found: {traj_path}")
    
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f" Loaded {len(trajectories)} baseline trajectories")
    return trajectories

def load_test_environments():
    """Load the test environment data for collision evaluation."""
    # Use the same dataset as the main training
    dataset_name = "shepherd_dataset_supervised_cleaned"
    data_files = get_dataset_files(dataset_name)
    
    # Load environment data (SDF and costmaps)
    env_data = []
    for i, data_file in enumerate(data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Load corresponding SDF and costmap files
        env_dir = os.path.dirname(data_file)
        sdf_path = os.path.join(env_dir, "sdf.npy")
        costmap_path = os.path.join(env_dir, "costmap.npy")
        
        if not os.path.exists(sdf_path):
            print(f"  SDF file not found for environment {i+1}: {sdf_path}")
            continue
        if not os.path.exists(costmap_path):
            print(f"  Costmap file not found for environment {i+1}: {costmap_path}")
            continue
            
        sdf = np.load(sdf_path)
        costmap = np.load(costmap_path)
        
        env_info = {
            'env_idx': i,  # 0-indexed
            'sdf': sdf,
            'costmap': costmap,
            'config': data['config']
        }
        env_data.append(env_info)
    
    print(f" Loaded {len(env_data)} environments from {dataset_name}")
    return env_data

def evaluate_baseline_on_environments(trajectories, env_data, test_env_indices):
    """
    Evaluate baseline trajectories against specific test environments.
    
    Args:
        trajectories: List of trajectory arrays from baseline model
        env_data: List of environment data dictionaries
        test_env_indices: List of environment indices to test (0-indexed)
    
    Returns:
        Dictionary mapping env_idx to (collision_rate, collision_severity) tuple
    """
    print(f" Evaluating baseline model on {len(test_env_indices)} test environments...")
    print(f" Test environments (1-indexed): {[env_idx + 1 for env_idx in test_env_indices]}")
    
    results = {}
    
    for env_idx in test_env_indices:
        if env_idx >= len(env_data):
            print(f"  Environment {env_idx + 1} (1-indexed) out of range, skipping...")
            continue
        
        print(f"🌍 Processing environment {env_idx + 1} (1-indexed display)...")
        
        # Get environment data
        env = env_data[env_idx]
        sdf = env['sdf']
        config = env['config']
        resolution = config.get('resolution', 0.05)
        
        # Compute collision metrics for this environment
        collision_rate, collision_severity = compute_collision_metrics(
            trajectories=trajectories,
            sdf=sdf,
            resolution=resolution
        )
        
        # Store results (using 0-indexed key)
        results[env_idx] = (collision_rate, collision_severity)
        
        print(f"     Environment {env_idx + 1}: Collision Rate: {collision_rate:.2f}%, Collision Severity: {collision_severity:.3f}")
    
    return results

def create_evaluation_report(results, test_env_indices, trajectories, output_path):
    """Create a detailed evaluation report similar to the map-conditioned model."""
    
    with open(output_path, 'w') as f:
        f.write("Baseline Model Test Evaluation Summary\n")
        f.write("=====================================\n")
        # Convert 0-indexed to 1-indexed for display
        env_display_numbers = [env_idx + 1 for env_idx in test_env_indices if env_idx in results]
        f.write(f"Environments (1-indexed): {env_display_numbers}\n")
        f.write(f"Number of Trajectories per Environment: {len(trajectories)}\n")
        f.write(f"Model Type: Baseline Open Space (No Map Conditioning)\n")
        f.write(f"\nDetailed Results:\n")
        f.write(f"{'Environment':<12} {'Collision Rate (%)':<18} {'Collision Severity':<18}\n")
        f.write(f"{'-'*50}\n")
        
        # Extract metrics for summary calculations
        collision_rates = []
        collision_severities = []
        
        for env_idx in sorted(results.keys()):  # env_idx is 0-indexed
            coll_rate, coll_severity = results[env_idx]
            collision_rates.append(coll_rate)
            collision_severities.append(coll_severity)
            # Display as 1-indexed in the table
            f.write(f"{env_idx + 1:<12} {coll_rate:<18.2f} {coll_severity:<18.3f}\n")
        
        # Write summary statistics
        f.write(f"{'-'*50}\n")
        f.write(f"{'Average':<12} {np.mean(collision_rates):<18.2f} {np.mean(collision_severities):<18.3f}\n")
        f.write(f"{'Std Dev':<12} {np.std(collision_rates):<18.2f} {np.std(collision_severities):<18.3f}\n")
        
        # Comparison note
        f.write(f"\nNotes:\n")
        f.write(f"- Baseline model trained on open space only (no map conditioning)\n")
        f.write(f"- Same trajectories tested against all environments\n")
        f.write(f"- Higher collision rates expected due to lack of obstacle awareness\n")
        f.write(f"\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    print(" BASELINE MODEL EVALUATION")
    print("=" * 40)
    
    # Load baseline trajectories
    trajectories = load_baseline_trajectories()
    
    # Load environment data
    env_data = load_test_environments()
    
    # Define the same test environments used in the map-conditioned evaluation
    # These are 0-indexed (subtract 1 from the 1-indexed display numbers)
    test_env_indices = [3, 10, 22, 33, 40, 62, 64, 67, 76, 94, 97, 101, 111, 117, 119]
    
    print(f" Using the same test environments as map-conditioned model:")
    print(f"   Original (1-indexed): [4, 11, 23, 34, 41, 63, 65, 68, 77, 95, 98, 102, 112, 118, 120]")
    print(f"   Internal (0-indexed): {test_env_indices}")
    
    # Evaluate baseline model
    results = evaluate_baseline_on_environments(trajectories, env_data, test_env_indices)
    
    # Create output directory
    output_dir = os.path.join(SCRIPT_DIR, "baseline_models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate evaluation report
    report_path = os.path.join(output_dir, "baseline_evaluation_report.txt")
    create_evaluation_report(results, test_env_indices, trajectories, report_path)
    
    print(f"\n📝 Evaluation report saved: {report_path}")
    
    # Print summary to console
    print(f"\n BASELINE MODEL SUMMARY:")
    print(f"=" * 40)
    
    if results:
        collision_rates = [results[env_idx][0] for env_idx in sorted(results.keys())]
        collision_severities = [results[env_idx][1] for env_idx in sorted(results.keys())]
        
        print(f"Average Collision Rate: {np.mean(collision_rates):.2f}% (±{np.std(collision_rates):.2f}%)")
        print(f"Average Collision Severity: {np.mean(collision_severities):.3f} (±{np.std(collision_severities):.3f})")
        print(f"Environments Tested: {len(results)}")
        print(f"Trajectories per Environment: {len(trajectories):,}")
        
        # Comparison with map-conditioned model
        print(f"\n COMPARISON WITH MAP-CONDITIONED MODEL:")
        print(f"Map-Conditioned Average Collision Rate: 21.06% (±10.41%)")
        print(f"Map-Conditioned Average Collision Severity: 0.566 (±0.278)")
        print(f"Baseline Average Collision Rate: {np.mean(collision_rates):.2f}% (±{np.std(collision_rates):.2f}%)")
        print(f"Baseline Average Collision Severity: {np.mean(collision_severities):.3f} (±{np.std(collision_severities):.3f})")
        
        # Calculate improvement
        map_cond_rate = 21.06
        baseline_rate = np.mean(collision_rates)
        rate_improvement = ((baseline_rate - map_cond_rate) / map_cond_rate) * 100
        
        map_cond_severity = 0.566
        baseline_severity = np.mean(collision_severities)
        severity_improvement = ((baseline_severity - map_cond_severity) / map_cond_severity) * 100
        
        print(f"\nRelative Performance:")
        if rate_improvement > 0:
            print(f" Collision Rate: {rate_improvement:+.1f}% worse than map-conditioned")
        else:
            print(f" Collision Rate: {abs(rate_improvement):.1f}% better than map-conditioned")
            
        if severity_improvement > 0:
            print(f" Collision Severity: {severity_improvement:+.1f}% worse than map-conditioned")
        else:
            print(f" Collision Severity: {abs(severity_improvement):.1f}% better than map-conditioned")
    
    print(f"\n Full report available at: {report_path}")
    print("=" * 40)

if __name__ == "__main__":
    main() 