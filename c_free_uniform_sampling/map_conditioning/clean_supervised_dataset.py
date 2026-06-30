import os
import numpy as np
import pickle
import shutil
from scipy.spatial.distance import cosine
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import time

def load_sdf(env_path):
    """Load SDF from environment folder."""
    sdf_path = os.path.join(env_path, "sdf.npy")
    if os.path.exists(sdf_path):
        return np.load(sdf_path)
    return None

def check_valid_environment(env_path):
    """
    Check if environment is valid by verifying action_probs.pkl exists and is not empty.
    
    Args:
        env_path (str): Path to environment folder
        
    Returns:
        bool: True if environment is valid, False otherwise
    """
    action_probs_path = os.path.join(env_path, "action_probs.pkl")
    
    if not os.path.exists(action_probs_path):
        return False
    
    try:
        with open(action_probs_path, 'rb') as f:
            action_prob_data = pickle.load(f)
        
        # Check if level sets exist and are not empty
        if 'level_sets' not in action_prob_data:
            return False
            
        level_sets = action_prob_data['level_sets']
        if not level_sets or len(level_sets) == 0:
            return False
            
        # Check if at least one level set has valid data
        valid_level_sets = 0
        for level_set in level_sets:
            if level_set is not None and len(level_set) > 0:
                valid_level_sets += 1
        
        # Need at least 2 valid level sets for meaningful trajectory generation
        return valid_level_sets >= 2
        
    except Exception as e:
        print(f"    Error checking {os.path.basename(env_path)}: {str(e)}")
        return False

def compute_sdf_similarity(sdf1, sdf2, method='cosine'):
    """
    Compute similarity between two SDFs.
    
    Args:
        sdf1, sdf2 (np.ndarray): SDF arrays to compare
        method (str): Similarity method ('cosine', 'mse', 'structural')
        
    Returns:
        float: Similarity score (higher = more similar)
    """
    if sdf1.shape != sdf2.shape:
        return 0.0
    
    if method == 'cosine':
        # Flatten and compute cosine similarity
        flat1 = sdf1.flatten()
        flat2 = sdf2.flatten()
        # Cosine similarity returns values in [-1, 1], convert to [0, 1]
        similarity = cosine_similarity([flat1], [flat2])[0, 0]
        return (similarity + 1) / 2  # Convert to [0, 1] range
        
    elif method == 'mse':
        # Mean squared error (lower = more similar)
        mse = np.mean((sdf1 - sdf2) ** 2)
        # Convert to similarity score (higher = more similar)
        max_possible_mse = np.mean((np.max(sdf1) - np.min(sdf1)) ** 2)
        return 1.0 - (mse / max_possible_mse)
        
    elif method == 'structural':
        # Structural similarity based on obstacle patterns
        # Convert SDFs to binary obstacle maps
        obs1 = (sdf1 <= 0).astype(float)
        obs2 = (sdf2 <= 0).astype(float)
        
        # Compute Jaccard similarity for obstacle patterns
        intersection = np.sum(obs1 * obs2)
        union = np.sum((obs1 + obs2) > 0)
        
        if union == 0:
            return 1.0  # Both empty
        return intersection / union
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def analyze_dataset_similarity(input_dir, similarity_threshold=0.85, method='cosine'):
    """
    Analyze the dataset and identify similar consecutive environments.
    
    Args:
        input_dir (str): Input dataset directory
        similarity_threshold (float): Threshold for considering environments similar
        method (str): Similarity computation method
        
    Returns:
        tuple: (env_list, similarity_scores, similar_pairs)
    """
    print(" Analyzing dataset similarity...")
    
    # Get all environment directories
    env_dirs = sorted([d for d in os.listdir(input_dir) 
                      if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("env_")])
    
    if len(env_dirs) < 2:
        print("    Less than 2 environments found, no similarity analysis needed")
        return env_dirs, [], []
    
    print(f"   Found {len(env_dirs)} environments")
    
    # Load SDFs and compute consecutive similarities
    similarity_scores = []
    similar_pairs = []
    
    print(f"  🧮 Computing {method} similarities between consecutive environments...")
    
    for i in range(len(env_dirs) - 1):
        env1_path = os.path.join(input_dir, env_dirs[i])
        env2_path = os.path.join(input_dir, env_dirs[i + 1])
        
        sdf1 = load_sdf(env1_path)
        sdf2 = load_sdf(env2_path)
        
        if sdf1 is None or sdf2 is None:
            similarity = 0.0
            print(f"      Missing SDF for {env_dirs[i]} or {env_dirs[i+1]}")
        else:
            similarity = compute_sdf_similarity(sdf1, sdf2, method)
        
        similarity_scores.append(similarity)
        
        if similarity >= similarity_threshold:
            similar_pairs.append((i, i + 1, similarity))
            print(f"    🔗 Similar pair: {env_dirs[i]} ↔ {env_dirs[i+1]} (similarity: {similarity:.3f})")
    
    print(f"   Similarity analysis complete:")
    print(f"    • Mean similarity: {np.mean(similarity_scores):.3f}")
    print(f"    • Max similarity: {np.max(similarity_scores):.3f}")
    print(f"    • Min similarity: {np.min(similarity_scores):.3f}")
    print(f"    • Similar pairs (≥{similarity_threshold}): {len(similar_pairs)}")
    
    return env_dirs, similarity_scores, similar_pairs

def select_representative_environments(env_dirs, similar_pairs, selection_strategy='first'):
    """
    Select representative environments from similar groups.
    
    Args:
        env_dirs (list): List of environment directory names
        similar_pairs (list): List of (idx1, idx2, similarity) tuples
        selection_strategy (str): Strategy for selecting representatives ('first', 'random')
        
    Returns:
        set: Set of indices to keep
    """
    print(f" Selecting representative environments using '{selection_strategy}' strategy...")
    
    # Build groups of similar environments
    groups = []
    processed = set()
    
    for idx1, idx2, similarity in similar_pairs:
        if idx1 in processed and idx2 in processed:
            continue
            
        # Find existing group that contains either index
        found_group = None
        for group in groups:
            if idx1 in group or idx2 in group:
                found_group = group
                break
        
        if found_group is not None:
            found_group.update([idx1, idx2])
        else:
            groups.append({idx1, idx2})
        
        processed.update([idx1, idx2])
    
    # Add isolated environments
    for i in range(len(env_dirs)):
        if i not in processed:
            groups.append({i})
    
    print(f"   Found {len(groups)} groups:")
    
    # Select representatives from each group
    selected_indices = set()
    
    for i, group in enumerate(groups):
        group_list = sorted(list(group))
        
        if len(group_list) == 1:
            selected_idx = group_list[0]
            print(f"    Group {i+1}: {env_dirs[selected_idx]} (isolated)")
        else:
            if selection_strategy == 'first':
                selected_idx = group_list[0]
            elif selection_strategy == 'random':
                selected_idx = np.random.choice(group_list)
            else:
                raise ValueError(f"Unknown selection strategy: {selection_strategy}")
            
            group_names = [env_dirs[idx] for idx in group_list]
            print(f"    Group {i+1}: {group_names} → selected {env_dirs[selected_idx]}")
        
        selected_indices.add(selected_idx)
    
    print(f"  ✓ Selected {len(selected_indices)} representative environments")
    return selected_indices

def copy_environment(src_env_path, dst_env_path):
    """Copy an environment directory with all its files."""
    if os.path.exists(dst_env_path):
        shutil.rmtree(dst_env_path)
    
    shutil.copytree(src_env_path, dst_env_path)

def clean_supervised_dataset(
        input_dataset_dir,
        output_dataset_dir,
        similarity_threshold=0.85,
        similarity_method='cosine',
        selection_strategy='first',
        skip_existing=True
    ):
    """
    Clean the supervised dataset by filtering invalid environments and removing similar ones.
    
    Args:
        input_dataset_dir (str): Path to input dataset directory
        output_dataset_dir (str): Path to output cleaned dataset directory
        similarity_threshold (float): Threshold for considering environments similar
        similarity_method (str): Method for computing similarity ('cosine', 'mse', 'structural')
        selection_strategy (str): Strategy for selecting representatives ('first', 'random')
        skip_existing (bool): Skip processing if output directory exists
    """
    start_time = time.time()
    
    print("="*80)
    print("SUPERVISED DATASET CLEANING")
    print(f"Input:  {input_dataset_dir}")
    print(f"Output: {output_dataset_dir}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Similarity method: {similarity_method}")
    print(f"Selection strategy: {selection_strategy}")
    print("="*80)
    
    # Check input directory
    if not os.path.exists(input_dataset_dir):
        print(f" Input directory does not exist: {input_dataset_dir}")
        return
    
    # Create output directory
    if os.path.exists(output_dataset_dir):
        if skip_existing:
            print(f"⏭  Output directory already exists: {output_dataset_dir}")
            return
        else:
            print(f"🗑  Removing existing output directory: {output_dataset_dir}")
            shutil.rmtree(output_dataset_dir)
    
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    # Get all environment directories
    env_dirs = sorted([d for d in os.listdir(input_dataset_dir) 
                      if os.path.isdir(os.path.join(input_dataset_dir, d)) and d.startswith("env_")])
    
    if not env_dirs:
        print(f" No environment directories found in {input_dataset_dir}")
        return
    
    print(f" Found {len(env_dirs)} environments in input dataset")
    
    # Step 1: Filter out invalid environments
    print("\n Step 1: Filtering invalid environments...")
    valid_envs = []
    invalid_count = 0
    
    for env_name in env_dirs:
        env_path = os.path.join(input_dataset_dir, env_name)
        if check_valid_environment(env_path):
            valid_envs.append(env_name)
        else:
            invalid_count += 1
            print(f"   Invalid: {env_name}")
    
    print(f"  ✓ Valid environments: {len(valid_envs)}/{len(env_dirs)}")
    print(f"   Invalid environments: {invalid_count}")
    
    if len(valid_envs) == 0:
        print(" No valid environments found!")
        return
    
    # Step 2: Analyze similarity between consecutive valid environments
    print(f"\n Step 2: Analyzing similarity between consecutive environments...")
    valid_env_paths = [os.path.join(input_dataset_dir, env_name) for env_name in valid_envs]
    
    # Compute similarities only for valid environments
    similarity_scores = []
    similar_pairs = []
    
    if len(valid_envs) > 1:
        print(f"  🧮 Computing {similarity_method} similarities...")
        
        for i in range(len(valid_envs) - 1):
            env1_path = valid_env_paths[i]
            env2_path = valid_env_paths[i + 1]
            
            sdf1 = load_sdf(env1_path)
            sdf2 = load_sdf(env2_path)
            
            if sdf1 is None or sdf2 is None:
                similarity = 0.0
                print(f"      Missing SDF for {valid_envs[i]} or {valid_envs[i+1]}")
            else:
                similarity = compute_sdf_similarity(sdf1, sdf2, similarity_method)
            
            similarity_scores.append(similarity)
            
            if similarity >= similarity_threshold:
                similar_pairs.append((i, i + 1, similarity))
                print(f"    🔗 Similar pair: {valid_envs[i]} ↔ {valid_envs[i+1]} (similarity: {similarity:.3f})")
        
        print(f"   Similarity analysis complete:")
        print(f"    • Mean similarity: {np.mean(similarity_scores):.3f}")
        print(f"    • Max similarity: {np.max(similarity_scores):.3f}")
        print(f"    • Min similarity: {np.min(similarity_scores):.3f}")
        print(f"    • Similar pairs (≥{similarity_threshold}): {len(similar_pairs)}")
    else:
        print("    Only one valid environment, no similarity analysis needed")
    
    # Step 3: Select representative environments
    if len(similar_pairs) > 0:
        print(f"\n Step 3: Selecting representative environments...")
        selected_indices = select_representative_environments(valid_envs, similar_pairs, selection_strategy)
        selected_envs = [valid_envs[i] for i in sorted(selected_indices)]
    else:
        print(f"\n Step 3: No similar environments found, keeping all valid environments")
        selected_envs = valid_envs
    
    print(f"  ✓ Selected {len(selected_envs)} environments for cleaned dataset")
    
    # Step 4: Copy selected environments to output directory with continuous numbering
    print(f"\n📋 Step 4: Copying selected environments to cleaned dataset with continuous numbering...")
    
    # Create mapping from original names to new continuous names
    original_to_new_mapping = {}
    
    for i, original_env_name in enumerate(selected_envs):
        # Create new continuous environment name: env_001, env_002, env_003, etc.
        new_env_name = f"env_{i+1:03d}"
        original_to_new_mapping[original_env_name] = new_env_name
        
        src_path = os.path.join(input_dataset_dir, original_env_name)
        dst_path = os.path.join(output_dataset_dir, new_env_name)
        
        print(f"   Copying {original_env_name} → {new_env_name} ({i+1}/{len(selected_envs)})...")
        copy_environment(src_path, dst_path)
    
    # Verify continuous numbering
    final_env_dirs = sorted([d for d in os.listdir(output_dataset_dir) 
                            if os.path.isdir(os.path.join(output_dataset_dir, d)) and d.startswith("env_")])
    
    expected_names = [f"env_{i+1:03d}" for i in range(len(selected_envs))]
    
    if final_env_dirs == expected_names:
        print(f"   Successfully created {len(selected_envs)} environments with continuous numbering:")
        print(f"      Range: env_001 to env_{len(selected_envs):03d}")
    else:
        print(f"    Warning: Environment numbering may not be continuous")
        print(f"      Expected: {expected_names[:5]}...")
        print(f"      Found: {final_env_dirs[:5]}...")
    
    # Step 5: Generate summary report
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(" CLEANING SUMMARY")
    print(f"Original environments: {len(env_dirs)}")
    print(f"Valid environments: {len(valid_envs)}")
    print(f"Similar pairs found: {len(similar_pairs)}")
    print(f"Final cleaned environments: {len(selected_envs)}")
    print(f"Environment numbering: env_001 to env_{len(selected_envs):03d} (continuous)")
    print(f"Reduction: {len(env_dirs) - len(selected_envs)} environments ({((len(env_dirs) - len(selected_envs))/len(env_dirs)*100):.1f}%)")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f" Cleaned dataset saved to: {output_dataset_dir}")
    print("="*80)
    
    # Save cleaning report with mapping information
    report_path = os.path.join(output_dataset_dir, "cleaning_report.txt")
    with open(report_path, 'w') as f:
        f.write("SUPERVISED DATASET CLEANING REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Input dataset: {input_dataset_dir}\n")
        f.write(f"Output dataset: {output_dataset_dir}\n")
        f.write(f"Cleaning parameters:\n")
        f.write(f"  - Similarity threshold: {similarity_threshold}\n")
        f.write(f"  - Similarity method: {similarity_method}\n")
        f.write(f"  - Selection strategy: {selection_strategy}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Original environments: {len(env_dirs)}\n")
        f.write(f"  - Valid environments: {len(valid_envs)}\n")
        f.write(f"  - Similar pairs found: {len(similar_pairs)}\n")
        f.write(f"  - Final cleaned environments: {len(selected_envs)}\n")
        f.write(f"  - Environment numbering: env_001 to env_{len(selected_envs):03d} (continuous)\n")
        f.write(f"  - Reduction: {len(env_dirs) - len(selected_envs)} environments ({((len(env_dirs) - len(selected_envs))/len(env_dirs)*100):.1f}%)\n")
        f.write(f"  - Processing time: {total_time:.2f}s\n\n")
        
        f.write("Environment mapping (original → cleaned):\n")
        for original_name, new_name in original_to_new_mapping.items():
            f.write(f"  - {original_name} → {new_name}\n")
        
        if similar_pairs:
            f.write(f"\nSimilar pairs (threshold ≥{similarity_threshold}):\n")
            for idx1, idx2, similarity in similar_pairs:
                f.write(f"  - {valid_envs[idx1]} ↔ {valid_envs[idx2]} (similarity: {similarity:.3f})\n")
    
    print(f"📄 Cleaning report saved to: {report_path}")

def main():
    """Main function to clean the supervised dataset."""
    # Get the directory where this script is located (map_conditioning)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    # input_dataset_dir = os.path.join(script_dir, "dataset_supervised", "shepherd_dataset_supervised")
    # output_dataset_dir = os.path.join(script_dir, "dataset_supervised", "shepherd_dataset_supervised_cleaned")
    input_dataset_dir = os.path.join(script_dir, "dataset_supervised", "shepherd_dataset_supervised_heuristic")
    output_dataset_dir = os.path.join(script_dir, "dataset_supervised", "shepherd_dataset_supervised_heuristic_cleaned")
    
    # Run the cleaning process
    clean_supervised_dataset(
        input_dataset_dir=input_dataset_dir,
        output_dataset_dir=output_dataset_dir,
        similarity_threshold=0.98,   # More conservative threshold - environments with >95% similarity are considered similar
        similarity_method='cosine',  # Use cosine similarity
        selection_strategy='first',  # Select first environment from similar groups
        skip_existing=False  # Overwrite existing cleaned dataset
    )

if __name__ == "__main__":
    main() 