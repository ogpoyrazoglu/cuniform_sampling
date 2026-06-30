import os
import pickle
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from utility_helper_map import save_level_set_visualization

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class ActionProbDataset(Dataset):
    """Dataset for training on action probabilities with environment information."""
    def __init__(self, data_files: List[str], augmentation_params: Dict = None):
        # Assertions for input validation
        assert isinstance(data_files, list), f"data_files must be a list, got {type(data_files)}"
        assert len(data_files) > 0, "data_files cannot be empty"
        
        self.data = []
        self.env_data = []
        self.env_file_paths = []  # Store original file paths for boundary file lookup
        self.state_dim = 3  # x, y, theta
        
        # Store augmentation parameters for later use
        if augmentation_params is not None:
            assert isinstance(augmentation_params, dict), f"augmentation_params must be dict, got {type(augmentation_params)}"
            assert 'boundary_samples_per_state' in augmentation_params, "augmentation_params must contain 'boundary_samples_per_state'"
            self.boundary_samples_per_state = augmentation_params['boundary_samples_per_state']
        else:
            self.boundary_samples_per_state = 0
        
        # Note: env_idx is 0-indexed internally, but displayed as 1-indexed for human readability
        for env_idx, file_path in tqdm(enumerate(data_files), desc="Loading environment data", total=len(data_files)):
            
            assert os.path.exists(file_path), f"Data file does not exist: {file_path}"
            with open(file_path, 'rb') as f:
                env_data = pickle.load(f)
            
            # Assertions for environment data structure
            assert isinstance(env_data, dict), f"Environment data must be a dict, got {type(env_data)}"
            assert 'level_sets' in env_data, f"Environment data must contain 'level_sets' key"
            assert 'config' in env_data, f"Environment data must contain 'config' key"
            
            # Load corresponding SDF map
            env_dir = os.path.dirname(file_path)
            sdf_path = os.path.join(env_dir, "sdf.npy")
            costmap_path = os.path.join(env_dir, "costmap.npy")
            
            # Assertions for required files
            assert os.path.exists(sdf_path), f"SDF file not found at {sdf_path}"
            assert os.path.exists(costmap_path), f"Costmap file not found at {costmap_path}"
                
            sdf = np.load(sdf_path)
            costmap = np.load(costmap_path)
            
            # Assertions for SDF and costmap properties
            assert isinstance(sdf, np.ndarray), f"SDF must be numpy array, got {type(sdf)}"
            assert isinstance(costmap, np.ndarray), f"Costmap must be numpy array, got {type(costmap)}"
            assert sdf.ndim == 2, f"SDF must be 2D, got shape {sdf.shape}"
            assert costmap.ndim == 2, f"Costmap must be 2D, got shape {costmap.shape}"
            assert sdf.shape == costmap.shape, f"SDF and costmap must have same shape: {sdf.shape} vs {costmap.shape}"
            
            # Store environment information and file path
            self.env_data.append({
                'sdf': sdf,
                'costmap': costmap,
                'config': env_data['config'],
            })
            self.env_file_paths.append(file_path)
            
            # Process level sets and add environment index
            level_sets = env_data['level_sets']
            assert isinstance(level_sets, list), f"level_sets must be a list, got {type(level_sets)}"
            
            total_samples = 0
            for level_set_idx, level_set_data in enumerate(level_sets):
                if level_set_data is None or len(level_set_data) == 0:
                    continue
                
                # Assertions for level set data
                assert isinstance(level_set_data, np.ndarray), f"Level set data must be numpy array, got {type(level_set_data)}"
                assert level_set_data.ndim == 2, f"Level set data must be 2D, got shape {level_set_data.shape}"
                
                num_samples = level_set_data.shape[0]
                total_samples += num_samples
                for sample_idx in range(num_samples):
                    sample = level_set_data[sample_idx]
                    
                    assert len(sample) >= self.state_dim, f"Sample must have at least {self.state_dim} elements for state, got {len(sample)}"
                    state = sample[:self.state_dim]
                    action_probs = sample[self.state_dim:]
                    
                    # Assertions for state and action probabilities
                    assert len(state) == self.state_dim, f"State must have {self.state_dim} dimensions, got {len(state)}"
                    assert len(action_probs) > 0, f"Action probabilities cannot be empty"
                    assert np.all(action_probs >= 0), f"Action probabilities must be non-negative"
                    assert np.abs(np.sum(action_probs) - 1.0) < 1e-5, f"Action probabilities must sum to 1, got sum={np.sum(action_probs)}"
                    
                    # Store number of actions from first sample
                    if env_idx == 0 and level_set_idx == 0 and sample_idx == 0:
                        self.num_actions = len(action_probs)
                        tqdm.write(f"Detected {self.num_actions} actions from first sample")
                    else:
                        # Ensure all samples have same number of actions
                        assert len(action_probs) == self.num_actions, f"Inconsistent number of actions: expected {self.num_actions}, got {len(action_probs)}"
                    
                    # Add environment index to the sample
                    extended_sample = np.concatenate([
                        state,                   # state (3 dims)
                        action_probs,            # action probabilities (num_actions dims)
                        [env_idx]                # environment index (1 dim)
                    ])
                    self.data.append(extended_sample)
        assert len(self.data) > 0, "Dataset is empty after loading all files"
        
        self.data = np.array(self.data)
        print(f"Total dataset size: {len(self.data)} samples from {len(self.env_data)} environments")
        
        # Calculate and assert dimensions
        expected_sample_dim = self.state_dim + self.num_actions + 1  # +1 for env_idx
        assert self.data.shape[1] == expected_sample_dim, f"Sample dimension mismatch: expected {expected_sample_dim}, got {self.data.shape[1]}"
        print(f"State dimensions: {self.state_dim}, Action dimensions: {self.num_actions}")
    
    def apply_initial_level_set_augmentation(self, env_indices: List[int], initial_level_set_augmentation_samples: int):
        """Apply initial level set augmentation to specific environments only.
        
        This method generates noisy samples around the initial state [0,0,0] for each specified environment.
        
        Args:
            env_indices: List of environment indices (0-indexed) to augment
            initial_level_set_augmentation_samples: Number of noisy samples to generate around initial state
        """
        assert isinstance(env_indices, list), f"env_indices must be a list, got {type(env_indices)}"
        assert len(env_indices) > 0, f"env_indices cannot be empty"
        assert initial_level_set_augmentation_samples >= 0, f"initial_level_set_augmentation_samples must be non-negative"
        
        if initial_level_set_augmentation_samples == 0:
            print("No initial level set augmentation will be applied")
            return
        
        print(f"Applying initial level set augmentation with {initial_level_set_augmentation_samples} samples per environment")
        print(f"Applying to {len(env_indices)} environments: {[env_idx + 1 for env_idx in env_indices]} (1-indexed display)")
        
        all_augmented_samples = []
        original_size = len(self.data)
        
        # Apply augmentation to each specified environment
        for env_idx in tqdm(env_indices, desc="Applying initial level set augmentation"):
            assert 0 <= env_idx < len(self.env_data), f"Environment index {env_idx} out of range [0, {len(self.env_data)})"
            
            # Load environment data to get the initial level set
            original_data_file = self.env_file_paths[env_idx]
            with open(original_data_file, 'rb') as f:
                env_data = pickle.load(f)
            
            level_sets = env_data['level_sets']
            initial_level_data = level_sets[0]
            assert initial_level_data.shape[0] == 1, \
                f"Initial level set for env {env_idx+1} must have exactly one sample, but found {initial_level_data.shape[0]}."

            initial_sample = initial_level_data[0]
            initial_state = initial_sample[:self.state_dim]
            initial_action_probs = initial_sample[self.state_dim:]

            config = env_data['config']
            thresholds = np.array(config['thresholds'])[:self.state_dim]
            
            # Set variance using the 3-sigma rule (noise is within half resolution)
            variances = (thresholds / 6.0) ** 2
            
            rng = np.random.RandomState(2025 + env_idx)
            noise = rng.multivariate_normal(np.zeros(self.state_dim), np.diag(variances), size=initial_level_set_augmentation_samples)
            
            augmented_states = initial_state + noise
            # Handle theta wraparound
            augmented_states[:, 2] = (augmented_states[:, 2] + np.pi) % (2 * np.pi) - np.pi

            # Create and append the new augmented samples
            for aug_state in augmented_states:
                extended_sample = np.concatenate([
                    aug_state,
                    initial_action_probs,
                    [env_idx]
                ])
                all_augmented_samples.append(extended_sample)
        
        if all_augmented_samples:
            new_samples_array = np.array(all_augmented_samples)
            self.data = np.concatenate([self.data, new_samples_array], axis=0)
        
        new_size = len(self.data)
        added_samples = new_size - original_size
        print(f"Initial level set augmentation complete! Added {added_samples} samples")
        print(f"Total dataset size: {original_size} → {new_size} samples\n")
    
    def _add_boundary_augmentation(self, boundary_file: str, env_idx: int, config: dict):
        """Add augmented boundary states to the dataset by generating N noisy samples around each boundary state.
        Args: env_idx: Environment index (0-indexed)
        """
        assert os.path.exists(boundary_file), f"Boundary file does not exist: {boundary_file}"
        with open(boundary_file, 'rb') as f:
            boundary_states_with_probs = pickle.load(f)
        
        if len(boundary_states_with_probs) == 0:
            print(f"      No boundary states found")
            return
        
        # Convert to numpy array and validate structure
        boundary_array = np.array(boundary_states_with_probs)
        assert boundary_array.ndim == 2, f"Boundary array must be 2D, got shape {boundary_array.shape}"
        
        boundary_states = boundary_array[:, :self.state_dim]  # Extract states
        boundary_action_probs = boundary_array[:, self.state_dim:]  # Extract action probabilities
        
        # Assertions for boundary data
        assert boundary_states.shape[1] == self.state_dim, f"Boundary states must have {self.state_dim} dimensions"
        assert boundary_action_probs.shape[1] == self.num_actions, f"Boundary action probs must have {self.num_actions} dimensions"
        
        # Get thresholds for covariance calculation
        thresholds = np.array(config['thresholds'])
        assert len(thresholds) == 3, f"Expected 3D state space (x, y, theta), got {len(thresholds)} thresholds"
        
        # Set variance using the 3-sigma rule so ~99.7% of noise on each axis is within half the state resolution.
        cov_x = (thresholds[0] / 6) ** 2
        cov_y = (thresholds[1] / 6) ** 2
        cov_theta = (thresholds[2] / 6) ** 2
        
        # Generate noisy samples for all boundary states
        num_boundary_states = len(boundary_states)
        total_noise_samples = num_boundary_states * self.boundary_samples_per_state
        
        # Repeat boundary states for vectorized noise generation
        repeated_states = np.repeat(boundary_states, self.boundary_samples_per_state, axis=0)  # Shape: (total_noise_samples, 3)
        repeated_action_probs = np.repeat(boundary_action_probs, self.boundary_samples_per_state, axis=0)  # Shape: (total_noise_samples, num_actions)
        
        # Generate noise for all samples at once using deterministic random state
        rng = np.random.RandomState(2025+env_idx)  # Use deterministic random state
        noise_x = rng.normal(0, np.sqrt(cov_x), size=(total_noise_samples, 1))
        noise_y = rng.normal(0, np.sqrt(cov_y), size=(total_noise_samples, 1))
        noise_theta = rng.normal(0, np.sqrt(cov_theta), size=(total_noise_samples, 1))
        noise = np.concatenate([noise_x, noise_y, noise_theta], axis=1)  # Shape: (total_noise_samples, 3)
        
        noisy_states = repeated_states + noise # apply noise
        noisy_states[:, 2] = (noisy_states[:, 2] + np.pi) % (2*np.pi) - np.pi # handle theta wraparound
        
        # Add environment index to all synthesized samples
        final_augmented_samples = []
        for i in range(total_noise_samples):
            state = noisy_states[i]
            action_probs = repeated_action_probs[i]
            
            # Assertions for synthesized samples
            assert len(state) == self.state_dim, f"Synthesized state must have {self.state_dim} dimensions"
            assert len(action_probs) == self.num_actions, f"Synthesized action probs must have {self.num_actions} dimensions"
            assert np.all(action_probs >= 0), f"Synthesized action probabilities must be non-negative"
            assert np.abs(np.sum(action_probs) - 1.0) < 1e-5, f"Synthesized action probabilities must sum to 1"
            
            augmented_sample = np.concatenate([
                state,                      # state (3 dims)
                action_probs,               # action probabilities (num_actions dims) 
                [env_idx]                   # environment index (1 dim)
            ])
            final_augmented_samples.append(augmented_sample)
        
        # Add to main dataset
        # self.data.extend(final_augmented_samples)
        # print(f"      Added: {len(final_augmented_samples)} noisy samples\n")
        return final_augmented_samples

    def apply_boundary_augmentation_to_environments(self, 
            env_indices: List[int],
            boundary_samples_per_state: int,
            visualization_save_dir: Optional[str] = None):
        """Apply boundary augmentation to specific environments only.
        
        This method is designed to be called AFTER train/val/test split to apply augmentation
        only to training environments, keeping validation and test sets clean.
        
        Args:
            env_indices: List of environment indices (0-indexed) to augment
            boundary_samples_per_state: Number of noisy samples to generate per boundary state
            visualization_save_dir: Directory to save visualizations to, if None, no visualizations will be saved
        """
        assert isinstance(env_indices, list), f"env_indices must be a list, got {type(env_indices)}"
        assert len(env_indices) > 0, f"env_indices cannot be empty"
        self.boundary_samples_per_state = boundary_samples_per_state
        if self.boundary_samples_per_state <= 0:
            print("No boundary augmentation will be applied")
            print(f"Final dataset shape: {self.data.shape}")
            return
        else:
            print(f"Applying boundary augmentation with {self.boundary_samples_per_state} samples per boundary state")
        
        print(f"Applying boundary augmentation to {len(env_indices)} training environments...")
        print(f"Environments to augment: {[env_idx + 1 for env_idx in env_indices]} (1-indexed display)")

        all_newly_augmented_samples = []
        
        original_size = len(self.data)

        # visualization blocks
        if visualization_save_dir is not None:
            vis_dir = os.path.join(SCRIPT_DIR, visualization_save_dir)
            os.makedirs(vis_dir, exist_ok=True)
            print(f"Visualizations will be saved to: {vis_dir}")

        # Apply augmentation to each specified environment
        for env_idx in tqdm(env_indices, desc="Applying boundary augmentation"):
            assert 0 <= env_idx < len(self.env_data), f"Environment index {env_idx} out of range [0, {len(self.env_data)})"
            
            # Construct boundary file path assuming the boundary files are in the same structure as the original dataset
            boundary_file = self._find_boundary_file_for_env(env_idx)
            
            if boundary_file and os.path.exists(boundary_file):
                config = self.env_data[env_idx]['config']
                new_samples = self._add_boundary_augmentation(boundary_file, env_idx, config)
                if new_samples:
                    all_newly_augmented_samples.extend(new_samples)

                    if visualization_save_dir is not None:
                        # 1. Load original level sets for this environment to serve as the background
                        original_data_file = self.env_file_paths[env_idx]
                        with open(original_data_file, 'rb') as f:
                            original_env_data = pickle.load(f)
                        level_set_representatives = original_env_data['level_sets']
                        
                        # 2. Extract state data from the newly generated samples for plotting
                        augmented_states_for_vis = [sample[:self.state_dim] for sample in new_samples]

                        # 3. Define a unique filepath for this environment's plot
                        vis_filepath = os.path.join(vis_dir, f"augmented_env_{env_idx + 1}.png")
                        # 4. Call the visualization function for the current environment
                        save_level_set_visualization(
                            level_set_representatives=level_set_representatives,
                            filepath=vis_filepath,
                            config=config,
                            boundary_states=augmented_states_for_vis,
                        )
            else:
                print(f"    Warning: Boundary states file not found for environment {env_idx + 1}")

        if all_newly_augmented_samples:
            new_samples_array = np.array(all_newly_augmented_samples)
            self.data = np.concatenate([self.data, new_samples_array], axis=0)
        
        new_size = len(self.data)
        added_samples = new_size - original_size
        print(f"Boundary augmentation complete! Added {added_samples} samples to training environments")
        print(f"Total dataset size: {original_size} → {new_size} samples\n")

    
    def _find_boundary_file_for_env(self, env_idx: int) -> Optional[str]:
        """Helper method to find boundary file for a given environment index.
        Uses the stored original file paths to locate the boundary states file.
        """
        if env_idx >= len(self.env_file_paths):
            return None
        # Get the directory containing the original action_probs.pkl file
        original_file_path = self.env_file_paths[env_idx]
        env_dir = os.path.dirname(original_file_path)
        
        # Construct boundary file path
        boundary_file = os.path.join(env_dir, "boundary_states_with_probs.pkl")
        return boundary_file if os.path.exists(boundary_file) else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self.data), f"Index {idx} out of range [0, {len(self.data)})"
        sample = self.data[idx]
        state = sample[:self.state_dim]
        action_probs = sample[self.state_dim:self.state_dim + self.num_actions]
        env_idx = int(sample[self.state_dim + self.num_actions])
        
        assert len(state) == self.state_dim, f"State dimension mismatch"
        assert len(action_probs) == self.num_actions, f"Action probabilities dimension mismatch"
        assert 0 <= env_idx < len(self.env_data), f"Environment index {env_idx} out of range"
        
        return {
            'state': torch.tensor(state, dtype=torch.float32),
            'action_probs': torch.tensor(action_probs, dtype=torch.float32),
            'env_idx': env_idx
        }


def environment_level_split(dataset: ActionProbDataset, train_ratio: float = 0.8, 
                          val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[Subset, Subset, Subset, Dict[str, List[int]]]:
    """Split dataset at environment level with proper 80/10/10 split."""
    # Assertions for input validation
    assert isinstance(dataset, ActionProbDataset), f"dataset must be ActionProbDataset, got {type(dataset)}"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    assert train_ratio > 0 and val_ratio > 0 and test_ratio > 0, f"All ratios must be positive"
    
    num_envs = len(dataset.env_data)
    assert num_envs > 0, f"Dataset must have at least one environment"
    
    print(f"Splitting {num_envs} environments with ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    rng = np.random.RandomState(2025)
    
    # Handle single environment case with sample-level splitting
    if num_envs == 1:
        print("Single environment detected. Using sample-level splitting instead of environment-level splitting.")
        
        # Get all sample indices from the single environment
        all_indices = list(range(len(dataset)))
        
        # Shuffle indices for random splitting
        rng.shuffle(all_indices)
        
        # Calculate split sizes
        num_samples = len(all_indices)
        num_train_samples = int(num_samples * train_ratio)
        num_val_samples = int(num_samples * val_ratio)
        num_test_samples = num_samples - num_train_samples - num_val_samples
        
        # Split indices
        train_indices = all_indices[:num_train_samples]
        val_indices = all_indices[num_train_samples:num_train_samples + num_val_samples]
        test_indices = all_indices[num_train_samples + num_val_samples:]
        
        # Create datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        print(f"Sample distribution: {len(train_indices)} train samples, {len(val_indices)} val samples, {len(test_indices)} test samples")
        
        # All splits use the same environment (environment 0)
        env_assignments = {
            'train': [0],
            'val': [0], 
            'test': [0]
        }
        return train_dataset, val_dataset, test_dataset, env_assignments
    
    # For 2 environments, use overlapping assignment
    elif num_envs == 2:
        print(f"Warning: Only {num_envs} environments available. Using overlapping environment assignment.")
        train_envs = [0, 1]
        val_envs = [1]
        test_envs = [0]
    else:
        # Calculate split sizes
        num_train = max(1, int(num_envs * train_ratio))
        num_val = max(1, int(num_envs * val_ratio))
        num_test = max(1, num_envs - num_train - num_val)  # Ensure all environments are used
        
        # Adjust splits if they exceed total environments
        total_assigned = num_train + num_val + num_test
        if total_assigned > num_envs:
            # Ensure each split gets at least 1 environment for datasets with >= 3 environments
            # Prioritize: train gets most, then val and test get 1 each
            num_test = 1
            num_val = 1
            num_train = num_envs - num_val - num_test
        
        # Randomly assign environments to splits using isolated random state
        env_indices = list(range(num_envs))
        rng.shuffle(env_indices)
        
        train_envs = env_indices[:num_train]
        val_envs = env_indices[num_train:num_train + num_val]
        test_envs = env_indices[num_train + num_val:num_train + num_val + num_test]
        
        # Ensure no environment is assigned to multiple splits
        all_assigned = set(train_envs + val_envs + test_envs)
        assert len(all_assigned) == num_envs, f"Some environments not assigned or assigned multiple times"
        assert len(set(train_envs) & set(val_envs)) == 0, f"Train and val splits overlap"
        assert len(set(train_envs) & set(test_envs)) == 0, f"Train and test splits overlap"
        assert len(set(val_envs) & set(test_envs)) == 0, f"Val and test splits overlap"
    
    # Group samples by environment
    env_indices = {}
    for idx, sample in enumerate(dataset.data):
        env_idx = int(sample[dataset.state_dim + dataset.num_actions])
        if env_idx not in env_indices:
            env_indices[env_idx] = []
        env_indices[env_idx].append(idx)
    
    # Validate that all environments have samples
    for env_idx in range(num_envs):
        assert env_idx in env_indices, f"Environment {env_idx} has no samples"
        assert len(env_indices[env_idx]) > 0, f"Environment {env_idx} has empty sample list"
    
    # Create splits based on environment assignment
    train_indices = []
    val_indices = []
    test_indices = []
    
    for env_idx in train_envs:
        train_indices.extend(env_indices[env_idx])
    
    for env_idx in val_envs:
        val_indices.extend(env_indices[env_idx])
    
    for env_idx in test_envs:
        test_indices.extend(env_indices[env_idx])
    
    # Assertions for split quality
    assert len(train_indices) > 0, f"Training split is empty"
    assert len(val_indices) > 0, f"Validation split is empty"
    assert len(test_indices) > 0, f"Test split is empty"
    assert len(set(train_indices) & set(val_indices)) == 0, f"Train and val sample indices overlap"
    assert len(set(train_indices) & set(test_indices)) == 0, f"Train and test sample indices overlap"
    assert len(set(val_indices) & set(test_indices)) == 0, f"Val and test sample indices overlap"
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(dataset), f"Not all samples assigned to splits"
    
    print(f"Sample distribution: {len(train_indices)} train samples from {len(train_envs)} training envs, " +
          f"{len(val_indices)} val samples from {len(val_envs)} val envs, " +
          f"{len(test_indices)} test samples from {len(test_envs)} test envs")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create environment assignment dictionary
    env_assignments = {
        'train': sorted(train_envs),
        'val': sorted(val_envs),
        'test': sorted(test_envs)
    }
    return train_dataset, val_dataset, test_dataset, env_assignments


def get_dataset_files(dataset_name: str) -> List[str]:
    """Get all action_probs.pkl files from a dataset folder."""
    assert isinstance(dataset_name, str), f"dataset_name must be string, got {type(dataset_name)}"
    
    dataset_folder = os.path.join(SCRIPT_DIR, "dataset_supervised", dataset_name)
    assert os.path.exists(dataset_folder), f"Dataset folder not found: {dataset_folder}"
    
    # Find all environment directories
    env_dirs = glob.glob(os.path.join(dataset_folder, "env_*"))
    assert len(env_dirs) > 0, f"No environment directories found in {dataset_folder}"
    
    # Get action_probs.pkl files from each environment
    data_files = []
    for env_dir in sorted(env_dirs):  # Sort to ensure consistent ordering
        action_probs_file = os.path.join(env_dir, "action_probs.pkl")
        assert os.path.exists(action_probs_file), f"action_probs.pkl not found in {env_dir}"
        data_files.append(action_probs_file)
    
    print(f"Found {len(data_files)} environment files in {dataset_name}")
    return data_files


def find_latest_results_folder() -> str:
    """Find the latest results folder in the script directory."""
    results_folders = glob.glob(os.path.join(SCRIPT_DIR, "results_*"))
    
    if not results_folders:
        raise FileNotFoundError("No existing results folders found. Please run training first.")
    
    # Sort by modification time to get the latest
    results_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_folder = results_folders[0]
    
    print(f"Found latest results folder: {os.path.basename(latest_folder)}")
    return latest_folder


def custom_collate_fn(batch):
    """Custom collate function to handle the dictionary format."""
    assert len(batch) > 0, f"Batch cannot be empty"
    collated = {}
    for key in batch[0].keys():
        if key in ['state', 'action_probs']:
            tensors = [item[key] for item in batch]
            first_shape = tensors[0].shape
            for i, tensor in enumerate(tensors[1:], 1):
                assert tensor.shape == first_shape, f"Tensor shape mismatch at index {i}: expected {first_shape}, got {tensor.shape}"
            collated[key] = torch.stack(tensors)
        else:
            collated[key] = [item[key] for item in batch]
    return collated 


def compute_collision_metrics(trajectories: List, sdf: np.ndarray, resolution: float = 0.05) -> Tuple[float, float]:
    """
    Compute collision rate and collision severity for a list of trajectories using binary first-touch method.
    Includes line segment collision checking by interpolating points between trajectory states.
    
    Args:
        trajectories: List of trajectories, each a list of (state, action) tuples
        sdf: Signed distance function array (2D numpy array)
        resolution: Grid resolution in meters per cell
        
    Returns:
        Tuple of (collision_rate, collision_severity_score)
        - collision_rate: Percentage of trajectories that collide (0.0 to 100.0)
        - collision_severity_score: Average number of remaining timesteps per collided trajectory after first collision
    """
    assert isinstance(trajectories, list), f"trajectories must be list, got {type(trajectories)}"
    assert len(trajectories) > 0, f"trajectories list cannot be empty"
    assert isinstance(sdf, np.ndarray), f"sdf must be numpy array, got {type(sdf)}"
    assert sdf.ndim == 2, f"sdf must be 2D array, got shape {sdf.shape}"
    assert resolution > 0, f"resolution must be positive, got {resolution}"
    
    H, W = sdf.shape
    center_index_x = (W - 1) / 2.0
    center_index_y = (H - 1) / 2.0
    
    # Number of interpolation points between consecutive states (including endpoints)
    num_interp_points = 5
    
    # Collect all states from all trajectories with trajectory mapping
    all_states = []
    trajectory_start_indices = []
    trajectory_lengths = []
    
    for traj_idx, traj in enumerate(trajectories):
        assert isinstance(traj, list), f"Each trajectory must be a list"
        assert len(traj) > 0, f"Trajectory cannot be empty"
        
        trajectory_start_indices.append(len(all_states))
        
        # Extract states from trajectory
        traj_states = []
        for state, _ in traj:
            assert len(state) >= 2, f"State must have at least x,y coordinates, got {len(state)}"
            traj_states.append([state[0], state[1]])  # Extract x, y
        
        # Add interpolated points between consecutive states
        trajectory_all_points = []
        
        for i in range(len(traj_states)):
            # Always add the current state
            trajectory_all_points.append(traj_states[i])
            
            # If not the last state, add interpolated points to next state
            if i < len(traj_states) - 1:
                current_state = np.array(traj_states[i])
                next_state = np.array(traj_states[i + 1])
                
                # Create interpolation points between current and next state (excluding endpoints)
                for j in range(1, num_interp_points - 1):
                    alpha = j / (num_interp_points - 1)
                    interp_point = (1 - alpha) * current_state + alpha * next_state
                    trajectory_all_points.append(interp_point.tolist())
        
        all_states.extend(trajectory_all_points)
        trajectory_lengths.append(len(trajectory_all_points))
    
    # Convert to numpy array for vectorized operations
    all_states = np.array(all_states)  # Shape: (total_states, 2)
    
    # Vectorized coordinate conversion
    x_coords = all_states[:, 0]
    y_coords = all_states[:, 1]
    
    cols = center_index_x + x_coords / resolution
    rows = center_index_y - y_coords / resolution
    
    # Convert to grid indices with bounds checking
    grid_x = np.clip(np.round(cols).astype(int), 0, W - 1)
    grid_y = np.clip(np.round(rows).astype(int), 0, H - 1)
    
    # Sample SDF values for all states at once
    sdf_values = sdf[grid_y, grid_x]  # Shape: (total_states,)
    
    # Determine collisions (SDF <= 0)
    collisions = sdf_values <= 0.0
    
    # Binary first-touch counting: count remaining states after first collision
    collision_count = 0
    total_collision_score = 0.0
    
    for traj_idx in range(len(trajectories)):
        start_idx = trajectory_start_indices[traj_idx]
        traj_length = trajectory_lengths[traj_idx]
        end_idx = start_idx + traj_length
        
        # Check collisions for this trajectory (including interpolated points)
        traj_collisions = collisions[start_idx:end_idx]
        
        # Find first collision timestep
        collision_indices = np.where(traj_collisions)[0]
        
        if len(collision_indices) > 0:
            collision_count += 1
            
            # For collision severity, we need to map back to original trajectory timesteps
            # Since we interpolated, we need to convert interpolated index back to original trajectory index
            first_collision_interp_idx = collision_indices[0]
            
            # Calculate which original trajectory segment this collision belongs to
            # Each original segment has (num_interp_points - 1) interpolated points between states
            points_per_segment = num_interp_points - 1
            original_traj_len = len(trajectories[traj_idx])
            
            # Find which original segment the collision occurred in
            if first_collision_interp_idx == 0:
                # Collision at first state
                first_collision_timestep = 0
            else:
                # Find the segment: each segment contributes (num_interp_points - 1) points + 1 original point
                cumulative_points = 1  # First original point
                segment_idx = 0
                
                while segment_idx < original_traj_len - 1 and cumulative_points + points_per_segment < first_collision_interp_idx:
                    cumulative_points += points_per_segment
                    segment_idx += 1
                
                # Collision occurred in segment between segment_idx and segment_idx+1
                # For severity calculation, consider collision at segment_idx+1 (conservative)
                first_collision_timestep = min(segment_idx + 1, original_traj_len - 1)
            
            # Count remaining timesteps in original trajectory
            remaining_steps = original_traj_len - first_collision_timestep
            total_collision_score += remaining_steps
    
    # Calculate final metrics
    collision_rate = (collision_count / len(trajectories)) * 100.0
    
    # Calculate collision severity: average penetration depth of collided trajectories only
    if collision_count > 0:
        avg_collision_severity = total_collision_score / collision_count
    else:
        avg_collision_severity = 0.0
    
    # Assertions for output validation
    assert 0.0 <= collision_rate <= 100.0, f"Collision rate must be between 0-100%, got {collision_rate}"
    assert avg_collision_severity >= 0.0, f"Collision severity must be non-negative, got {avg_collision_severity}"
    
    return collision_rate, avg_collision_severity