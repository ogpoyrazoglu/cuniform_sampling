import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import copy
import pickle
import shutil
import random
import torch
import contextlib
from classes.grid import *
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import math
from pprint import pprint
from flow_Cuniform.dynamics_helpers import (
    cuda_dynamics_KS_3d_steering_angle_vectorized,
)
from utility_helper_map import (
    bilinear_sample_sdf_features,
    aggregate_environment_data,
    split_environments_data,
    test_and_save_trajectories,
    plot_loss_curves,
)
from model_parts import *
import logging
from collections import defaultdict
import time

# Performance logging utilities
class PerformanceLogger:
    def __init__(self):
        self.timers = defaultdict(float)
        self.counters = defaultdict(int)
        self.start_times = {}
        
    def start(self, name):
        """Start timing an operation"""
        self.start_times[name] = time.time()
        
    def end(self, name):
        """End timing an operation and record the elapsed time"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.timers[name] += elapsed
            self.counters[name] += 1
            del self.start_times[name]
            return elapsed
        return 0
    
    def log_operation(self, name, elapsed):
        """Log an operation that was timed externally"""
        self.timers[name] += elapsed
        self.counters[name] += 1
    
    def summary(self, prefix=""):
        """Return a summary of all timed operations"""
        indent = 2
        header_indent = " " * indent
        line_indent   = " " * (indent * 2)
        result = [f"{header_indent}{prefix} Performance Summary:"]
        total_time = sum(self.timers.values())
        
        # Sort operations by time spent (descending)
        sorted_ops = sorted(self.timers.items(), key=lambda x: x[1], reverse=True)
        
        for name, time_spent in sorted_ops:
            count = self.counters[name]
            avg_time = time_spent / count if count > 0 else 0
            percentage = (time_spent / total_time * 100) if total_time > 0 else 0
            result.append(f"{line_indent}{name}: {time_spent:.4f}s total, {avg_time:.4f}s avg ({count} calls) - {percentage:.2f}%")
        
        return "\n".join(result)
    
    def reset(self):
        """Reset all timers and counters"""
        self.timers.clear()
        self.counters.clear()
        self.start_times.clear()

# Global performance logger
perf_logger = PerformanceLogger()
torch.set_printoptions(sci_mode=False, precision=6)
np.set_printoptions(suppress=True, precision=6)

class UnifiedStateDataset(Dataset):
    def __init__(self, env_data_list, device, fixed_oversampled_level_sets):
        """
        Unified dataset with states, level set indices, and environment indices as columns.

        Args:
            env_data_list: List of environment data dictionaries.
            device: PyTorch device to store tensors.
        """
        # Pre-allocate lists to collect data
        all_samples = []

        for env_idx, env in enumerate(env_data_list):
            if fixed_oversampled_level_sets:
                with open("/home/mikasa/RSN/traj_sampling/map_conditioning/Kinematic_3D_trained_models/fixed_oversampled_level_sets.pkl", "rb") as f:
                    oversampled_level_sets = pickle.load(f)
                print(f"  --Fixed Level sets samples loaded for env {env_idx}--")
            else:
                oversampled_level_sets = env["oversampled_level_sets"]
                print(f"  --Environment specific Level sets samples loaded for nev {env_idx}--")
            for level_idx, level_set in enumerate(oversampled_level_sets):
                for state in level_set:
                    # Each sample is [state..., level_idx, env_idx]
                    sample = torch.tensor(list(state) + [float(level_idx), float(env_idx)], dtype=torch.float32)
                    all_samples.append(sample)

        # Stack into a single tensor: columns are [state_dim..., level_idx, env_idx]
        self.data = torch.stack(all_samples).to(device)  # Shape: (total_samples, state_dim + 2)
        self.state_dim = len(state)
        self.env_data_list = env_data_list
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        state = sample[:self.state_dim]  # First state_dim columns
        level_idx = sample[self.state_dim].long()  # Level set index
        env_idx = sample[self.state_dim + 1].long()  # Environment index
        return {
            "state": state,
            "level_set_idx": level_idx,
            "env_idx": env_idx,
            "sdf_map": torch.tensor(self.env_data_list[env_idx.item()]["sdf"], dtype=torch.float32).unsqueeze(0),
            # "map_embedding": self.env_data_list[env_idx.item()]["map_embedding"]
            # "costmap":  torch.tensor( self.env_data_list[env_idx.item()]["costmap"], dtype=torch.float32).unsqueeze(0)
        }

def unified_collate_fn(batch): 
    """ Custom collate function to combine a list of dictionaries into one dictionary. For keys with tensor values, we stack them; for others, we collect them in a list. """ 
    collated = {} # Assume all items have the same keys. 
    for key in batch[0]: # Try stacking; if shapes mismatch, return a list. 
        try: 
            collated[key] = torch.stack([b[key] for b in batch]) 
        except Exception: 
            collated[key] = [b[key] for b in batch] 
    return collated
    
def precompute_env_representatives(env_data_list, device):
    """
    Precompute representatives into a tensor indexed by env_idx and level_idx.

    Returns:
        torch.Tensor: Shape (num_envs, max_levels, num_reps, rep_dim)
    """
    # Assert all envs have same number of level sets
    level_counts = [len(env["representative_list"]) for env in env_data_list]
    assert all(count == level_counts[0] for count in level_counts), \
        f"Inconsistent number of level sets across environments: {level_counts}"

    num_envs = len(env_data_list)
    max_levels = level_counts[0] - 1  # Levels 1 to L
    max_reps = max(max(len(level) for level in env["representative_list"][1:]) for env in env_data_list)
    rep_dim = len(env_data_list[0]["representative_list"][1][0])  # e.g., 3 for [x, y, theta]

    # Initialize tensor with NaNs for padding
    reps_tensor = torch.full((num_envs, max_levels, max_reps, rep_dim), float('nan'), device=device)

    for e, env in enumerate(env_data_list):
        for l, reps in enumerate(env["representative_list"][1:], start=0):  # Skip level 0
            # shape (num_rep, state_dim)
            reps_stack = torch.stack([torch.as_tensor(rep, dtype=torch.float32, device=device) for rep in reps])
            reps_tensor[e, l, :len(reps)] = reps_stack
    return reps_tensor

def count_updation_vectorized(
    accumulators,                    # Shape: (num_envs, max_levels, max_reps)
    accumulators_uniform_action,     # Shape: (num_envs, max_levels, max_reps)
    pred_probs,                      # Shape: (batch_size, num_actions)
    states,                          # Shape: (batch_size, state_dim)
    env_indices,                     # Shape: (batch_size,)
    level_indices,                   # Shape: (batch_size,)
    reps_tensor,                     # Shape: (num_envs, max_levels, max_reps, rep_dim)
    actions,                         # Shape: (num_actions,)
    dt,                              # Scalar: time step
    v,                               # Scalar: velocity
    config,                          # Configuration dictionary
    device                           # PyTorch device
):
    """
    Vectorized version of count_updation for multiple environments and level sets.
    Updates accumulators using exponential distance decay and computes linear obstacle loss.
    Includes state propagation logic internally.
    """
    batch_size, state_dim = states.shape
    num_actions = actions.shape[0]
    num_envs, max_levels, max_reps, rep_dim = reps_tensor.shape

    # Compute next states for all actions
    next_states = cuda_dynamics_KS_3d_steering_angle_vectorized(
        states, actions, dt, v
    )  # Shape: (batch_size*num_actions, state_dim)
    next_states_expanded = torch.hstack((
        next_states[:, 0:2],                        # x, y components
        torch.sin(next_states[:, 2]).unsqueeze(1),  # sin(theta)
        torch.cos(next_states[:, 2]).unsqueeze(1)   # cos(theta)
    ))  # Shape: (batch_size * num_actions, 4)

    # Get XY components for obstacle loss
    xy_next_states = next_states[:, :2] # Shape: (batch_size * num_actions, 2)

    # Threshold for obstacle loss
    T = (torch.sqrt(torch.tensor(config["thresholds"][0]**2) + torch.tensor(config["thresholds"][1]**2))/1.0).float().to(device)

    # Decay factor for accumulator updates
    decay_factor = -50.0  # Scalar

    # Uniform action probability
    uniform_action_prob = 1.0 / num_actions  # Scalar
    uniform_probs = torch.full(
        (batch_size, num_actions), uniform_action_prob, device=device, dtype=pred_probs.dtype
    )  # Shape: (batch_size, num_actions)

    # Initialize sums for obstacle losses
    total_obs_sum = 0.0  # Scalar
    total_uniform_obs_sum = 0.0  # Scalar

    # Get unique (env_idx, level_idx) pairs in the batch
    pairs = torch.stack([env_indices, level_indices], dim=1)  # Shape: (batch_size, 2)
    unique_pairs, inverse_indices = torch.unique(pairs, dim=0, return_inverse=True)  # unique_pairs: (num_unique, 2), inverse_indices: (batch_size,)

    perf_logger.start("cu_pair_processing")
    for pair_idx, (e_idx, l_idx) in enumerate(unique_pairs):
        # Samples in this group
        pair_start = time.time()
        mask = inverse_indices == pair_idx  # Shape: (batch_size,)
        group_size = mask.sum()  # Scalar
        if group_size == 0:
            continue
        expanded_mask = mask.repeat_interleave(num_actions)  # Shape: (batch_size * num_actions,)

        # Extract group data
        group_states_flat = next_states_expanded[expanded_mask]  # Shape: (group_size * num_actions, state_dim)
        group_xy_states_flat = xy_next_states[expanded_mask]     # Shape: (group_size * num_actions, 2)
        group_probs = pred_probs[mask]                           # Shape: (group_size, num_actions)
        group_probs_flat = group_probs.reshape(-1)               # Shape: (group_size * num_actions,)
        group_uniform_probs = uniform_probs[mask]                # Shape: (group_size, num_actions)
        group_uniform_probs_flat = group_uniform_probs.reshape(-1)

        # Get representatives for this environment and next level
        next_l_idx = l_idx  # Scalar, no +1 because initial level set representatives is skipped
        reps = reps_tensor[e_idx, next_l_idx]      # Shape: (max_reps, rep_dim)
        reps_expanded = torch.hstack((
            reps[:, 0:2],
            torch.sin(reps[:, 2]).unsqueeze(1),
            torch.cos(reps[:, 2]).unsqueeze(1)
        )) # shape num_representatives x 4
        xy_reps = reps_expanded[:, :2]                      # Shape: (max_reps, 2)
        valid_reps_mask = ~torch.isnan(reps_expanded[:, 0]) # Shape: (max_reps,)
        valid_reps = reps_expanded[valid_reps_mask]         # Shape: (num_valid_reps, rep_dim)
        valid_xy_reps = xy_reps[valid_reps_mask]   # Shape: (num_valid_reps, 2)
        assert len(valid_reps) > 0, "no valid_reps??"

        # Compute distances with flattened group states # Shape: (group_size * num_actions, num_valid_reps)
        distance_matrix = torch.cdist(group_states_flat, valid_reps, p=2) 
        xy_distance_matrix = torch.cdist(group_xy_states_flat, valid_xy_reps, p=2)

        # Obstacle loss with linear distance
        min_xy_distances = xy_distance_matrix.min(dim=1)[0]  # Shape: (group_size * num_actions,)
        unsafe_mask = torch.clamp(min_xy_distances - T, min=0)  # Shape: (group_size * num_actions,)
        
        # Compute safe_mask to exclude collided propagated states
        # safe_mask = (min_xy_distances <= T).to(group_probs_flat.dtype)

        decayed_distance = torch.exp(decay_factor * distance_matrix)  

        # contrib = ((group_probs_flat * safe_mask).unsqueeze(-1) * decayed_distance).sum(dim=0)  # Shape: (num_valid_reps,)
        # contrib_uniform = ((group_uniform_probs_flat * safe_mask).unsqueeze(-1) * decayed_distance).sum(dim=0)
        contrib = ((group_probs_flat).unsqueeze(-1) * decayed_distance).sum(dim=0)  # Shape: (num_valid_reps,)
        contrib_uniform = ((group_uniform_probs_flat).unsqueeze(-1) * decayed_distance).sum(dim=0)

        # Update accumulators, shape: (num_valid_reps,)
        perf_logger.start("cu_accumulator_update")
        accumulators[e_idx, next_l_idx, :len(valid_reps)] += contrib 
        accumulators_uniform_action[e_idx, next_l_idx, :len(valid_reps)] += contrib_uniform  # Shape: (num_valid_reps,)

        total_obs_sum += (unsafe_mask * group_probs_flat).sum()  # Scalar
        total_uniform_obs_sum += (unsafe_mask * group_uniform_probs_flat).sum()  # Scalar
        perf_logger.end("cu_accumulator_update")
        
        # # Log time for this pair
        # pair_time = time.time() - pair_start
        # perf_logger.log_operation(f"cu_pair_{e_idx}_{l_idx}", pair_time)
        
        ''' TODO: 
        normalize the obs loss here so each enviornment contribute roughly the same,
        also insepct the kl loss difference across level sets across enviornments. (should I do this normalization?)
        '''
    perf_logger.end("cu_pair_processing")
    return accumulators, accumulators_uniform_action, total_obs_sum, total_uniform_obs_sum

def compute_kl_divergence(acc, reps_slice, device):
    # Assumes padded reps in `acc` are zero so that normalizing full `acc` (before masking) 
    # yields the same valid‐only distribution as normalizing after masking
    P = acc / acc.sum() # normalize to get probability distribution P
    valid_mask = ~torch.isnan(reps_slice) # get valid representatives (non-NaN)
    num_valid = valid_mask.sum().item()
    assert num_valid > 0, "No valid representative in kl calculation? What's wrong?"

    # Extract only the contributions corresponding to valid representatives.
    p_valid = P[valid_mask]

    # Define the target uniform distribution over valid representatives.
    q_valid = torch.full(p_valid.shape, 1.0 / num_valid, device=device, dtype=P.dtype)

    # Compute the KL divergence: sum_i p_valid[i] * log(p_valid[i] / q_valid[i])
    kl_loss = torch.sum(p_valid * torch.log((p_valid + 1e-8) / (q_valid + 1e-8)))
    return kl_loss

def initialize_weighted_accumulators(reps_tensor):
    """
    Initializes accumulators using a weighted uniform distribution,
    where each representative starts with a count equal to the number
    of representatives in its respective level set divide by some constant(environment-specific).

    Args:
        reps_tensor (torch.Tensor): Tensor of shape (num_envs, max_levels, max_reps, rep_dim)
            containing representatives, with NaNs as padding for invalid entries.

    Returns:
        torch.Tensor: Initialized accumulators of shape (num_envs, max_levels, max_reps)
    """
    # Mask indicating valid (non-NaN) representatives
    valid_mask = ~torch.isnan(reps_tensor[..., 0])  # Shape: (num_envs, max_levels, max_reps)

    # Count valid representatives for each environment-level combination
    reps_count = valid_mask.sum(dim=2, keepdim=True).float()  # Shape: (num_envs, max_levels, 1)
    # reps_count /= 1.0
    reps_count = 1.0

    # Set each valid representative's initial value to the number of representatives in its level set
    weighted_uniform_acc = valid_mask.float() * reps_count  # Broadcasting along max_reps dimension
    return weighted_uniform_acc

def process_one_epoch(
        loader, train_mode, reps_tensor,
        lambda_obs, lambda_entropy,
        model, feature_extractor, device, optimizer,
        actions, dt, v, config,                          
        epoch, num_epochs,
    ):
    logging.info("")
    total_samples = 0
    epoch_start_time = time.time()

    # Initialize accumulators, Shape: (num_envs, max_levels, max_reps)
    accumulators = initialize_weighted_accumulators(reps_tensor)
    accumulators_uniform_action = initialize_weighted_accumulators(reps_tensor)
    num_envs, max_levels, max_reps, rep_dim = reps_tensor.shape

    total_loss = 0.0
    total_kl_loss = 0.0
    total_obs_loss = 0.0
    total_uniform_kl_loss = 0.0
    total_uniform_obs_loss = 0.0

    batch_logs = []  # List to collect per-batch loss logs
    
    # Track the total batch processing time
    total_batch_time = 0.0
    
    for batch_idx, batch in enumerate(loader):
        batch_start_time = time.time()
        
        states_batch = batch["state"]           # Shape: (batch_size, state_dim)
        level_indices = batch["level_set_idx"]  # Shape: (batch_size,)
        env_indices = batch["env_idx"]          # Shape: (batch_size,)
        # map_embeddings = batch["map_embedding"] # Shape: (batch_size, embedding_dim)
        batch_size = states_batch.shape[0]
        total_samples += batch_size

        # Predict action probabilities
        net_input = torch.hstack([
            states_batch[:, 0:2],
            torch.sin(states_batch[:, 2]).unsqueeze(1),
            torch.cos(states_batch[:, 2]).unsqueeze(1)
        ]).to(device) # [batch_size_of_states, state_dim]

        # Instead of processing a duplicate SDF per state sample, process each unique environment once. #NOTE: feature extraction take <1% of training time.
        unique_envs = torch.unique(env_indices).tolist()
        dense_features_dict = {}
        for env in unique_envs:
            if feature_extractor is None:
                continue
            # Find all indices where env_indices equals the current environment 'env'
            matching_indices = torch.nonzero(env_indices == env, as_tuple=False) # shape [num_matches, 1]
            first_idx = matching_indices[0].item() # get first matching index
            sdf_tensor_env = batch["sdf_map"][first_idx].unsqueeze(0).to(device)  # shape: [1, 1, H, W]
            # costmap_env = batch["costmap"][first_idx].unsqueeze(0).to(device) # shape: [1, 1, H, W]

            # Precompute dense features for this environment
            # dense_features_dict[int(env)] = feature_extractor(costmap_env)
            dense_features_dict[int(env)] = feature_extractor(sdf_tensor_env)  # shape: [1, feature_dim, H, W]

        # for each sample in the batch, directly sample from the corresponding dense feature.
        # Group states by environment for batched processing
        if feature_extractor is None:
            sampled_features = None
        else:
            sampled_features = torch.zeros((net_input.shape[0], dense_features_dict[unique_envs[0]].shape[1]), device=device)
        
        # Process each environment's states in a single batch
        for env in unique_envs:
            if feature_extractor is None:
                continue
            # Find all indices where env_indices equals the current environment
            env_mask = env_indices == env
            env_indices_where = torch.nonzero(env_mask, as_tuple=False).squeeze(1)
            
            if env_indices_where.numel() == 0:
                continue
                
            # Get all states for this environment
            env_states = net_input[env_mask]
            
            # Get the dense feature map for this environment
            single_env_dense_feature = dense_features_dict[int(env)]
            
            # Process all states from this environment in one batch operation
            env_features = bilinear_sample_sdf_features(
                sdf_features=single_env_dense_feature,
                states=env_states,  # All states for this environment, not just one
                resolution=0.05
            )
            
            # Place the results back in the correct positions of the output tensor
            sampled_features[env_indices_where] = env_features
            
        interpolated_features = sampled_features  # Already in the right shape: [batch_size, feature_dim]

        if feature_extractor is None:
            pred_probs = model(net_input) #NOTE: this is the openspace baseline
        else:
            pred_probs = model(net_input, interpolated_features)  # Shape: (batch_size, num_actions)

        # Update accumulators and compute losses using vectorized function
        perf_logger.start("count_updation")
        accumulators, accumulators_uniform_action, batch_obs_sum, batch_uniform_obs_sum = count_updation_vectorized(
            accumulators,                    # Shape: (num_envs, max_levels, max_reps)
            accumulators_uniform_action,     # Shape: (num_envs, max_levels, max_reps)
            pred_probs,                      # Shape: (batch_size, num_actions)
            states_batch,                    # Shape: (batch_size, state_dim)
            env_indices,                     # Shape: (batch_size,)
            level_indices,                   # Shape: (batch_size,)
            reps_tensor,                     # Shape: (num_envs, max_levels, max_reps, rep_dim)
            actions,                         # Shape: (num_actions,)
            dt,                              # Scalar
            v,                               # Scalar
            config,                          # Config dict
            device
        )
        perf_logger.end("count_updation")
        
        # get KL divergence for model predictions
        # Compute KL divergence for model predictions (aggregated over environments and levels)
        batch_total_kl_loss = 0.0 
        for e in range(num_envs):
            for l in range(max_levels):
                batch_total_kl_loss += compute_kl_divergence( # accumulator do not use l+1 because initial level set is skipped
                    accumulators[e, l], reps_tensor[e, l, :, 0], device
                )

        # get KL divergence for uniform action predictions
        batch_total_uniform_kl_loss = 0.0
        for e in range(num_envs):
            for l in range(max_levels):
                batch_total_uniform_kl_loss += compute_kl_divergence(
                    accumulators_uniform_action[e, l], reps_tensor[e, l, :, 0], device
                )

        # Combine losses and calculate lambda_obs dynamically
        batch_scaled_normalized_kl = lambda_entropy * batch_total_kl_loss / (num_envs * max_levels)
        batch_scaled_normalized_kl_uniform = lambda_entropy * batch_total_uniform_kl_loss / (num_envs * max_levels)

        batch_scaled_obs = lambda_obs * batch_obs_sum
        batch_scaled_obs_uniform = lambda_obs * batch_uniform_obs_sum

        '''
        # NOTE: We do NOT divide the KL loss by the batch size because the divergence is computed
        # on a probability distribution (accumulated over the entire batch), so it is already an normalized measure. 
        # In contrast, the obstacle loss is summed over samples and needs to be normalized by batch size.
        '''
        batch_normalized_loss = batch_scaled_normalized_kl + (batch_scaled_obs / batch_size)

        if train_mode:
            perf_logger.start("backward_pass")
            optimizer.zero_grad()
            batch_normalized_loss.backward()
            optimizer.step()
            perf_logger.end("backward_pass")

            # Detach accumulators to prevent memory buildup
            accumulators = accumulators.detach()  # Shape: (num_envs, max_levels, max_reps)
            accumulators_uniform_action = accumulators_uniform_action.detach()

        # Accumulate losses
        total_loss += batch_normalized_loss.item()
        total_kl_loss += batch_scaled_normalized_kl.item()
        total_obs_loss += batch_scaled_obs.item()
        total_uniform_kl_loss += batch_scaled_normalized_kl_uniform.item()
        total_uniform_obs_loss += batch_scaled_obs_uniform.item()

        # Calculate batch processing time
        batch_time = time.time() - batch_start_time
        total_batch_time += batch_time

        # store the per-batch per sample loss log in a list
        batch_logs.append(
            f"    Batch {batch_idx+1:03d} | Time: {batch_time:.2f}s | KL Loss: {batch_scaled_normalized_kl.item():.5f} | "
            f"Uniform KL Loss: {batch_scaled_normalized_kl_uniform.item():.5f} | "
            f"Obs Loss: {(batch_scaled_obs.item() / batch_size):.5f} | "
            f"Uniform Obs Loss: {(batch_scaled_obs_uniform.item() / batch_size):.5f}"
        )

    # Average losses for each sample
    avg_loss = total_loss / len(loader) 
    avg_per_batch_kl_loss = total_kl_loss / len(loader)
    avg_obs_loss = total_obs_loss / total_samples
    avg_per_batch_kl_loss_uniform = total_uniform_kl_loss / len(loader)
    avg_obs_loss_uniform = total_uniform_obs_loss / total_samples
    
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = total_batch_time / len(loader) if len(loader) > 0 else 0

    print(
        f"Epoch {epoch+1}/{num_epochs} | {'Train' if train_mode else 'Val'} | Total Time: {epoch_time:.2f}s | Avg Batch Time: {avg_batch_time:.2f}s | Total Loss: {avg_loss:.5f}\n"
        f"  KL Loss: {avg_per_batch_kl_loss:.5f} | "
        f"Uniform KL Loss: {avg_per_batch_kl_loss_uniform:.5f} | "
        f"Obs Loss: {avg_obs_loss:.5f} | "
        f"Uniform Obs Loss: {avg_obs_loss_uniform:.5f}"
    )
    logging.info(
        f"Epoch {epoch+1}/{num_epochs} | {'Train' if train_mode else 'Val'} | Total Time: {epoch_time:.2f}s | Avg Batch Time: {avg_batch_time:.2f}s | Total Loss: {avg_loss:.5f}\n"
        f"  KL Loss: {avg_per_batch_kl_loss:.5f} | "
        f"Uniform KL Loss: {avg_per_batch_kl_loss_uniform:.5f} | "
        f"Obs Loss: {avg_obs_loss:.5f} | "
        f"Uniform Obs Loss: {avg_obs_loss_uniform:.5f}"
    )
    
    # Log performance breakdown
    perf_summary = perf_logger.summary(f"Epoch {epoch+1}")
    print(perf_summary)
    logging.info(perf_summary)
    
    for log_line in batch_logs:
        logging.info(log_line)
    
    # Reset performance logger for the next epoch
    perf_logger.reset()

    return (
        avg_per_batch_kl_loss, 
        avg_obs_loss, 
        avg_per_batch_kl_loss_uniform, 
        avg_obs_loss_uniform, 
        avg_loss
    )

def train_model_standardized(
        train_dataset, val_dataset, model, feature_extractor, experiment_folder, device, config, is_single_env
    ):
    """
    Train the model using a unified dataset with vectorized loss computation.

    Args:
        train_dataset: UnifiedStateDataset for training.
        val_dataset: UnifiedStateDataset for validation.
        model: Neural network model.
        experiment_folder: Directory to save results.
        device: PyTorch device.
        config: Configuration dictionary.
        is_single_env: true if training and test on single env

    Returns:
        tuple: Lists of training and validation loss histories:
            (train_entropy_loss_list, train_obstacle_loss_list,
             train_uniform_entropy_loss_list, train_uniform_obstacle_loss_list,
             val_entropy_loss_list, val_obstacle_loss_list,
             val_uniform_entropy_loss_list, val_uniform_obstacle_loss_list)
    """
    start_time = time.time()
    logging.basicConfig(
        filename="most_recent_trial_logging",
        level=logging.INFO,
        format="%(message)s",
        filemode="w"  # overwrite the file each time the script runs
    )
    
    # Create a separate logger for performance metrics
    perf_log_file = os.path.join(experiment_folder, "performance_metrics.log")
    perf_file_handler = logging.FileHandler(perf_log_file, mode="w")
    perf_file_handler.setLevel(logging.INFO)
    perf_file_handler.setFormatter(logging.Formatter("%(message)s"))
    perf_logger_file = logging.getLogger("performance")
    perf_logger_file.setLevel(logging.INFO)
    perf_logger_file.addHandler(perf_file_handler)
    perf_logger_file.propagate = False
    
    # Log system info
    perf_logger_file.info(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    perf_logger_file.info(f"Device: {device}")
    if device == 'cuda':
        perf_logger_file.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        perf_logger_file.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize optimizer
    lr = 3e-5
    if feature_extractor is not None:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(feature_extractor.parameters()),
            lr=lr
        )
    else:
        optimizer = torch.optim.Adam(
            list(model.parameters()),
            lr=lr
        )
    
    best_val_loss = float('inf')
    num_epochs = 15
    lambda_entropy = 100.0 
    lambda_obs = 1.0
    if feature_extractor is None:
        lambda_obs = 0.0
        print("Using openspace baseline, no obstacle loss")

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=unified_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True, collate_fn=unified_collate_fn)

    # Precompute representatives, Shape: (num_envs, max_levels, max_reps, rep_dim)
    train_reps_tensor = precompute_env_representatives(train_dataset.env_data_list, device)  
    val_reps_tensor = precompute_env_representatives(val_dataset.env_data_list, device)
    print("train_reps_tensor.shape: ", train_reps_tensor.shape)
    print("val_reps_tensor.shape: ", val_reps_tensor.shape)

    # Extract actions, dt, and v from config
    actions = torch.tensor(config["actions"][:, 0], device=device)  # Shape: (num_actions,)
    dt = config["dt"]  # Scalar
    v = config["vrange"][0]  # Scalar

    train_entropy_loss_list = []
    train_obs_loss_list = []
    train_uniform_entropy_loss_list = []
    train_uniform_obs_loss_list = []

    val_entropy_loss_list = []
    val_obs_loss_list = []
    val_uniform_entropy_loss_list = []
    val_uniform_obs_loss_list =[]
    
    total_epoch_time = 0.0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        perf_logger.start("train_epoch")
        model.train()
        if feature_extractor is not None:
            feature_extractor.train()
        (
            avg_train_entropy_loss, avg_train_obs_loss, 
            avg_train_uniform_entropy_loss, avg_train_uniform_obs_loss,
            avg_train_loss,
        ) = process_one_epoch(
            #only first line 3 variables differed across trainning dataset vs validation dataset
            loader=train_loader, train_mode=True, reps_tensor=train_reps_tensor, 
            model=model, feature_extractor=feature_extractor, device=device, optimizer=optimizer,
            lambda_obs=lambda_obs, lambda_entropy=lambda_entropy,
            actions=actions, dt=dt, v=v, config=config,     
            epoch=epoch, num_epochs=num_epochs,
        )
        perf_logger.end("train_epoch")
        
        train_entropy_loss_list.append(avg_train_entropy_loss)
        train_obs_loss_list.append(avg_train_obs_loss)
        train_uniform_entropy_loss_list.append(avg_train_uniform_entropy_loss)
        train_uniform_obs_loss_list.append(avg_train_uniform_obs_loss)

        # NOTE: use these to save model for every epoch
        # backup_folder = os.path.join(experiment_folder, "backup_models")
        # os.makedirs(backup_folder, exist_ok=True)
        # backup_file = os.path.join(backup_folder, f"backup_epoch_{epoch+1}_{model.__class__.__name__}.pt")
        # torch.save(model, backup_file)

        if is_single_env: 
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                torch.save(model, os.path.join(
                        experiment_folder, f"{model.__class__.__name__}_best_model_single_env.pt"
                    ))
                if feature_extractor is not None:
                    torch.save(feature_extractor, os.path.join(
                        experiment_folder, f"{feature_extractor.__class__.__name__}_best_model_single_env.pt"
                    ))
                else:
                    # Save a dummy None for openspace baseline
                    torch.save(None, os.path.join(
                        experiment_folder, f"NoneType_best_model_single_env.pt"
                    ))
                print(f"  -------------Model saved at epoch {epoch+1} based on training loss-------------")
            continue

        perf_logger.start("validation")
        model.eval()
        if feature_extractor is not None:
            feature_extractor.eval()
        with torch.no_grad():
            perf_logger.reset()
            perf_logger.start("visualization")
            vis_folder = os.path.join(
                experiment_folder, f"epoch_{epoch}_{model.__class__.__name__}_train_env_visualization"
            )
            os.makedirs(vis_folder, exist_ok=True)
            for env in train_dataset.env_data_list:
                with contextlib.redirect_stdout(None):
                    test_and_save_trajectories(
                        env=env, model=model, feature_extractor=feature_extractor, 
                        vis_folder=vis_folder, num_trajectories=5000,
                        show_vis=False, save_vis=True
                    )
            perf_logger.end("visualization")
            
            perf_logger.start("val_epoch")
            (
                avg_val_entropy_loss, avg_val_obs_loss, 
                avg_val_uniform_entropy_loss, avg_val_uniform_obs_loss,
                avg_val_loss,
            ) = process_one_epoch(
                #only first line 3 variables differed across trainning dataset vs validation dataset
                loader=val_loader, train_mode=False, reps_tensor=val_reps_tensor, 
                model=model, feature_extractor=feature_extractor, device=device, optimizer=optimizer,
                lambda_obs=lambda_obs, lambda_entropy=lambda_entropy,
                actions=actions, dt=dt, v=v, config=config,     
                epoch=epoch, num_epochs=num_epochs,
            )
            perf_logger.end("val_epoch")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model, os.path.join(
                        experiment_folder, f"{model.__class__.__name__}_best_model.pt"
                    ))
                if feature_extractor is not None:
                    torch.save(feature_extractor, os.path.join(
                        experiment_folder, f"{feature_extractor.__class__.__name__}_best_model.pt"
                    ))
                else:
                    # Save a dummy None for openspace baseline
                    torch.save(None, os.path.join(
                        experiment_folder, f"NoneType_best_model.pt"
                    ))
                print(f"  ---Model saved at epoch {epoch+1} based on validation losses---")
        perf_logger.end("validation")
        
        val_entropy_loss_list.append(avg_val_entropy_loss)
        val_obs_loss_list.append(avg_val_obs_loss)
        val_uniform_entropy_loss_list.append(avg_val_uniform_entropy_loss)
        val_uniform_obs_loss_list.append(avg_val_uniform_obs_loss)

        epoch_time = time.time() - epoch_start
        total_epoch_time += epoch_time
        
        # Log per-epoch performance metrics
        epoch_perf_summary = perf_logger.summary(f"Epoch {epoch+1}")
        perf_logger_file.info(f"\n{'='*50}\nEpoch {epoch+1} Complete - Time: {epoch_time:.2f}s")
        perf_logger_file.info(epoch_perf_summary)
        perf_logger_file.info(f"Cumulative training time: {total_epoch_time:.2f}s")
        perf_logger_file.info(f"Estimated time per remaining epoch: {total_epoch_time/(epoch+1):.2f}s")
        perf_logger_file.info(f"Estimated remaining time: {total_epoch_time/(epoch+1) * (num_epochs-epoch-1):.2f}s")
        perf_logger.reset()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f} seconds.")
    
    # Log final training statistics
    perf_logger_file.info(f"\n{'='*50}\nTraining Complete")
    perf_logger_file.info(f"Total training time: {total_time:.2f}s")
    perf_logger_file.info(f"Average time per epoch: {total_time/num_epochs:.2f}s")
    
    return (
        train_entropy_loss_list,
        train_obs_loss_list,
        train_uniform_entropy_loss_list,
        train_uniform_obs_loss_list,
        val_entropy_loss_list,
        val_obs_loss_list,
        val_uniform_entropy_loss_list,
        val_uniform_obs_loss_list
    )

def main():
    torch.manual_seed(2025)
    np.random.seed(2025)
    random.seed(2025)
    torch.cuda.manual_seed_all(2025)
    os.environ["PYTHONHASHSEED"] = str(2025)

    '''
    When set to True, use a fixed set of oversampled states from an open-space environment 
      across all training environments.  
    Normalizes the input and still preserve unique environment-specific level set representatives.
    '''
    SKIP_TRAINING = False
    OPENSPACE_BASELINE = True
    FIXED_OVERSAMPLED_LEVEL_SETS = not OPENSPACE_BASELINE

    # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/env_random_circle_obs_v_2"
    # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/env_single_random_circle_obs_v_2"
    # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/single_random_single_circle_v_2/"
    # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/random_single_circle_v_2/"
    # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/single_env_experiment/"
    # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/realworld_lidar_scan_dataset_valid/"
    if OPENSPACE_BASELINE:
        # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/open_space_env_v_2"
        # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/open_space_env_v_2.5"
        # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/open_space_env_v_2.5_dt0.1"
        # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/open_space_env_v_1.25_dt0.2"
        # base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/open_space_env_v_1.25_dt0.1"
        base_dir = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset/open_space_env_v_1.25_dt0.1_t2.41"
    else:
        base_dir = "TODO"

    print("\n" + "="*60)
    print(" TRAINING CONFIGURATION")
    print("="*60)
    print(f"SKIP_TRAINING:                  {SKIP_TRAINING}")
    print(f"FIXED_OVERSAMPLED_LEVEL_SETS:   {FIXED_OVERSAMPLED_LEVEL_SETS}")
    print(f"OPENSPACE_BASELINE:             {OPENSPACE_BASELINE}")
    print(f"base_dir: {base_dir}")
    print("="*60 + "\n")

    # experiment_folder = "/home/mikasa/RSN/traj_sampling/map_conditioning/unsupervised_models_openspace_v_2.5"
    # experiment_folder = "/home/mikasa/RSN/traj_sampling/map_conditioning/unsupervised_models_openspace_v_2.5_dt0.1"
    # experiment_folder = "/home/mikasa/RSN/traj_sampling/map_conditioning/unsupervised_models_openspace_v_1.25_dt0.2"
    # experiment_folder = "/home/mikasa/RSN/traj_sampling/map_conditioning/unsupervised_models_openspace_v_1.25_dt0.1"
    experiment_folder = "/home/mikasa/RSN/traj_sampling/map_conditioning/unsupervised_models_openspace_v_1.25_dt0.1_t2.41"
    os.makedirs(experiment_folder, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda', "cuda not available"

    if OPENSPACE_BASELINE:
        pixel_feature_extractor = None
        action_predictor = MapAct(state_dim=4, num_actions=31, hidden_dim=512).to(device)
    else:
        # pixel-level feature extractor with decoder
        pixel_feature_extractor = MapPixelFeature().to(device)
        # Create the new action predictor that takes state and interpolated pixel features
        action_predictor = MapAct_PixelInterpolated().to(device)

    # action_predictor = MapAct_directEmbeddingInput().to(device)
    action_predictor_name = action_predictor.__class__.__name__
    feature_extractor_name = pixel_feature_extractor.__class__.__name__
    
    ### Aggregate each environment's data.
    # Each element in env_data_list is a dictionary containing:
    #   "env_folder", "oversampled_level_sets", "representative_list", "sdf", "config", and "map_embedding"
    skip_oversampled_calculation = SKIP_TRAINING or FIXED_OVERSAMPLED_LEVEL_SETS
    env_data_list = aggregate_environment_data(base_dir, skip=skip_oversampled_calculation, openspace = OPENSPACE_BASELINE)
    print("len(env_data_list): ", len(env_data_list))

    ### Split environments into training, validation, and test sets (splitting by environment).
    if len(env_data_list) > 2:
        # train_envs, val_envs, test_envs = split_environments_data(env_data_list, train_ratio=0.72, val_ratio=0.18)
        # train_envs, val_envs, test_envs = split_environments_data(env_data_list, train_ratio=0.80, val_ratio=0.10)
        train_envs, val_envs, test_envs = split_environments_data(
            env_data_list, train_ratio=0.40, val_ratio=0.05
        )
    else:
        train_envs = env_data_list
        val_envs = train_envs
        test_envs = train_envs
    
    config = train_envs[0]["config"]  # Assuming all envs share the same config

    print("Creating Training Dataset...")
    train_envs = UnifiedStateDataset(train_envs, device, fixed_oversampled_level_sets=FIXED_OVERSAMPLED_LEVEL_SETS)
    print("Creating Validation Dataset...")
    val_envs = UnifiedStateDataset(val_envs, device, fixed_oversampled_level_sets=FIXED_OVERSAMPLED_LEVEL_SETS)

    if not SKIP_TRAINING:
        losses = train_model_standardized(
            train_envs, val_envs, action_predictor, pixel_feature_extractor, 
            experiment_folder, device, config, len(env_data_list) < 2
        )
        (
            train_entropy_loss_list, train_obstacle_loss_list,
            train_uniform_entropy_loss_list, train_uniform_obstacle_loss_list,
            val_entropy_loss_list, val_obstacle_loss_list,
            val_uniform_entropy_loss_list, val_uniform_obstacle_loss_list,
        ) = losses

    print("\n" + "=" * 100 + "\nTesting on test environments...")
    # loading models
    MapAct_model_path = os.path.join(
        experiment_folder, 
        f"{action_predictor_name}_best_model.pt" if len(env_data_list) > 2 else 
        f"{action_predictor_name}_best_model_single_env.pt"
    )
    print(f"Action prediction model loaded from {MapAct_model_path}...")
    feature_extractor_path = os.path.join(
        experiment_folder, 
        f"{feature_extractor_name}_best_model.pt" if len(env_data_list) > 2 else 
        f"{feature_extractor_name}_best_model_single_env.pt"
    )
    print(f"Feature extraction model loaded from {feature_extractor_path}...")
    action_predictor_loaded = torch.load(MapAct_model_path, weights_only=False)
    if not OPENSPACE_BASELINE:
        feature_extractor_loaded = torch.load(feature_extractor_path, weights_only=False)
    else:
        feature_extractor_loaded = None
        print("OpenSpace baseline: feature_extractor_loaded is None")

    action_predictor_loaded.eval()
    if feature_extractor_loaded is not None:
        feature_extractor_loaded.eval()
    else:
        print("OpenSpace baseline: skipping feature_extractor_loaded.eval()")

    vis_folder = os.path.join(experiment_folder, f"{action_predictor_name}_test_visualization")
    if os.path.exists(vis_folder):
        shutil.rmtree(vis_folder)
    os.makedirs(vis_folder)
    os.makedirs(vis_folder, exist_ok=True)

    if not SKIP_TRAINING:
        train_recon_loss_list = []
        val_recon_loss_list = []
        plot_loss_curves(
            train_entropy_loss_list, train_obstacle_loss_list,
            train_uniform_entropy_loss_list, train_uniform_obstacle_loss_list,
            val_entropy_loss_list, val_obstacle_loss_list,
            train_recon_loss_list, val_recon_loss_list,
            val_uniform_entropy_loss_list, val_uniform_obstacle_loss_list,
            vis_folder
        )
    for env in test_envs:
        folder = env['env_folder']
        last_bit = os.path.basename(folder)
        print(f"Testing on {last_bit}...")
        with contextlib.redirect_stdout(None):
            test_and_save_trajectories(
                env=env, 
                model=action_predictor_loaded, feature_extractor=feature_extractor_loaded, 
                vis_folder=vis_folder, num_trajectories=10000,
                show_vis=False, save_vis=True, save_pickle=True
            )

if __name__ == "__main__":
    main() 