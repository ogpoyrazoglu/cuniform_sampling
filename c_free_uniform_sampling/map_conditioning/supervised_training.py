# sdf resolution is currently set to 0.05 for all environments during training, make sure dataset matches this
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.optim as optim
import contextlib
import random
import datetime
import time
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from model_parts import MapPixelFeature, MapAct_PixelInterpolated
from utility_helper_map import (
    cuda_dynamics_KS_3d_steering_angle_batched,
    sample_trajectories_feasible,
    bilinear_sample_sdf_features,
    bilinear_sample_sdf_features_batched,
    visualize_trajectories_background,
    set_trajectory_sampling_seeds
)
from dataset_utils import (
    ActionProbDataset,
    environment_level_split, 
    get_dataset_files, 
    find_latest_results_folder, 
    custom_collate_fn, 
    compute_collision_metrics
)

def process_epoch(model: nn.Module, feature_extractor: nn.Module, data_loader: DataLoader, 
                 dataset: ActionProbDataset, optimizer: Optional[torch.optim.Optimizer], 
                 device: torch.device, mode: str = "train", scaler: Optional[torch.amp.GradScaler] = None) -> float:
    """Process one epoch for training, validation, or testing."""
    # Assertions for input validation
    assert isinstance(model, nn.Module), f"model must be nn.Module, got {type(model)}"
    assert isinstance(feature_extractor, nn.Module), f"feature_extractor must be nn.Module, got {type(feature_extractor)}"
    assert isinstance(data_loader, DataLoader), f"data_loader must be DataLoader, got {type(data_loader)}"
    assert isinstance(dataset, ActionProbDataset), f"dataset must be ActionProbDataset, got {type(dataset)}"
    assert mode in ["train", "val", "test"], f"mode must be one of ['train', 'val', 'test'], got {mode}"
    
    if mode == "train":
        assert optimizer is not None, f"optimizer cannot be None for training mode"
        model.train()
        feature_extractor.train()
    else:
        model.eval()
        feature_extractor.eval()
    
    total_loss = 0.0
    num_batches = len(data_loader)
    assert num_batches > 0, f"DataLoader is empty"
    
    context = torch.no_grad() if mode != "train" else contextlib.nullcontext()

    progress_desc = f"{mode.capitalize()} Epoch" if mode == "train" else f"{mode.capitalize()} Evaluation"
    with context:
        for batch_idx, batch in tqdm(enumerate(data_loader), desc=progress_desc, total=num_batches, leave=False):
            # Data preparation with assertions
            states = batch['state'].to(device)
            action_probs = batch['action_probs'].to(device)
            env_indices = torch.tensor(batch['env_idx'], device=device)
            
            # Assertions for batch data
            assert states.dim() == 2, f"States must be 2D tensor, got shape {states.shape}"
            assert action_probs.dim() == 2, f"Action probs must be 2D tensor, got shape {action_probs.shape}"
            assert states.shape[0] == action_probs.shape[0], f"Batch size mismatch: states {states.shape[0]} vs action_probs {action_probs.shape[0]}"
            assert states.shape[1] == dataset.state_dim, f"State dimension mismatch: expected {dataset.state_dim}, got {states.shape[1]}"
            assert action_probs.shape[1] == dataset.num_actions, f"Action dimension mismatch: expected {dataset.num_actions}, got {action_probs.shape[1]}"
            
            # Convert state to network input format (x, y, sin(theta), cos(theta))
            theta = states[:, 2]
            network_states = torch.stack([
                states[:, 0], states[:, 1],
                torch.sin(theta), torch.cos(theta)
            ], dim=1)
            
            # Assertions for network states
            assert network_states.shape == (states.shape[0], 4), f"Network states shape mismatch: expected ({states.shape[0]}, 4), got {network_states.shape}"
            unique_envs_tensor, local_env_indices = torch.unique(env_indices, return_inverse=True)
            sdf_batch_tensor = dataset.master_sdf_tensor[unique_envs_tensor].unsqueeze(1)
            resolutions_batch = dataset.master_resolution_tensor[unique_envs_tensor] # (N_unique_envs,)

            '''
            unique_envs_list = unique_envs_tensor.tolist()
            assert len(unique_envs_list) > 0, f"No environments found in batch"

            # Gather required SDF tensors and Resolutions (already on GPU)
            sdf_tensors_list = []
            resolutions_list = []
            for env in unique_envs_list:
                assert 0 <= env < len(dataset.env_data), f"Environment index {env} out of range"
                sdf_tensors_list.append(dataset.env_data[env]['sdf_tensor_gpu'].detach())
                resolutions_list.append(dataset.env_data[env]['resolution_gpu'].detach())

            # Stack into batches
            # SDF Batch: (N_envs, 1, H, W)
            sdf_batch_tensor = torch.stack(sdf_tensors_list, dim=0).unsqueeze(1)
            # Resolutions Batch: (N_envs)
            resolutions_batch = torch.cat(resolutions_list, dim=0)
            '''
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
                # Run feature extractor ONCE (Batched execution)
                # Output Shape (N_envs, C, H, W)
                dense_features_batch = feature_extractor(sdf_batch_tensor)
                feature_dim = dense_features_batch.shape[1]

                # The "Expand Trick": Align Features and Resolutions with States
                # We use local_env_indices to select the correct feature map/resolution for each state.
                
                # Expand features: (N_envs, C, H, W) -> (Batch_Size, C, H, W) (Batch_Size = len(states))
                expanded_features = dense_features_batch[local_env_indices]
                # Expand resolutions: (N_envs,) -> (Batch_Size,)
                expanded_resolutions = resolutions_batch[local_env_indices]

                # Batched Bilinear Sampling 
                # Input: (Batch_Size, C, H, W), States: (Batch_Size, 4), Resolutions: (Batch_Size,)
                # Output: (Batch_Size, C)
                interpolated_features = bilinear_sample_sdf_features_batched(
                    sdf_features=expanded_features,
                    states=network_states, # Pass the full network states (B, 4)
                    resolutions=expanded_resolutions
                )
                assert interpolated_features.shape == (network_states.shape[0], feature_dim), \
                    f"Interpolated feature shape mismatch: expected ({network_states.shape[0]}, {feature_dim}), got {interpolated_features.shape}"

                # Forward pass
                predicted_probs_amp = model(network_states, interpolated_features)
            predicted_probs = predicted_probs_amp.float()
    
            # Assertions for model output
            assert predicted_probs.shape == action_probs.shape, f"Model output shape mismatch: expected {action_probs.shape}, got {predicted_probs.shape}"
            assert torch.all(predicted_probs >= 0), f"Model output contains negative probabilities"
            assert torch.allclose(predicted_probs.sum(dim=1), torch.ones(predicted_probs.shape[0], device=device), atol=1e-4), f"Model output doesn't sum to 1"
            
            # Normalize target probabilities
            target_prob_sums = action_probs.sum(dim=1, keepdim=True)
            assert torch.all(target_prob_sums > 1e-8), f"Some target probabilities sum to zero"
            target_probs = action_probs / target_prob_sums
            
            # Compute KL divergence loss
            log_predicted_probs = torch.log(predicted_probs + 1e-8)
            loss = torch.nn.functional.kl_div(
                log_predicted_probs, target_probs, reduction='batchmean'
            )
            
            # Assertions for loss
            assert not torch.isnan(loss), f"Loss is NaN"
            assert not torch.isinf(loss), f"Loss is infinite"
            assert loss >= 0, f"KL divergence loss must be non-negative, got {loss.item()}"
            
            # Backward pass (only for training)
            if mode == "train":
                optimizer.zero_grad()
                # scale(loss) multiplies the loss by a scaling factor so .backward() computes gradients on this scaled loss.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update() # adjusts the scaling factor for stability and performance.
                # loss.backward()
                # optimizer.step()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    assert avg_loss >= 0, f"Average loss must be non-negative, got {avg_loss}"
    
    return avg_loss

def train_model(model: nn.Module, feature_extractor: nn.Module, train_loader: DataLoader, 
               val_loader: DataLoader, dataset: ActionProbDataset, device: torch.device, 
               hyperparams: Dict, val_env_indices: List[int] = None, results_folder: str = None) -> Dict[str, List[float]]:
    """Train the model with early stopping."""
    # Assertions for input validation
    assert isinstance(model, nn.Module), f"model must be nn.Module, got {type(model)}"
    assert isinstance(feature_extractor, nn.Module), f"feature_extractor must be nn.Module, got {type(feature_extractor)}"
    assert isinstance(train_loader, DataLoader), f"train_loader must be DataLoader, got {type(train_loader)}"
    assert isinstance(val_loader, DataLoader), f"val_loader must be DataLoader, got {type(val_loader)}"
    assert isinstance(dataset, ActionProbDataset), f"dataset must be ActionProbDataset, got {type(dataset)}"
    assert isinstance(device, torch.device), f"device must be torch.device, got {type(device)}"
    assert isinstance(hyperparams, dict), f"hyperparams must be dict, got {type(hyperparams)}"
    assert val_env_indices is not None, f"val_env_indices must be provided"
    assert results_folder is not None, f"results_folder must be provided"
    
    print("Starting training...")
    
    # Extract hyperparameters
    num_epochs = hyperparams['num_epochs']
    learning_rate = hyperparams['learning_rate']
    weight_decay = hyperparams['weight_decay']
    patience = hyperparams['early_stopping_patience']
    min_delta = hyperparams['early_stopping_min_delta']
    
    # Create validation results folder
    validation_folder = os.path.join(results_folder, "validation")
    os.makedirs(validation_folder, exist_ok=True)
    print(f"Validation results will be saved to: {validation_folder}")
    
    # Create models folder
    models_folder = os.path.join(results_folder, "models")
    os.makedirs(models_folder, exist_ok=True)
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(feature_extractor.parameters()), 
        lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # Monitor validation loss minimization
        factor=hyperparams['lr_scheduler_factor'],
        patience=hyperparams['lr_scheduler_patience'],
        min_lr=hyperparams['lr_scheduler_min_lr'],
    )
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Current LR: {current_lr:.6f}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_val_counter = 0  # Counter for best validation improvements
    
    # Early stopping variables
    early_stopping_counter = 0
    early_stopped = False
    
    print(f"Training configuration:")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Early stopping min delta: {min_delta}")
    
    scaler = torch.amp.GradScaler(device='cuda', enabled=True)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        train_loss = process_epoch(model, feature_extractor, train_loader, dataset, optimizer, device, "train", scaler=scaler)
        train_losses.append(train_loss)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validation phase
        val_loss = process_epoch(model, feature_extractor, val_loader, dataset, None, device, "val", scaler=None)
        val_losses.append(val_loss)
        print(f"  Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        
        # Assertions for training progress
        assert not np.isnan(train_loss), f"Training loss is NaN at epoch {epoch+1}"
        assert not np.isnan(val_loss), f"Validation loss is NaN at epoch {epoch+1}"
        assert train_loss >= 0, f"Training loss is negative at epoch {epoch+1}: {train_loss}"
        assert val_loss >= 0, f"Validation loss is negative at epoch {epoch+1}: {val_loss}"
        
        # Check for improvement and early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_counter += 1
            early_stopping_counter = 0  # Reset early stopping counter
            
            # Save models to models folder
            torch.save(model.state_dict(), os.path.join(models_folder, 'best_model.pth'))
            torch.save(feature_extractor.state_dict(), os.path.join(models_folder, 'best_feature_extractor.pth'))
            print(f"   Best model saved (improvement #{best_val_counter})")
            
            # Perform thorough validation every N improvements
            if best_val_counter % hyperparams['thorough_validation_interval'] == 0:
                print(f"  Performing thorough validation (improvement #{best_val_counter})...")
                
                # Create epoch-specific folder for this validation round
                epoch_folder = os.path.join(validation_folder, f"epoch_{epoch+1:02d}_improvement_{best_val_counter:02d}")
                os.makedirs(epoch_folder, exist_ok=True)
                
                with contextlib.redirect_stdout(None):
                    evaluate_environments(
                        model=model,
                        feature_extractor=feature_extractor,
                        dataset=dataset,
                        device=device,
                        env_indices=val_env_indices,
                        mode="validation",
                        epoch=epoch+1,
                        improvement_count=best_val_counter,
                        output_folder=epoch_folder,
                        num_trajectories=hyperparams['num_trajectories_eval']  # Use hyperparameter for trajectory count
                    )
        else:
            early_stopping_counter += 1
            print(f"  No improvement for {early_stopping_counter}/{patience} epochs")
            
            # Check if we should stop early
            if early_stopping_counter >= patience:
                print(f"  Early stopping triggered! No improvement for {patience} epochs")
                early_stopped = True
                break
    
    if early_stopped:
        print(f"\nTraining Complete (Early Stopped)! Best Loss: {best_val_loss:.4f}")
        print(f"Stopped at epoch {epoch+1}/{num_epochs}")
    else:
        print(f"\nTraining Complete! Best Loss: {best_val_loss:.4f}")
    
    print(f"Total best validation improvements: {best_val_counter}")
    
    # Final assertions
    assert len(train_losses) == epoch + 1, f"Training losses length mismatch"
    assert len(val_losses) == epoch + 1, f"Validation losses length mismatch"
    
    return {
        'train_losses': train_losses, 
        'val_losses': val_losses, 
        'early_stopped': early_stopped,
        'final_epoch': epoch + 1,
        'best_val_loss': best_val_loss
    }

def evaluate_environments(model: nn.Module, feature_extractor: nn.Module, 
                         dataset: ActionProbDataset, device: torch.device,
                         env_indices: List[int], mode: str = "validation",
                         epoch: int = None, improvement_count: int = None, 
                         output_folder: str = None, num_trajectories: int = 10000) -> Dict[int, Tuple[float, float, float]]:
    """
    Evaluate model on specified environments by sampling trajectories and computing losses.
    
    Args:
        model: The trained model
        feature_extractor: The feature extraction model
        dataset: The dataset containing environment data
        device: PyTorch device
        env_indices: List of environment indices to evaluate (0-indexed)
        mode: Either "validation", "test", or "training"
        epoch: Current epoch (for validation mode)
        improvement_count: Best validation improvement count (for validation mode)
        output_folder: Where to save visualizations
        num_trajectories: Number of trajectories to sample for each environment
        
    Returns:
        Dictionary mapping environment index (0-indexed) to (loss, collision_rate, collision_severity) tuple
    """
    assert mode in ["validation", "test", "training"], f"mode must be 'validation', 'test', or 'training', got {mode}"
    assert isinstance(env_indices, list), f"env_indices must be a list, got {type(env_indices)}"
    assert len(env_indices) > 0, f"env_indices cannot be empty"
    
    # Ensure deterministic evaluation by setting seeds before trajectory sampling
    set_trajectory_sampling_seeds(2025)
    
    if output_folder is None:
        output_folder = os.path.join(SCRIPT_DIR, "results", mode)
        os.makedirs(output_folder, exist_ok=True)
    
    print(f"    Starting {mode} evaluation...")
    print(f"    Evaluating {len(env_indices)} environments: {[env_idx + 1 for env_idx in env_indices]} (1-indexed display)")
    
    # Set models to evaluation mode
    model.eval()
    feature_extractor.eval()
    
    env_results = {}  # Will store (loss, collision_rate, collision_severity) tuples with 0-indexed keys
    
    with torch.no_grad():
        for env_idx in env_indices:  # env_idx is 0-indexed here
            if env_idx >= len(dataset.env_data):
                print(f"        Environment {env_idx + 1} (1-indexed) out of range, skipping...")
                continue
                
            print(f"    Processing environment {env_idx + 1} (1-indexed display)...")
            
            # Get environment data
            env_data = dataset.env_data[env_idx]
            config = env_data['config']
            sdf = env_data['sdf']
            costmap = env_data['costmap']
            
            # Assertions for environment data
            assert 'actions' in config, f"Config missing 'actions' key for env {env_idx}"
            assert 'total_t' in config, f"Config missing 'total_t' key for env {env_idx}"
            assert 'dt' in config, f"Config missing 'dt' key for env {env_idx}"
            assert isinstance(sdf, np.ndarray), f"SDF must be numpy array for env {env_idx}"
            assert isinstance(costmap, np.ndarray), f"Costmap must be numpy array for env {env_idx}"
            
            actions = config['actions'][:, 0]
            
            # Convert to tensors
            sdf_tensor = torch.tensor(sdf, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            costmap_tensor = torch.tensor(costmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Compute loss for this environment
            env_loss = _compute_environment_loss(
                model, feature_extractor, dataset, device, env_idx
            )
            
            try:
                # Sample trajectories for this environment
                trajectories = sample_trajectories_feasible(
                    initial_state=np.zeros(3).astype(np.float32),
                    actions=actions,
                    dynamics_cuda=cuda_dynamics_KS_3d_steering_angle_batched,
                    num_trajectories=num_trajectories,
                    trajectory_length=int(config['total_t'] / config['dt']),
                    model=model,
                    feature_extractor=feature_extractor,
                    config=config,
                    sdf_tensor=sdf_tensor,
                    costmap_tensor=costmap_tensor,
                    map_embedding=None,
                    uniform_sampling=False,
                )
                
                # Assertions for sampled trajectories
                assert trajectories is not None, f"No trajectories sampled for env {env_idx}"
                assert len(trajectories) > 0, f"Empty trajectory list for env {env_idx}"
                
                # Compute collision metrics using vectorized function
                collision_rate, collision_severity = compute_collision_metrics(
                    trajectories=trajectories,
                    sdf=sdf,
                    resolution=config.get('resolution', 0.05)
                )
                
                # Store results for this environment (using 0-indexed key)
                env_results[env_idx] = (env_loss, collision_rate, collision_severity)
                
                # Create visualization filename - use env_idx+1 for human-readable names
                if mode == "validation":
                    vis_filename = f"val_env_{env_idx+1:02d}_loss_{env_loss:.4f}.png"
                elif mode == "test":
                    vis_filename = f"test_env_{env_idx+1:02d}_loss_{env_loss:.4f}.png"
                else:  # training mode
                    vis_filename = f"train_env_{env_idx+1:02d}_loss_{env_loss:.4f}.png"
                    
                vis_path = os.path.join(output_folder, vis_filename)
                
                # Create title with loss and collision information - use 1-indexed for display
                title = f"Environment {env_idx + 1} - {mode.capitalize()}\nKL Loss: {env_loss:.4f} | Collision Rate: {collision_rate:.1f}% | Collision Severity: {collision_severity:.3f}"
                
                resolution = config.get('resolution', 0.05)
                
                # Visualize trajectories with loss in title using modified function
                visualize_trajectories_background(
                    trajectories=trajectories,
                    costmap=costmap,
                    resolution=resolution,
                    show_vis=False,
                    save_vis=True,
                    vis_filepath=vis_path,
                    alpha=0.4,
                    marker_size=1,
                    title=title  # Pass title to the function
                )
                
                print(f"       Saved {mode} visualization: {vis_filename}")
                
            except Exception as e:
                print(f"       Error sampling trajectories for env {env_idx + 1} (1-indexed display): {str(e)}")
                # Still store the loss even if trajectory sampling failed (using 0-indexed key)
                env_results[env_idx] = (env_loss, 0.0, 0.0)  # Default collision metrics
                continue
    
    # Create a summary file for this evaluation round
    if mode == "validation":
        summary_filename = f"validation_summary.txt"
    elif mode == "test":
        summary_filename = f"test_summary.txt"
    else:  # training mode
        summary_filename = f"training_summary.txt"
        
    summary_path = os.path.join(output_folder, summary_filename)
    
    with open(summary_path, 'w') as f:
        f.write(f"{mode.capitalize()} Evaluation Summary\n")
        f.write(f"{'=' * (len(mode) + 20)}\n")
        if mode == "validation":
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Best Validation Improvement Count: {improvement_count}\n")
        # Convert 0-indexed env_indices to 1-indexed for human-readable display
        env_display_numbers = [env_idx + 1 for env_idx in env_indices]
        f.write(f"Environments (1-indexed): {env_display_numbers}\n")
        f.write(f"Number of Trajectories per Environment: {num_trajectories}\n")
        f.write(f"\nDetailed Results:\n")
        f.write(f"{'Environment':<12} {'KL Loss':<10} {'Collision Rate (%)':<18} {'Collision Severity':<18}\n")
        f.write(f"{'-'*60}\n")
        
        # Extract metrics for summary calculations
        losses = []
        collision_rates = []
        collision_severities = []
        
        for env_idx in sorted(env_results.keys()):  # env_idx is 0-indexed
            loss, coll_rate, coll_severity = env_results[env_idx]
            losses.append(loss)
            collision_rates.append(coll_rate)
            collision_severities.append(coll_severity)
            # Display as 1-indexed in the table
            f.write(f"{env_idx + 1:<12} {loss:<10.6f} {coll_rate:<18.2f} {coll_severity:<18.3f}\n")
        
        # Write summary statistics
        f.write(f"{'-'*60}\n")
        f.write(f"{'Average':<12} {np.mean(losses):<10.6f} {np.mean(collision_rates):<18.2f} {np.mean(collision_severities):<18.3f}\n")
        f.write(f"{'Std Dev':<12} {np.std(losses):<10.6f} {np.std(collision_rates):<18.2f} {np.std(collision_severities):<18.3f}\n")
        f.write(f"\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"    {mode.capitalize()} summary saved: {summary_filename}")
    print(f"    {mode.capitalize()} evaluation complete!")
    
    return env_results

def _compute_environment_loss(model: nn.Module, feature_extractor: nn.Module,
                             dataset: ActionProbDataset, device: torch.device,
                             env_idx: int) -> float:
    """Compute the average loss for a specific environment.
    
    Args:
        env_idx: Environment index (0-indexed)
    """
    # Collect all samples from this environment
    env_samples = []
    for idx, sample in enumerate(dataset.data):
        sample_env_idx = int(sample[dataset.state_dim + dataset.num_actions])
        if sample_env_idx == env_idx:
            env_samples.append(idx)
    
    if len(env_samples) == 0:
        print(f"        No samples found for environment {env_idx + 1} (1-indexed display)")
        return 0.0
    
    # Create a subset for this environment
    env_subset = Subset(dataset, env_samples)
    env_loader = DataLoader(env_subset, batch_size=min(1024, len(env_samples)), 
                           shuffle=False, collate_fn=custom_collate_fn)
    
    # Compute loss using existing process_epoch function
    env_loss = process_epoch(model, feature_extractor, env_loader, dataset, None, device, "test", scaler=None)
    return env_loss

def test_model(model: nn.Module, feature_extractor: nn.Module, test_loader: DataLoader, 
              dataset: ActionProbDataset, device: torch.device) -> float:
    """Test the trained model."""
    # Assertions for input validation
    assert isinstance(model, nn.Module), f"model must be nn.Module, got {type(model)}"
    assert isinstance(feature_extractor, nn.Module), f"feature_extractor must be nn.Module, got {type(feature_extractor)}"
    assert isinstance(test_loader, DataLoader), f"test_loader must be DataLoader, got {type(test_loader)}"
    assert isinstance(dataset, ActionProbDataset), f"dataset must be ActionProbDataset, got {type(dataset)}"
    assert isinstance(device, torch.device), f"device must be torch.device, got {type(device)}"
    
    test_loss = process_epoch(model, feature_extractor, test_loader, dataset, None, device, "test", scaler=None)
    
    # Assertions for test results
    assert not np.isnan(test_loss), f"Test loss is NaN"
    assert test_loss >= 0, f"Test loss is negative: {test_loss}"
    
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss

def create_master_tensors(dataset: ActionProbDataset):
    print("Attempting to create master SDF and Resolution tensors...")
    assert dataset.env_data is not None, f"Dataset is empty"
    first_shape = dataset.env_data[0]['sdf_tensor_gpu'].shape
    for env in dataset.env_data:
        assert env['sdf_tensor_gpu'].shape == first_shape, f"SDF dimensions are not uniform. Expected {first_shape}, got {env['sdf_tensor_gpu'].shape}"

    all_sdfs = [env['sdf_tensor_gpu'].detach() for env in dataset.env_data]
    all_resolutions = [env['resolution_gpu'].detach() for env in dataset.env_data]
    
    dataset.master_sdf_tensor = torch.stack(all_sdfs, dim=0) #(Total_N_envs, H, W)
    dataset.master_resolution_tensor = torch.cat(all_resolutions, dim=0) # (Total_N_envs,)
    print(f"Master tensors created successfully. SDF Shape: {dataset.master_sdf_tensor.shape}")

def main():
    torch.set_float32_matmul_precision('high')
    # =============================================================================
    # HYPERPARAMETERS CONFIGURATION
    # =============================================================================
    hyperparams = {
        'random_seed': 2025, # do not change this seed, it's hardcoded as 2025 in other files
        
        # Training configuration
        'skip_training': False,
        # 'dataset_name': "shepherd_dataset_supervised_cleaned",
        # 'dataset_name': "shepherd_dataset_supervised_cleaned_resolution_0.05_v1_sdfNoInflation",
        'dataset_name': "barn_dataset_vmax2.5",
        # 'dataset_name': "barn_dataset",  
        # 'dataset_name': "temp_barn",  
        'num_epochs': 100,  
        'batch_size': 450,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        
        # Early stopping
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1e-3,
        
        # Learning rate scheduler
        'lr_scheduler_factor': 0.3,     # Reduce LR by 30% when triggered
        'lr_scheduler_patience': 2,     # Wait 2 epoches of no improvement (less than early stopping patience)
        'lr_scheduler_min_lr': 1e-6,    # Lower bound for LR
        
        # Dataset splits
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        
        # DataLoader settings
        'num_workers': 24,
        
        # Simple boundary augmentation: generate N noisy samples around each boundary state
        'boundary_samples_per_state': 0,    # Number of noisy samples generated per boundary state

        'initial_level_set_augmentation_samples': 100, # Number of noisy samples to add around the initial state [0,0,0]
        
        # Evaluation settings
        'thorough_validation_interval': 10,  # Every N improvements
        'num_trajectories_eval': 1000,
    }
    
    # =============================================================================
    # COMPREHENSIVE DETERMINISM SETUP
    # =============================================================================
    def set_all_seeds(seed):
        """Set all possible random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            
            # CUDA deterministic operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Environment variables for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Force synchronous CUDA operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Fix CuBLAS non-determinism
        
        print(f"All seeds set to {seed} for maximum reproducibility")
        print(f"   - CUDA deterministic: {torch.backends.cudnn.deterministic}")
        print(f"   - CUDA benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   - Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
    
    # Apply comprehensive seed setting
    set_all_seeds(hyperparams['random_seed'])
    
    # =============================================================================
    # DEVICE AND MODEL INITIALIZATION
    # =============================================================================
    # Create timestamp for this training run (month abbreviation + day + hour + minutes format)
    now = datetime.datetime.now()
    training_timestamp = f"{now.strftime('%b')}{now.day}-{now.strftime('%H-%M-%S')}"

    # Determine results folder based on skip_training setting
    if hyperparams['skip_training']:
        # Find the latest existing results folder
        results_folder = find_latest_results_folder()
        print(f"Using existing results folder: {results_folder}")
    else:
        # Create new timestamped results folder
        results_folder = os.path.join(SCRIPT_DIR, f"results_{training_timestamp}")
        os.makedirs(results_folder, exist_ok=True)
        print(f"Results will be saved to: {results_folder}")
    
    # Save the hyperparameters to the results folder as a txt file
    with open(os.path.join(results_folder, 'hyperparameters.txt'), 'w') as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =============================================================================
    # DATASET LOADING AND PROCESSING
    # =============================================================================
    # Get dataset files - sort for deterministic order
    data_files = sorted(get_dataset_files(hyperparams['dataset_name']))
    
    print("Supervised Training (Distribution Matching)")
    print(f"Dataset: {hyperparams['dataset_name']} ({len(data_files)} environments)")
    print(f"Training run timestamp: {training_timestamp}")
    print(f"Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    print("\nLoading dataset...")
    dataset = ActionProbDataset(
        data_files, 
        augmentation_params={
            'boundary_samples_per_state': hyperparams['boundary_samples_per_state'],
            'initial_level_set_augmentation_samples': hyperparams['initial_level_set_augmentation_samples']
        }
    )
    # Move all SDFs to GPU memory
    print(f"Moving {len(dataset.env_data)} environment SDFs to {device}...")
    for env_data in dataset.env_data:
        sdf_array = env_data['sdf'] # Get the numpy array stored under the 'sdf' key
        
        # Convert to tensor, move to device (GPU), and store it back in the dictionary
        # under a new key 'sdf_tensor_gpu'. Shape (H, W) e.g., (121, 121)
        env_data['sdf_tensor_gpu'] = torch.tensor(sdf_array, dtype=torch.float32).to(device)
        resolution = env_data['config'].get('resolution', 0.05)
        # Store as a 1-element tensor for easy concatenation later
        env_data['resolution_gpu'] = torch.tensor([resolution], dtype=torch.float32).to(device)


    print("Successfully moved SDFs to GPU.")
    
    # Assertions for loaded dataset
    assert len(dataset) > 0, f"Dataset is empty"
    assert dataset.num_actions > 0, f"Number of actions must be positive, got {dataset.num_actions}"
    assert len(dataset.env_data) == len(data_files), f"Environment data count mismatch: expected {len(data_files)}, got {len(dataset.env_data)}"
    
    # Split dataset with proper 80/10/10 split
    train_dataset, val_dataset, test_dataset, env_assignments = environment_level_split(
        dataset, 
        train_ratio=hyperparams['train_ratio'], 
        val_ratio=hyperparams['val_ratio'], 
        test_ratio=hyperparams['test_ratio']
    )

    original_dataset_size = len(dataset)

    print("\n--- (BEFORE AUGMENTATION) ---")
    print(f"Size of main dataset: {len(dataset):,}")
    print(f"Size of train_dataset subset: {len(train_dataset):,}")
    print(f"Size of val_dataset subset: {len(val_dataset):,}")
    print(f"Size of test_dataset subset: {len(test_dataset):,}")
    print("-------------------------------------------\n")
    
    # Assertions for dataset splits (before augmentation)
    original_dataset_size = len(dataset)
    assert len(train_dataset) > 0, f"Training dataset is empty"
    assert len(val_dataset) > 0, f"Validation dataset is empty"
    assert len(test_dataset) > 0, f"Test dataset is empty"
    assert len(train_dataset) + len(val_dataset) + len(test_dataset) == original_dataset_size, f"Split sizes don't match original dataset size"
    
    # Print environment assignments
    print(f"Environment assignments:")
    print(f"  Train environments: {[env+1 for env in env_assignments['train']]}")
    print(f"  Validation environments: {[env+1 for env in env_assignments['val']]}")
    print(f"  Test environments: {[env+1 for env in env_assignments['test']]}")
    
    # Apply augmentations ONLY to training environments
    print(f"\nApplying augmentations only to training environments...")
    # Apply initial level set augmentation first
    print(f"\nApplying initial level set augmentation only to training environments...")
    dataset.apply_initial_level_set_augmentation(
        env_indices=env_assignments['train'],
        initial_level_set_augmentation_samples=hyperparams['initial_level_set_augmentation_samples']
    )
    
    # Apply boundary augmentation
    print(f"\nApplying boundary augmentation only to training environments...")
    dataset.apply_boundary_augmentation_to_environments(
        env_indices=env_assignments['train'],
        boundary_samples_per_state=hyperparams['boundary_samples_per_state'],
        # visualization_save_dir=os.path.join(results_folder, "training_set_augmentation_visualizations"),
        visualization_save_dir=None
    )
    
    # Verify dataset size increased due to augmentation
    augmented_dataset_size = len(dataset)
    added_samples = augmented_dataset_size - original_dataset_size
    print(f"Dataset size after augmentation: {original_dataset_size} → {augmented_dataset_size} (+{added_samples} samples)")

    if added_samples > 0:
        print("Updating training subset to include newly augmented data...")
        original_train_indices = train_dataset.indices
        newly_added_indices = list(range(original_dataset_size, augmented_dataset_size))
        train_dataset.indices = original_train_indices + newly_added_indices

    # --- VERIFICATION STEP 3: FINAL SIZE CHECK (AFTER FIX) ---
    print(f"Final size of train_dataset subset: {len(train_dataset):,}")
    
    # =============================================================================
    # DATA LOADERS WITH DETERMINISTIC SETTINGS
    # =============================================================================
    
    # Worker init function for deterministic multiprocessing (if num_workers > 0)
    def worker_init_fn(worker_id):
        worker_seed = hyperparams['random_seed'] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    # Create data loaders with deterministic settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hyperparams['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate_fn, 
        num_workers=hyperparams['num_workers'],
        worker_init_fn=worker_init_fn if hyperparams['num_workers'] > 0 else None,
        generator=torch.Generator().manual_seed(hyperparams['random_seed']),  # Deterministic shuffling
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=hyperparams['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=hyperparams['num_workers'],
        worker_init_fn=worker_init_fn if hyperparams['num_workers'] > 0 else None,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=hyperparams['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=hyperparams['num_workers'],
        worker_init_fn=worker_init_fn if hyperparams['num_workers'] > 0 else None,
        pin_memory=True
    )
    
    # Assertions for data loaders
    assert len(train_loader) > 0, f"Training data loader is empty"
    assert len(val_loader) > 0, f"Validation data loader is empty"
    assert len(test_loader) > 0, f"Test data loader is empty"
    
    # =============================================================================
    # MODEL INITIALIZATION WITH DETERMINISTIC WEIGHTS
    # =============================================================================
    # Re-seed before model initialization for deterministic weights
    set_all_seeds(hyperparams['random_seed'])
    
    # Initialize models
    feature_extractor = MapPixelFeature()
    model = MapAct_PixelInterpolated(num_actions=dataset.num_actions)
    
    # Move models to device
    feature_extractor = feature_extractor.to(device)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in feature_extractor.parameters())
    print(f"Models initialized: {total_params:,} parameters")
    assert total_params > 0, f"Model has no parameters"
    
    print(f"Compiling models with torch.compile(dynamic=True). This may take a few minutes on the first run.")
    try:
        # Compile both models with dynamic=True to handle varying batch dimensions safely.
        feature_extractor_compiled = torch.compile(feature_extractor, dynamic=True, mode="default")
        model_compiled = torch.compile(model, dynamic=True, mode="default")
        print("Compilation successful.")
    except Exception as e:
        print(f"Warning: torch.compile failed with error: {e}. Proceeding in eager mode (slower).")
        feature_extractor_compiled = feature_extractor
        model_compiled = model

    plot_path = os.path.join(results_folder, f'training_progress_{hyperparams["dataset_name"]}.png')
    
    create_master_tensors(dataset)
    if not hyperparams['skip_training']:
        # Train model with validation environment tracking and results folder
        train_result = train_model(
            model_compiled, feature_extractor_compiled, train_loader, val_loader, dataset, device, 
            hyperparams, val_env_indices=env_assignments['val'], results_folder=results_folder
        )
        
        # Assertions for training results
        assert 'train_losses' in train_result, f"Training result missing train_losses"
        assert 'val_losses' in train_result, f"Training result missing val_losses"
        assert 'early_stopped' in train_result, f"Training result missing early_stopped"
        assert 'final_epoch' in train_result, f"Training result missing final_epoch"
        assert 'best_val_loss' in train_result, f"Training result missing best_val_loss"
        
        # Simple training plot
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_result['train_losses']) + 1)
        plt.plot(epochs, train_result['train_losses'], label='Training Loss', color='blue')
        plt.plot(epochs, train_result['val_losses'], label='Validation Loss', color='red')
        
        # Add early stopping info to title
        title = f'Training Progress - {hyperparams["dataset_name"]}'
        if train_result['early_stopped']:
            title += f' (Early Stopped at Epoch {train_result["final_epoch"]})'
        else:
            title += f' (Completed {train_result["final_epoch"]} Epochs)'
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Add best validation loss annotation
        plt.axhline(y=train_result['best_val_loss'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Best Val Loss: {train_result["best_val_loss"]:.4f}')
        plt.legend()
        
        plt.savefig(plot_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Training plot saved to {plot_path}")
    
    # Load best models for testing from models folder
    print("Loading best models for testing...")
    models_folder = os.path.join(results_folder, "models")
    best_model_path = os.path.join(models_folder, 'best_model.pth')
    best_feature_path = os.path.join(models_folder, 'best_feature_extractor.pth')
    
    # Assertions for model files
    assert os.path.exists(best_model_path), f"Best model file not found: {best_model_path}"
    assert os.path.exists(best_feature_path), f"Best feature extractor file not found: {best_feature_path}"
    
    model_compiled.load_state_dict(torch.load(best_model_path, weights_only=True))
    feature_extractor_compiled.load_state_dict(torch.load(best_feature_path, weights_only=True))
    
    # Test the model
    test_loss = test_model(model_compiled, feature_extractor_compiled, test_loader, dataset, device)
    
    # Create test results folder
    test_folder = os.path.join(results_folder, "test")
    os.makedirs(test_folder, exist_ok=True)
    
    # Perform thorough test evaluation for all test environments
    print(" Performing thorough test evaluation...")
    
    with contextlib.redirect_stdout(None):
        test_env_results = evaluate_environments(
            model=model_compiled,
            feature_extractor=feature_extractor_compiled,
            dataset=dataset,
            device=device,
            env_indices=env_assignments['test'],
            mode="test",
            output_folder=test_folder,
            num_trajectories=hyperparams['num_trajectories_eval']
        )
    
    # Create training evaluation folder and evaluate training environments
    print(" Performing training environment evaluation...")
    training_folder = os.path.join(results_folder, "training")
    os.makedirs(training_folder, exist_ok=True)
    
    # with contextlib.redirect_stdout(None):
    train_env_results = evaluate_environments(
        model=model_compiled,
        feature_extractor=feature_extractor_compiled,
        dataset=dataset,
        device=device,
        env_indices=env_assignments['train'],
        mode="training",
        output_folder=training_folder,
        num_trajectories=hyperparams['num_trajectories_eval']
    )
    
    # Print final summary
    print("\nFinal Summary:")
    print(f"  Timestamp: {training_timestamp}")
    print(f"  Dataset: {hyperparams['dataset_name']}")
    print(f"  Environments: {len(dataset.env_data)}")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Overall test loss: {test_loss:.4f}")
    print(f"  Model parameters: {total_params:,}")
    
    # Print training summary if training was performed
    if not hyperparams['skip_training']:
        print(f"  Training summary:")
        if train_result['early_stopped']:
            print(f"    Early stopped at epoch: {train_result['final_epoch']}")
        else:
            print(f"    Completed all epochs: {train_result['final_epoch']}")
        print(f"    Best validation loss: {train_result['best_val_loss']:.4f}")
    else:
        print(f"  Training was skipped - using existing models from: {os.path.basename(results_folder)}")
    
    print(f"  Environment splits:")
    print(f"    Train envs: {[env+1 for env in env_assignments['train']]}")
    print(f"    Val envs: {[env+1 for env in env_assignments['val']]}")
    print(f"    Test envs: {[env+1 for env in env_assignments['test']]}")
    
    # Print individual environment losses
    if train_env_results:
        print(f"  \nIndividual training environment results:")
        train_losses = []
        for env_idx, (loss, coll_rate, coll_severity) in sorted(train_env_results.items()):
            print(f"    Environment {env_idx + 1} (1-indexed): Loss: {loss:.4f}, Collision Rate: {coll_rate:.1f}%, Collision Severity: {coll_severity:.3f}")
            train_losses.append(loss)
        avg_train_loss = np.mean(train_losses)
        print(f"    Average training environment loss: {avg_train_loss:.4f}")
    
    if test_env_results:
        print(f"  \nIndividual test environment results:")
        test_losses = []
        for env_idx, (loss, coll_rate, coll_severity) in sorted(test_env_results.items()):
            print(f"    Environment {env_idx + 1} (1-indexed): Loss: {loss:.4f}, Collision Rate: {coll_rate:.1f}%, Collision Severity: {coll_severity:.3f}")
            test_losses.append(loss)
        avg_test_loss = np.mean(test_losses)
        print(f"    Average test environment loss: {avg_test_loss:.4f}")
    
    print(f"  Results saved to:")
    print(f"    Training plot: {plot_path if not hyperparams['skip_training'] else 'Not generated (training skipped)'}")
    print(f"    Models: {models_folder}")
    print(f"    Test results: {test_folder}")
    print(f"    Training evaluation: {training_folder}")
    if not hyperparams['skip_training']:
        print(f"    Validation results: {os.path.join(results_folder, 'validation')}")

if __name__ == "__main__":
    main()