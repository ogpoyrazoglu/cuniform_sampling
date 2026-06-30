"""
BASELINE SUPERVISED TRAINING FOR OPEN SPACE ENVIRONMENTS
========================================================

This is a simplified baseline training script for open space (no obstacles) environments.
It serves as a reference implementation and comparison point for the main supervised training.

Key differences from map conditioned c-uniform training:
- No map/SDF features (open space only)
- Simple MapAct model without feature extraction
"""

import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model_parts import MapAct
from utility_helper_map import cuda_dynamics_KS_3d_steering_angle_batched, sample_trajectories_feasible, visualize_trajectories_background

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class OpenSpaceDataset(Dataset):
    """Simple dataset for open space action probability training."""
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            
        self.samples = []
        state_dim = data['config']['state_dim']
        
        for level_set in data['action_prob_list']:
            for sample in level_set:
                state = sample[:state_dim]
                probs = sample[state_dim:]
                
                # Convert to network format: [x, y, sin(θ), cos(θ)]
                if len(state) >= 3:
                    theta = state[2]
                    network_state = np.array([state[0], state[1], np.sin(theta), np.cos(theta)])
                else:
                    network_state = state
                    
                self.samples.append((network_state, probs))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        state, probs = self.samples[idx]
        return {
            'state': torch.FloatTensor(state),
            'target_probs': torch.FloatTensor(probs)
        }

def train_baseline_model(model, train_loader, val_loader, device, num_epochs=500, patience=20):
    """Train the baseline model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(" Starting baseline training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            states = batch['state'].to(device)
            target_probs = batch['target_probs'].to(device)
            
            pred_probs = model(states)
            loss = criterion(torch.log(pred_probs + 1e-10), target_probs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch['state'].to(device)
                target_probs = batch['target_probs'].to(device)
                
                pred_probs = model(states)
                loss = criterion(torch.log(pred_probs + 1e-10), target_probs)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f" Early stopping at epoch {epoch + 1}")
            break
            
        # Print progress every epoch in the requested format
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses, best_val_loss

def main():
    # Configuration
    torch.manual_seed(2025)
    np.random.seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data file path
    data_file = "/home/mikasa/RSN/traj_sampling/map_conditioning/dataset_supervised/saved_pickles/C_Uniform_processed_disjoint_1_init_KS_3D_STEERING_ANGLE_obs_none_perturb_2.01_seed_2025_grid_0.100_0.100_6.000deg_t1.21_ts0.2_vrange_2.5_2.5_steer_range_-30.0_30.0_steering_31.pkl"
    
    print(" BASELINE SUPERVISED TRAINING (Open Space)")
    print("=" * 50)
    print(f"Device: {device}")
    
    # Load dataset
    dataset = OpenSpaceDataset(data_file)
    print(f"Dataset size: {len(dataset):,} samples")
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)
    
    # Create baseline model (simple MapAct without feature extraction)
    model = MapAct(state_dim=4, num_actions=31, hidden_dim=1024).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create baseline models directory
    baseline_dir = os.path.join(SCRIPT_DIR, "baseline_models_openspace_supervised")
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Train model
    train_losses, val_losses, best_val_loss = train_baseline_model(
        model, train_loader, val_loader, device
    )
    
    # Save best model
    model_path = os.path.join(baseline_dir, 'baseline_openspace_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f" Model saved: {model_path}")
    
    # Save training plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Baseline Training Progress (Open Space)')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(baseline_dir, 'baseline_training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved: {plot_path}")
    
    # Load config for trajectory sampling
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    config = data['config']
    
    # Sample trajectories for visualization
    print(" Sampling trajectories for visualization...")
    trajectories = sample_trajectories_feasible(
        initial_state=np.zeros(3).astype(np.float32),
        actions=config['actions'][:, 0],
        dynamics_cuda=cuda_dynamics_KS_3d_steering_angle_batched,
        num_trajectories=10000,
        trajectory_length=int(config['total_t'] / config['dt']),
        model=model,
        feature_extractor=None,  # No feature extractor for baseline
        config=config,
        sdf_tensor=None,  # Open space - no SDF
        costmap_tensor=None,  # Open space - no costmap
        map_embedding=None,
        uniform_sampling=False,
    )
    
    # Create visualization
    vis_path = os.path.join(baseline_dir, 'baseline_trajectories.png')
    
    # Create dummy costmap for open space (all False = free space)
    dummy_costmap = np.zeros((121, 121), dtype=bool)  # Boolean array: False = free space, True = obstacles
    
    visualize_trajectories_background(
        trajectories=trajectories,
        costmap=dummy_costmap,  # Dummy open space costmap
        resolution=0.05,
        show_vis=False,
        save_vis=True,
        vis_filepath=vis_path,
        alpha=0.4,
        marker_size=1,
        title=f"Baseline Open Space Trajectories\nBest Val Loss: {best_val_loss:.4f}"
    )
    print(f"Visualization saved: {vis_path}")
    
    # Save trajectories as pickle file in the same directory
    traj_path = os.path.join(baseline_dir, 'baseline_trajectories.pkl')
    with open(traj_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Trajectories saved: {traj_path}")
    
    # Summary
    print("\n" + "=" * 50)
    print(" BASELINE TRAINING COMPLETE")
    print(f" Final validation loss: {best_val_loss:.6f}")
    print(f" All outputs saved to: {baseline_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()