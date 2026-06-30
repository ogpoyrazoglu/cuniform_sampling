"""Dynamics utilities for vehicle dynamic simulation """
import numpy as np
import torch
import math


def bicycle_model_step(state: np.ndarray, control: np.ndarray, dt: float, wheelbase: float) -> np.ndarray:
    """Update vehicle state using bicycle model.
    
    Args:
        state: [x, y, theta, v] current state
        control: [v_cmd, steer_angle] control input
        dt: time step
        wheelbase: vehicle wheelbase
        
    Returns:
        Updated state [x, y, theta, v]
    """
    x, y, theta, _ = state
    v_cmd, steer_angle = control
    
    # Update theta first, then x and y (more physically accurate)
    new_theta = theta + dt * v_cmd * np.tan(steer_angle) / wheelbase
    new_x = x + dt * v_cmd * np.cos(new_theta)
    new_y = y + dt * v_cmd * np.sin(new_theta)
    
    return np.array([new_x, new_y, new_theta, v_cmd])

@torch.jit.script
def cuda_dynamics_KS_3d_scalar_v_batched(states: torch.Tensor, actions: torch.Tensor, dt: float, v: float, wheelbase: float) -> torch.Tensor:
    """
    Vectorized dynamic propagation for a batch of state-action pairs.
    
    Instead of propagating each state with all possible actions,
    this function assumes that for each state in the batch a corresponding 
    action is provided. Thus, the function computes one propagated state 
    per input state-action pair.
    
    Args:
        states (torch.Tensor): Tensor of shape (batch_size, 3) representing
            the current states, where each state is [x, y, theta].
        actions (torch.Tensor): Tensor of shape (batch_size,) or (batch_size,1)
            representing the steering angle for each corresponding state.
        dt (float): Time step for propagation.
        v (float): Velocity, assumed constant across the batch.
        wheelbase (float): Vehicle wheelbase (distance between axles).
        
    Returns:
        next_states (torch.Tensor): Tensor of shape (batch_size, 3) containing
            the propagated states for each state-action pair.
    """
    assert states.dim() == 2 and states.shape[1] == 3, "states must be of shape (batch_size, 3)"
    assert states.shape[0] == actions.shape[0], "Batch size of states and actions must match"
    assert actions.dim() in [1, 2] and (actions.dim() == 1 or actions.shape[1] == 1), \
        "actions must be of shape (batch_size,) or (batch_size, 1)"
    
    # Extract state components: x, y positions and orientation theta.
    # Shape: (batch_size,)
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    
    # Compute the new orientation (theta_new) for each state-action pair.
    # each element in 'actions' corresponds to the desired steering angle for the respective state.
    theta_new = theta + (v / wheelbase) * torch.tan(actions.squeeze()) * dt
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [−π, π]: 
    
    # Compute the new x position using the updated orientation.
    x_new = x + v * torch.cos(theta_new) * dt
    
    # Compute the new y position using the updated orientation.
    y_new = y + v * torch.sin(theta_new) * dt
    
    # Stack the propagated components to form the new states tensor.
    # Each state is now [x_new, y_new, theta_new].
    next_states = torch.stack([x_new, y_new, theta_new], dim=1)
    return next_states

@torch.jit.script
def cuda_dynamics_KS_3d_variable_v_batched(states: torch.Tensor, controls: torch.Tensor, dt: float, wheelbase: float) -> torch.Tensor:
    """
    Vectorized dynamic propagation for variable velocity inputs.
    
    Args:
        states (torch.Tensor): (batch_size, 3) [x, y, theta].
        controls (torch.Tensor): (batch_size, 2) [v_cmd, steer_angle].
        dt (float): Time step.
        wheelbase (float): Vehicle wheelbase.
        
    Returns:
        next_states (torch.Tensor): (batch_size, 3).
    """
    assert states.dim() == 2 and states.shape[1] == 3
    assert controls.dim() == 2 and controls.shape[1] == 2
    assert states.shape[0] == controls.shape[0]
    
    # Extract state components
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    
    # Extract control components
    v_cmd = controls[:, 0]
    steer_angle = controls[:, 1]
    
    # Compute the new orientation (theta_new) - Semi-Implicit Euler
    theta_new = theta + (v_cmd / wheelbase) * torch.tan(steer_angle) * dt
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [−π, π]
    
    # Compute the new position using the updated orientation.
    x_new = x + v_cmd * torch.cos(theta_new) * dt
    y_new = y + v_cmd * torch.sin(theta_new) * dt
    
    next_states = torch.stack([x_new, y_new, theta_new], dim=1)
    return next_states