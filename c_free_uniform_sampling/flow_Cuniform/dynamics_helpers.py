import numpy as np
import math
from pprint import pprint
import torch

# helper functions for robot dynamics, (vectorized)
def angle_diff(theta_target, theta_start):
# minimal difference between two angles in radians, wrapped to the interval [-pi, pi].
# when computing (theta_target-theta_start) raw difference can be large in magnitude instead of the minimal rotation
    diff = theta_target - theta_start
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff



########################################## DUBINS MODEL BELOW ##########################################
########################################## DUBINS MODEL BELOW ##########################################
########################################## DUBINS MODEL BELOW ##########################################
def dynamics_dubins_4d(state, action, dt, vrange):
    """
    Propagates a 4D state (x, y, theta, v) with a 2D control input (steering, v_control).
    Args:
        state (tuple): (x, y, theta, v)
        action (tuple): (steering, v_control)
        dt (float): time step.
    Returns: tuple: (x_new, y_new, theta_new, v_new)
    """
    x, y, theta, v0 = state
    steering, acceleration = action

    v_new = v0 + acceleration * dt
    v_new = np.clip(v_new, vrange[0], vrange[1])
    x_new = x + v_new * np.cos(theta) * dt
    y_new = y + v_new * np.sin(theta) * dt
    theta_new = theta + steering * dt
    return (x_new, y_new, theta_new, v_new)

def vectorized_dynamics_dubins_4d(states, actions, dt, vrange):
    """
    Computes the next states for many 4D states and a set of 2D actions.
    Args:
        states (np.ndarray): shape (n, 4) where each state is (x, y, theta, v).
        actions (np.ndarray): shape (nu, 2) where each action is (steering, v_control).
        dt (float): time step.
    Returns: np.ndarray: New states of shape (n * nu, 4).
    """
    n = states.shape[0]
    nu = actions.shape[0]
    # For every state, consider all actions:
    states_repeated = np.repeat(states, nu, axis=0)  # shape: (n * nu, 4)
    actions_tiled = np.tile(actions, (n, 1))         # shape: (n * nu, 2)

    x = states_repeated[:, 0]
    y = states_repeated[:, 1]
    theta = states_repeated[:, 2]
    v0 = states_repeated[:, 3]

    steering = actions_tiled[:, 0]
    acceleration = actions_tiled[:, 1]

    v_new = v0 + acceleration * dt 
    v_new = np.clip(v_new, vrange[0], vrange[1])
    x_new = x + v_new * np.cos(theta) * dt
    y_new = y + v_new * np.sin(theta) * dt
    theta_new = theta + steering * dt
    new_states = np.stack((x_new, y_new, theta_new, v_new), axis=1).astype(np.float32)
    return new_states

def dynamics_dubins(state, action, dt, vrange=None):
    """
    Computes the new state of the Dubins car after applying a control action.
    Args:
        state (tuple): Current state of the Dubins car (x, y, theta).
        action (float): Angular velocity (control input in radians).
    Returns: tuple: New state (x, y, theta) after applying the action.
    """
    x, y, theta = state
    angular_velocity, a = action
    assert np.all(np.abs(a) < 1e-8), "Accelerations are not all near zero (threshold 1e-8)"
    assert vrange[0] == vrange[1], "For constant velocity, vrange[0] must equal vrange[1]"
    v = vrange[0]
    x_new = x + v * np.cos(theta) * dt 
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + angular_velocity * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi # wrap back into [−π,π]
    return (x_new, y_new, theta_new) 

def inverse_dynamics_dubins(start_state, target_state, dt):
    """
    Computes the angular velocity (action) and linear velocity required to move from a start state to a target state.

    Args:
        start_state (tuple): Current state of the Dubins car (x, y, theta).
        target_state (tuple): Desired target state of the Dubins car (x, y, theta).
        dt (float): Time step duration (seconds).

    Returns:
        tuple: (v, action) where:
               - v (float): Linear velocity required to reach the target position.
               - action (float): Angular velocity (action) required to match the target orientation.
    """

    x_start, y_start, theta_start = start_state
    x_target, y_target, theta_target = target_state
    # Compute the linear velocity required
    distance = np.sqrt((x_target - x_start) ** 2 + (y_target - y_start) ** 2)
    v = distance / dt
    delta_theta = angle_diff(theta_target, theta_start)
    action = delta_theta / dt
    return v, action

def vectorized_dynamics_dubins(states, actions, dt, vrange):
    """
    Computes the new states of the Dubins car after applying a set of control actions.

    Args:
        states (np.ndarray): Current states of the Dubins car, shape (n, 3) where n is the number of states.
        actions (np.ndarray): Angular velocities (control inputs in radians), shape (nu, 1).
        dt (float): Time step duration.
    Returns:
        np.ndarray: New states, shape (n * nu, 3) after applying each action to each state.
    """
    n = states.shape[0]   # Number of states
    nu = actions.shape[0] # Number of actions
    # Repeat states nu times along axis 0 to compute all possible action combinations
    states_repeated = np.repeat(states, nu, axis=0)  # Shape: (n * nu, 3)
    # Tile actions n times along axis 0 so that each state will execute every action
    actions_tiled = np.tile(actions, (n, 1))  # Shape: (n * nu, 1)
    # Extract x, y, theta from repeated states
    x, y, theta = states_repeated[:, 0], states_repeated[:, 1], states_repeated[:, 2]
    angular_velocities, accelerations = actions_tiled[:, 0], actions_tiled[:, 1] # acceleration should be 0 here
    assert np.all(np.abs(accelerations) < 1e-8), "Accelerations are not all near zero (threshold 1e-8)"
    assert vrange[0] == vrange[1], "For constant velocity, vrange[0] must equal vrange[1]"
    v = np.clip(accelerations, vrange[0], vrange[1])

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + angular_velocities.flatten() * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi # wrap back into [−π,π]

    # NOTE: it's recommended to use 64 bit precision for reduced accumulated rounding errors if memory allow
    new_states = np.stack((x_new, y_new, theta_new), axis=1).astype(np.float32) # shape (n * nu, 3)
    # new_states = np.stack((x_new, y_new, theta_new), axis=1)
    return new_states

########################################## KS MODELS BELOW ##########################################
########################################## KS MODELS BELOW ##########################################
########################################## KS MODELS BELOW ##########################################
# NOTE: there are multiple versions of KS models each with different assumptions and inputs
# Reference: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf?ref_type=heads
########################################## KS 3D STEERING ANGLE ##########################################
def dynamics_KS_3d_steering_angle(state, action, dt, vrange=None): #constant velocity
    x, y, theta = state
    steering_angle, a = action
    assert np.all(np.abs(a) < 1e-8), "Accelerations are not all near zero (threshold 1e-8)"
    assert vrange[0] == vrange[1], "For constant velocity, vrange[0] must equal vrange[1]"
    v = vrange[0]
    L_wb = 0.324 # wheelbase for F1Tenth
    theta_new = theta + v/L_wb * np.tan(steering_angle) * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi # wrap back into [−π,π]
    x_new = x + v * np.cos(theta_new) * dt 
    y_new = y + v * np.sin(theta_new) * dt
    # version below update xy first
    # x_new = x + v * np.cos(theta) * dt 
    # y_new = y + v * np.sin(theta) * dt
    # theta_new = theta + v/L_wb * np.tan(steering_angle) * dt
    return (x_new, y_new, theta_new) 

def inverse_dynamics_KS_3d_steering_angle(start_state, target_state, dt):
    """
    Given the start state (x, y, theta) and target state (x, y, theta),
    compute the required velocity (v) and target steering angle (action).
    Returns: 
        v (velocity), action (steering angle in radians)
    """
    # Extract start and target states
    x_start, y_start, theta_start = start_state
    x_target, y_target, theta_target = target_state

    # Compute the required velocity to move from (x_start, y_start) to (x_target, y_target)
    dx = x_target - x_start
    dy = y_target - y_start
    v = np.sqrt(dx**2 + dy**2) / dt

    # Compute the target steering angle (action) based on the change in theta
    L_wb = 0.324  # Wheelbase for F1Tenth
    delta_theta = angle_diff(theta_target, theta_start)
    # Compute the steering angle using the inverse dynamics relationship
    action = np.arctan((delta_theta * L_wb) / (v * dt))
    return v, action

def cuda_dynamics_KS_3d_steering_angle_vectorized(states, actions, dt, v):
    '''Apply vectorized dynamic propagation'''
    L_wb = 0.324  # Wheelbase for F1Tenth (distance between axles)
    n = states.shape[0]   # Number of states
    nu = actions.shape[0] # Number of actions

    # Repeat states to pair each state with every action
    states_repeated = states.repeat_interleave(nu, dim=0)  # Shape: (n * nu, 3)

    # Tile actions to match each state with all actions
    actions_tiled = actions.repeat(n)  # Shape: (n * nu,)

    # Extract current state components
    x = states_repeated[:, 0]    # Shape: (n * nu,)
    y = states_repeated[:, 1]    # Shape: (n * nu,)
    theta = states_repeated[:, 2]  # Shape: (n * nu,)

    # Update theta first for each state-action pair
    theta_new = theta + (v / L_wb) * torch.tan(actions_tiled) * dt  # Shape: (n * nu,)
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [−π, π]: 
    x_new = x + v * torch.cos(theta_new) * dt  # Shape: (n * nu,)
    y_new = y + v * torch.sin(theta_new) * dt  # Shape: (n * nu,)
    next_states = torch.stack([x_new, y_new, theta_new], dim=1)  # Shape: (n * nu, 3)
    return next_states

def cuda_dynamics_KS_4d_steering_angle_v_vectorized(states, actions, dt, vrange):
    '''
    Apply vectorized dynamic propagation for 4D kinematic model with variable velocity
    Args:
        states: torch.Tensor of shape (n, 4) - each row is [x, y, theta, v]
        actions: torch.Tensor of shape (nu, 2) - each row is [steering_angle, acceleration]
        dt: float - time step
        vrange: tuple - (v_min, v_max) velocity constraints
    Returns:
        torch.Tensor of shape (n * nu, 4) - next states
    '''
    L_wb = 0.324  # Wheelbase for F1Tenth (distance between axles)
    n = states.shape[0]   # Number of states
    nu = actions.shape[0] # Number of actions

    # Repeat states to pair each state with every action
    states_repeated = states.repeat_interleave(nu, dim=0)  # Shape: (n * nu, 4)

    # Tile actions to match each state with all actions
    actions_tiled = actions.repeat(n, 1)  # Shape: (n * nu, 2)

    # Extract current state components
    x = states_repeated[:, 0]      # Shape: (n * nu,)
    y = states_repeated[:, 1]      # Shape: (n * nu,)
    theta = states_repeated[:, 2]  # Shape: (n * nu,)
    v0 = states_repeated[:, 3]     # Shape: (n * nu,)

    # Extract action components
    steering_angles = actions_tiled[:, 0]  # Shape: (n * nu,)
    accelerations = actions_tiled[:, 1]    # Shape: (n * nu,)

    # Update velocity with acceleration and clip to bounds
    v_new = v0 + accelerations * dt
    v_clipped = torch.clamp(v_new, vrange[0], vrange[1])  # Shape: (n * nu,)

    # Update theta first for each state-action pair
    theta_new = theta + (v_clipped / L_wb) * torch.tan(steering_angles) * dt  # Shape: (n * nu,)
    theta_new = (theta_new + math.pi) % (2*math.pi) - math.pi # wrap into [−π, π]
    
    # Update position using new theta and clipped velocity
    x_new = x + v_clipped * torch.cos(theta_new) * dt  # Shape: (n * nu,)
    y_new = y + v_clipped * torch.sin(theta_new) * dt  # Shape: (n * nu,)
    
    # Stack the new states
    next_states = torch.stack([x_new, y_new, theta_new, v_clipped], dim=1)  # Shape: (n * nu, 4)
    return next_states

def vectorized_dynamics_KS_3d_steering_angle(states, actions, dt, vrange): # constant velocity
    # Ensure inputs are in float32
    states = states.astype(np.float32)
    actions = actions.astype(np.float32)
    jt = np.float32(dt)
    vrange = np.array(vrange, dtype=np.float32)
    # L_wb = a + b, a and b is distance from spring mass center of gravity to front axle and rear axle respectively
    L_wb = 0.324
    n = states.shape[0]   # Number of states
    nu = actions.shape[0] # Number of actions
    states_repeated = np.repeat(states, nu, axis=0)  # Shape: (n * nu, 3)
    # Tile actions n times along axis 0 so that each state will execute every action
    actions_tiled = np.tile(actions, (n, 1))         # Shape: (n * nu, 1)
    x, y, theta = (
        states_repeated[:, 0],
        states_repeated[:, 1],
        states_repeated[:, 2],
    )
    steering_angles, accelerations = actions_tiled[:, 0], actions_tiled[:, 1] # acceleration should be 0 here
    assert np.all(np.abs(accelerations) < 1e-8), "Accelerations are not all near zero (threshold 1e-8)"
    assert vrange[0] == vrange[1], "For constant velocity, vrange[0] must equal vrange[1]"
    v = np.clip(accelerations, vrange[0], vrange[1])

    theta_new = theta + (v / L_wb) * np.tan(steering_angles) * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi # wrap back into [−π,π]
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt

    # version below update xy first
    # x_new = x + v * np.cos(theta) * dt
    # y_new = y + v * np.sin(theta) * dt
    # theta_new = theta + (v / L_wb) * np.tan(steering_angles) * dt
    # Stack new states into an array with shape (n * nu, 3)
    new_states = np.stack((x_new, y_new, theta_new), axis=1).astype(np.float32)
    return new_states

########################################## KS 4D STEERING ANGLE and Velocity ##########################################
def dynamics_KS_4d_steering_angle_v(state, action, dt, vrange):
    x, y, theta, v0 = state # theta is yaw angle
    steering_angle, accleration = action
    L_wb = 0.324 # wheelbase for F1Tenth
    v_new = v0 + accleration * dt
    v_clipped = np.clip(v_new, vrange[0], vrange[1])
    theta_new = theta + v_clipped/L_wb * np.tan(steering_angle) * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    x_new = x + v_clipped * np.cos(theta_new) * dt 
    y_new = y + v_clipped * np.sin(theta_new) * dt
    return (x_new, y_new, theta_new, v_clipped) 

# this function can be further optimized using CPU/GPU parallelization
# but there is no need since this is not bottleneck and numpy is already efficient
def vectorized_dynamics_KS_4d_steering_angle_v(states, actions, dt, vrange):
    # Ensure inputs are in float32
    states = states.astype(np.float32)
    actions = actions.astype(np.float32)
    dt = np.float32(dt)
    vrange = np.array(vrange, dtype=np.float32)
    L_wb = 0.324
    n = states.shape[0]     # Number of states
    nu = actions.shape[0]   # Number of actions
    states_repeated = np.repeat(states, nu, axis=0)  # Shape: (n*nu, 4)
    actions_tiled = np.tile(actions, (n, 1))         # Shape: (n*nu, 2)
    # Extract components
    x, y, theta, v0 = (
        states_repeated[:, 0],
        states_repeated[:, 1],
        states_repeated[:, 2],
        states_repeated[:, 3],
    )
    steering_angles, accelerations = actions_tiled[:, 0], actions_tiled[:, 1]
    v_new = v0 + accelerations * dt
    v_clipped = np.clip(v_new, vrange[0], vrange[1])

    # Update position and heading
    theta_new = theta + (v_clipped / L_wb) * np.tan(steering_angles) * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi # wrap back into [−π,π]
    x_new = x + v_clipped * np.cos(theta_new) * dt
    y_new = y + v_clipped * np.sin(theta_new) * dt
    new_states = np.stack((x_new, y_new, theta_new, v_clipped), axis=1).astype(np.float32)
    return new_states

def inverse_dynamics_KS_4d_steering_angle_v_vectorized(start_states, target_states, dt, vrange):
    """
    Vectorized version of inverse_dynamics_KS_4d_steering_angle_v.
    
    Given arrays of start states (x, y, theta, v) and target states (x, y, theta, v),
    compute the required steering angles and accelerations for all pairs simultaneously.
    
    Args:
        start_states (np.ndarray): Array of shape (N, 4) with start states
        target_states (np.ndarray): Array of shape (N, 4) with target states  
        dt (float): Time step
        vrange (tuple): Velocity range (min, max)
        
    Returns:
        tuple: (steering_angles, accelerations) where each is array of shape (N,)
    """
    # Ensure inputs are numpy arrays
    start_states = np.asarray(start_states, dtype=np.float32)
    target_states = np.asarray(target_states, dtype=np.float32)
    dt = np.float32(dt)
    vrange = np.array(vrange, dtype=np.float32)
    
    # Extract state components (vectorized)
    x0, y0, theta0, v0 = start_states[:, 0], start_states[:, 1], start_states[:, 2], start_states[:, 3]
    xt, yt, thetat, vt = target_states[:, 0], target_states[:, 1], target_states[:, 2], target_states[:, 3]
    
    # 1) Infer speeds that match the planar displacements (vectorized)
    dx = xt - x0
    dy = yt - y0
    dist = np.hypot(dx, dy)
    v_req = dist / dt
    
    # 2) Compute accelerations to reach those speeds (vectorized)
    accelerations = (v_req - v0) / dt
    v_new = v0 + accelerations * dt
    v_clipped = np.clip(v_new, vrange[0], vrange[1])
    
    # 3) Recover steering from Δheading (vectorized)
    delta_theta = angle_diff(thetat, theta0)
    L_wb = 0.324
    
    # Handle division by zero cases (where v_clipped * dt ≈ 0)
    denominator = v_clipped * dt
    mask_nonzero = ~np.isclose(denominator, 0.0)
    
    steering_angles = np.zeros_like(delta_theta)
    steering_angles[mask_nonzero] = np.arctan2(
        delta_theta[mask_nonzero] * L_wb, 
        denominator[mask_nonzero]
    )
    # steering_angles remains 0.0 where mask_nonzero is False
    
    return steering_angles, accelerations

########################################## KS 3D V_CMD (Kinematic Single-Track, direct velocity command) ##########################################
# State:   (x, y, theta)              — 3D
# Action:  (steering_angle, v_cmd)    — 2D  [steering in radians, velocity in m/s]
# Model:   Semi-Implicit Euler (heading updated first, then position)
def dynamics_KS_3d_v_cmd(state, action, dt, vrange=None):
    """
    Single-state forward dynamics for the 3D Kinematic Single-Track model
    with direct velocity command input (no acceleration state).

    Args:
        state (tuple): (x, y, theta) — position and yaw angle.
        action (tuple): (steering_angle [rad], v_cmd [m/s])
        dt (float): time step in seconds.
        vrange (tuple, optional): (v_min, v_max) — used to clip v_cmd.

    Returns:
        tuple: (x_new, y_new, theta_new)
    """
    x, y, theta = state
    steering_angle, v_cmd = action
    L_wb = 0.324  # F1Tenth wheelbase (m)
    if vrange is not None:
        v_cmd = np.clip(v_cmd, vrange[0], vrange[1])
    # Semi-Implicit Euler: update heading first, then integrate position with new heading
    theta_new = theta + (v_cmd / L_wb) * np.tan(steering_angle) * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi  # wrap into [-pi, pi]
    x_new = x + v_cmd * np.cos(theta_new) * dt
    y_new = y + v_cmd * np.sin(theta_new) * dt
    return (x_new, y_new, theta_new)


def vectorized_dynamics_KS_3d_v_cmd(states, actions, dt, vrange=None):
    """
    Vectorized forward dynamics for the 3D KS model with direct velocity command.
    Computes next states for every (state, action) pair in the Cartesian product.

    Args:
        states  (np.ndarray): shape (n, 3)  — each row is (x, y, theta).
        actions (np.ndarray): shape (nu, 2) — each row is (steering_angle [rad], v_cmd [m/s]).
        dt (float): time step in seconds.
        vrange (tuple, optional): (v_min, v_max) — used to clip v_cmd.

    Returns:
        np.ndarray: shape (n * nu, 3) — next states for all state-action pairs.
    """
    states  = states.astype(np.float32)
    actions = actions.astype(np.float32)
    dt      = np.float32(dt)
    L_wb    = np.float32(0.324)

    n  = states.shape[0]
    nu = actions.shape[0]

    states_repeated = np.repeat(states,  nu, axis=0)  # (n*nu, 3)
    actions_tiled   = np.tile(actions, (n,  1))        # (n*nu, 2)

    x     = states_repeated[:, 0]
    y     = states_repeated[:, 1]
    theta = states_repeated[:, 2]

    steering_angles = actions_tiled[:, 0]  # radians
    v_cmd           = actions_tiled[:, 1]  # m/s

    if vrange is not None:
        v_cmd = np.clip(v_cmd, vrange[0], vrange[1])

    # Semi-Implicit Euler: heading first
    theta_new = theta + (v_cmd / L_wb) * np.tan(steering_angles) * dt
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi  # wrap into [-pi, pi]
    x_new     = x + v_cmd * np.cos(theta_new) * dt
    y_new     = y + v_cmd * np.sin(theta_new) * dt

    return np.stack((x_new, y_new, theta_new), axis=1).astype(np.float32)


def inverse_dynamics_KS_3d_v_cmd(start_state, target_state, dt, vrange=None):
    """
    Inverse dynamics for the 3D KS model with direct velocity command.
    Given a start and target state, recovers the (steering_angle, v_cmd) action
    that would drive the model from start to target in one step.

    Args:
        start_state  (array-like): (x0, y0, theta0)
        target_state (array-like): (xt, yt, thetat)
        dt (float): time step in seconds.
        vrange (tuple, optional): (v_min, v_max) — for clipping recovered v_cmd.

    Returns:
        tuple: (steering_angle [rad], v_cmd [m/s])
    """
    x0, y0, theta0   = start_state[0], start_state[1], start_state[2]
    xt, yt, thetat   = target_state[0], target_state[1], target_state[2]
    L_wb = 0.324

    # Recover velocity command from Euclidean displacement
    dist  = np.hypot(xt - x0, yt - y0)
    v_cmd = dist / dt
    if vrange is not None:
        v_cmd = np.clip(v_cmd, vrange[0], vrange[1])

    # Recover steering angle from heading change
    delta_theta  = angle_diff(thetat, theta0)
    denominator  = v_cmd * dt
    if np.abs(denominator) < 1e-6:
        steering_angle = 0.0
    else:
        steering_angle = np.arctan2(delta_theta * L_wb, denominator)

    return steering_angle, v_cmd


def inverse_dynamics_KS_3d_v_cmd_vectorized(start_states, target_states, dt, vrange=None):
    """
    Vectorized inverse dynamics for the 3D KS model with direct velocity command.

    Args:
        start_states  (np.ndarray): shape (N, 3) — start states.
        target_states (np.ndarray): shape (N, 3) — target states.
        dt (float): time step in seconds.
        vrange (tuple, optional): (v_min, v_max) — for clipping recovered v_cmd.

    Returns:
        tuple: (steering_angles [rad], v_cmds [m/s]) — each shape (N,).
    """
    start_states  = np.asarray(start_states,  dtype=np.float32)
    target_states = np.asarray(target_states, dtype=np.float32)
    dt = np.float32(dt)
    L_wb = np.float32(0.324)

    dx   = target_states[:, 0] - start_states[:, 0]
    dy   = target_states[:, 1] - start_states[:, 1]
    dist = np.hypot(dx, dy)

    v_cmds = dist / dt
    if vrange is not None:
        v_cmds = np.clip(v_cmds, vrange[0], vrange[1])

    delta_theta  = angle_diff(target_states[:, 2], start_states[:, 2])
    denominators = v_cmds * dt

    mask_nonzero   = ~np.isclose(denominators, 0.0)
    steering_angles = np.zeros_like(delta_theta)
    steering_angles[mask_nonzero] = np.arctan2(
        delta_theta[mask_nonzero] * L_wb,
        denominators[mask_nonzero]
    )

    return steering_angles, v_cmds


########################################## 2D RANDOM WALK MODEL BELOW ##########################################
########################################## 2D RANDOM WALK MODEL BELOW ##########################################
########################################## 2D RANDOM WALK MODEL BELOW ##########################################
def dynamics_2d_random_walk(state, action, dt, vrange=None):
    # Dynamic when input state is a tuple
    dy, a = action
    steering_angle, a = action
    assert np.all(np.abs(a) < 1e-8), "Accelerations are not all near zero (threshold 1e-8)"
    assert vrange[0] == vrange[1], "For constant velocity, vrange[0] must equal vrange[1]"
    v = vrange[0]
    p_x, p_y = state
    p_x_new = p_x + v * dt
    p_y_new = p_y + dy * dt
    return (p_x_new, p_y_new)

def vectorized_dynamics_2D_Walk(states, actions, dt, vrange):
    """
    Computes the new states of the 2D Walk after applying a set of control actions.
    Args:
        states (np.ndarray): Current states of the robots, shape (n, 2), where n is the number of states.
        actions (np.ndarray): Discrete actions, shape (nu, 1), where nu is the number of actions.
        dt (float): Time step duration.

    Returns:
        np.ndarray: New states, shape (n * nu, 2) after applying each action to each state.
    """
    n = states.shape[0]  # Number of states
    nu = actions.shape[0]  # Number of actions

    # Repeat states nu times along axis 0 to compute all possible action combinations
    states_repeated = np.repeat(states, nu, axis=0)  # Shape: (n * nu, 2)

    # Tile actions n times along axis 0 so that each state will execute every action
    actions_tiled = np.tile(actions, (n, 1))  # Shape: (n * nu, 1)

    # Extract p_x and p_y from repeated states
    p_x, p_y = states_repeated[:, 0], states_repeated[:, 1]
    steering_angles, accelerations = actions_tiled[:, 0], actions_tiled[:, 1] # acceleration should be 0 here
    assert np.all(np.abs(accelerations) < 1e-8), "Accelerations are not all near zero (threshold 1e-8)"
    assert vrange[0] == vrange[1], "For constant velocity, vrange[0] must equal vrange[1]"
    v = np.clip(accelerations, vrange[0], vrange[1])

    # Compute new p_x and p_y positions
    p_x_new = p_x + v * dt
    p_y_new = p_y + steering_angles.flatten() * dt

    # Stack new states into an array with shape (n * nu, 2)
    new_states = np.stack((p_x_new, p_y_new), axis=1).astype(np.float32)
    return new_states

def inverse_dynamics_2d_random_walk(start_state, target_state, dt):
    return (target_state[1] - start_state[1]) / dt 