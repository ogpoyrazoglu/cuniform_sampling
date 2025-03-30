import numpy as np

# helper functions for robot dynamics
def angle_diff(theta_target, theta_start):
# minimal difference between two angles in radians, wrapped to the interval [-pi, pi].
# when computing (theta_target-theta_start) raw difference can be large in magnitude instead of the minimal rotation
    diff = theta_target - theta_start
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

########################################## DUBINS MODEL BELOW ##########################################
########################################## DUBINS MODEL BELOW ##########################################
########################################## DUBINS MODEL BELOW ##########################################
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
    # updating theta first
    theta_new = theta + angular_velocity * dt  # action represents angular velocity
    x_new = x + v * np.cos(theta_new) * dt 
    y_new = y + v * np.sin(theta_new) * dt

    # updating xy first
    # x_new = x + v * np.cos(theta) * dt 
    # y_new = y + v * np.sin(theta) * dt
    # theta_new = theta + angular_velocity * dt
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

    theta_new = theta + angular_velocities.flatten() * dt
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt
    # x_new = x + v * np.cos(theta) * dt
    # y_new = y + v * np.sin(theta) * dt
    # theta_new = theta + angular_velocities.flatten() * dt

    new_states = np.stack((x_new, y_new, theta_new), axis=1).astype(np.float32) # shape (n * nu, 3)
    return new_states

########################################## 2D RANDOM WALK MODEL BELOW ##########################################
########################################## 2D RANDOM WALK MODEL BELOW ##########################################
########################################## 2D RANDOM WALK MODEL BELOW ##########################################
def dynamics_2d_random_walk(state, action, dt, vrange=None):
    # Dynamic when input state is a tuple
    dy, a = action
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
    x_new = x + v * np.cos(theta) * dt 
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + v/L_wb * np.tan(steering_angle) * dt
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

def vectorized_dynamics_KS_3d_steering_angle(states, actions, dt, vrange): # constant velocity
    # Ensure inputs are in float32
    states = states.astype(np.float32)
    actions = actions.astype(np.float32)
    dt = np.float32(dt)
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

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + (v / L_wb) * np.tan(steering_angles) * dt
    # Stack new states into an array with shape (n * nu, 3)
    new_states = np.stack((x_new, y_new, theta_new), axis=1).astype(np.float32)
    return new_states