import pickle 
import numpy as np
from classes.grid import Grid
import math
import torch.nn as nn
import torch
import copy
from Model import *
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)  # This sets the seed for all GPUs
np.random.seed(0)

def sample_trajectories_representative(grid, initial_state, actions, dynamics, t_step, num_trajectories, trajectory_length, v, model):
    """
    Sample multiple trajectories, using different action probabilities for each time step.

    Args:
        grid (Grid): The grid object used to discretize the state space.
        initial_state (tuple): The starting state of the system (x, y, theta).
        action_probabilities_list (list): List of dictionaries mapping grid indices to action probability distributions for each time step.
        actions (list): List of discretized actions available for the system.
        dynamics (function): The function defining the robot's dynamics.
        t_step (float): The time step for the robot dynamics.
        num_trajectories (int): The number of trajectories to generate.
        trajectory_length (int): The maximum number of steps for each trajectory.

    Returns:
        list: A list of sampled trajectories, where each trajectory is a list of (state, action) tuples.
    """
    all_trajectories = []  # To store all generated trajectories

    for _ in range(num_trajectories):
        trajectory = []  # Initialize an empty list for the current trajectory
        current_state = initial_state # Start from the initial state

        for step in range(trajectory_length):
            ## Determining the action probabilities
            perturbed_states_theta_x = torch.sin(torch.tensor(current_state[2]))
            perturbed_states_theta_y = torch.cos(torch.tensor(current_state[2]))
            network_current_state = torch.hstack((torch.tensor(current_state[0:2]), perturbed_states_theta_x.unsqueeze(0), perturbed_states_theta_y.unsqueeze(0))).to(torch.float32).cuda()
            if step == 0:
                ## Uniform actions for 0th level set
                chosen_action_index = np.random.choice(range(len(actions)))
            else:
                ## Sampling using action probabilities
                current_probability_distribution = model(network_current_state.unsqueeze(0)).detach().cpu().numpy()[0]
                chosen_action_index = np.random.choice(range(len(actions)), p=current_probability_distribution)
            ## Chosen action
            chosen_action = actions[chosen_action_index]
            ## Putting it inside the trajectory generation list
            trajectory.append((current_state, chosen_action))
            ## Updating the state using F1tenth dynamics
            new_state = dynamics(current_state, chosen_action, t_step, v)
            current_time = step * t_step
            current_state = copy.deepcopy(new_state)
        trajectory.append((current_state, None))
        all_trajectories.append(trajectory)
    return all_trajectories 

def dynamics_KS_3d_steering_angle(state, action, dt, v = 1, vrange=None): #constant velocity
    '''
    F1tenth dynamics update
    '''
    x, y, theta = state
    steering_angle= action
    L_wb = 0.324 # wheelbase for F1Tenth
    ## Updating the x and y 
    x_new = x + v * np.cos(theta) * dt 
    y_new = y + v * np.sin(theta) * dt
    ## Updating the theta
    theta_new = theta + v/L_wb * np.tan(steering_angle) * dt
    return (x_new, y_new, theta_new) 

## Getting the config file
with open("/home/isleri/mahes092/C_Uniform/Datasets/C_Uniform_processed_disjoint_KS_3D_STEERING_ANGLE_perturb_2.01_slack_0.0_seed_2025_grid_0.03_0.03_6.00deg_t3.01_ts0.2vrange_1.0_1.0_steering_13.pkl", "rb") as f:
    data = pickle.load(f)
## Declaring initial state
initial_state = np.zeros(3).astype(np.float32)
## Determining the actions using the config file
actions = np.linspace(data["config"]["actions"][0,0], data["config"]["actions"][-1,0], 13)
## Declaring the velocity, time and time discretization
v=1
lookback_length = 3.01
t_step = 0.2
## Getting the grid and thresholds through the config
thresholds = data["config"]["thresholds"]
g = Grid(thresholds=thresholds)
## Creating the trained model instance
model = torch.load("/home/isleri/mahes092/C_Uniform/Dubins_Car_Uniformity/Models/Kinematic_Unsupervised_All_Weight_Perturbation_Latest/best_model.pt")
model.eval()   
## Declaring the number of trajectories
num_trajectories_list = [1000]      
for num_trajectories in num_trajectories_list:
    ## Generating and saving the trajectories
    sampled_trajectories_representative = sample_trajectories_representative(
        grid=g,
        initial_state=initial_state,
        actions=actions,
        dynamics=dynamics_KS_3d_steering_angle,
        # dynamics=dynamics_bicycle,
        t_step=t_step,
        num_trajectories=num_trajectories,
        trajectory_length=int(lookback_length / t_step),
        v=v,
        model = model
    )
    filename = f"representative_KS_perturb_1000_0.2_3.01_13.pickle"
    with open(filename, 'wb') as file:
        pickle.dump(sampled_trajectories_representative, file)
    print(f"Sampled {num_trajectories} trajectories and saved to {filename}.")
