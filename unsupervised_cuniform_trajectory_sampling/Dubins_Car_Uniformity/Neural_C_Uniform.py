import pickle
import torch
from classes.grid import *
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)  # This sets the seed for all GPUs
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import numpy as np
import os
import random
import scipy 
import math
import torch.distributions as dist
from Model import *

with open("/home/isleri/mahes092/C_Uniform/Datasets/C_Uniform_processed_disjoint_KS_3D_STEERING_ANGLE_perturb_2.01_slack_0.0_seed_2025_grid_0.03_0.03_6.00deg_t3.01_ts0.2vrange_1.0_1.0_steering_13.pkl", "rb") as f:
    data = pickle.load(f)

## Getting the velocity from the config file
velocity = data["config"]["vrange"][0]
## Getting the actions from the config file
actions = torch.tensor(np.linspace(data["config"]["actions"][0,0], data["config"]["actions"][-1,0], 45)).cuda().to(torch.float32)
## Creating a grid object
g = Grid(data["config"]["thresholds"])
## Saving folder
experiment_folder = "/home/isleri/mahes092/C_Uniform/Dubins_Car_Uniformity/Models/Kinematic_Unsupervised_All_Weight_Perturbation_Latest_Finer_Action/"
## Creating model object
model_t =MLP().cuda()
## Creating lists for saving the dataset and representatives
dataset = list()
level_set_t1 = list()
representative_list = list()
## Initilizing the test set for different experiments
test = []
## Iterating through the level sets
for time in range(1, len(data["reachable_representative_across_LS"])):
    level_set_states = list()
    rep_list = list()
    reach_rep_list = list(data["reachable_representative_across_LS"])
    for each_state in range(len(reach_rep_list[time])):
        current_state = np.array(list(reach_rep_list[time])[each_state])
        ## Get the level set information only if not the last level set
        if time != len(data["reachable_representative_across_LS"]) - 1:
            level_set_states.append(current_state)
        ## Getting the representatives for all level sets
        rep_list.append(torch.tensor(current_state))
    ## Adding perturbed points for having all level sets having same number of representatives
    if time != len(data["reachable_representative_across_LS"]) - 1:
        added_perturbation_total_number = len(list(data["reachable_representative_across_LS"])[-2])
        additional_points = added_perturbation_total_number - len(level_set_states)
        if additional_points > 0:
            total_number_of_points_per_cell = int(round(additional_points/len(level_set_states)))
            for each_point in range(len(level_set_states)):
                perturbed_points = g.perturb_state(level_set_states[each_point], division_factor=2.01, num_samples=total_number_of_points_per_cell)
                for each_point_per in range(len(perturbed_points)):
                    level_set_states.append(perturbed_points[each_point_per])
    ## Gathering level set dataset
    if time != len(data["reachable_representative_across_LS"]) - 1:
        level_set_t1.append(level_set_states)
    ## Gathering representative dataset
    representative_list.append(rep_list)

def dynamics_KS_3d_steering_angle(state, action, dt = 0.2, v = 1):
    '''
    F1 Tenth dynamic model Pytorch
    '''
    L_wb = 0.324 # wheelbase for F1Tenth
    ## Updating theta
    theta_new = state[:,2].unsqueeze(1) + v/L_wb * torch.tan(action) * dt  # action represents angular velocity
    theta_new = theta_new.flatten().unsqueeze(1)
    ## Updating X and Y position
    x_new = state[:,0] + v * torch.cos(state[:,2]) * dt
    y_new = state[:,1] + v * torch.sin(state[:,2]) * dt
    repeated_x_new = x_new.repeat_interleave(len(action)).unsqueeze(1)
    repeated_y_new = y_new.repeat_interleave(len(action)).unsqueeze(1)
    return torch.hstack((repeated_x_new, repeated_y_new, theta_new))

def count_updation(current_level_set_entropy, probability, representative_t_1, actions, real_states_batch, velocity):
    '''
    Estimating the count of next states from the current level set
    '''
    ## Determining the next states
    next_states = dynamics_KS_3d_steering_angle(real_states_batch, actions, v = velocity)
    ## Converting theta into sin and cos components for distance calculation
    next_states_expanded = torch.hstack((next_states[:,0:2], torch.sin(next_states[:,2]).unsqueeze(1), torch.cos(next_states[:,2]).unsqueeze(1)))
    representative_expanded = torch.hstack((representative_t_1[:,0:2], torch.sin(representative_t_1[:,2]).unsqueeze(1), torch.cos(representative_t_1[:,2]).unsqueeze(1)))
    next_states_occurence = probability.reshape(-1).unsqueeze(1)
    ## Determining the distance based occurences
    ## 20 is the weighing factor for tradeoff between hard assignment and sof assignment
    decayed_distance = next_states_occurence * torch.exp(-20*torch.cdist(next_states_expanded, representative_expanded, p=2))
    ## Row Sum to get the occurences for each bin
    probability = torch.sum(decayed_distance, axis = 0)
    ## Adding to existing count
    current_level_set_entropy += probability
    return current_level_set_entropy

def entropy_maximization(current_level_set_entropy):
    '''
    Entropy estimation from the occurences
    1. Converting into probability distribution
    2. Calculating the Entropy
    '''
    ## Normalization to convert the occurences to probability distribtion
    probability = current_level_set_entropy / torch.sum(current_level_set_entropy)
    ## Calcualting entropy
    entropy_loss = torch.sum(probability * torch.log(probability + 1e-8)).mean()
    return entropy_loss

## Optimizer declaration
optimizer = torch.optim.Adam(model_t.parameters(), lr=1e-4)  
## Number of epochs
iterations = 20
## Early stopping
best_loss = None
## Calculating the maximum entropy which is the uniform distribution
max_entropy_rep = np.array([1/len(representative_list[-1]) for i in range(len(representative_list[-1]))])
max_entropy_weight = -np.sum(max_entropy_rep * np.log(max_entropy_rep))
## Training Loop
for epoch in range(iterations+1):
    running_loss = 0.0
    running_entropy = 0.0
    model_t.train()
    ## Parse through all level sets
    for time in range(len(level_set_t1)):
        if time not in test:
            representative_t_1 = torch.stack(representative_list[time + 1]).cuda().to(torch.float32)
            max_entropy = np.array([1/len(representative_t_1) for i in range(len(representative_t_1))])
            max_entropy_lev = -np.sum(max_entropy * np.log(max_entropy))
            print("Max Extropy", -np.sum(max_entropy * np.log(max_entropy)))
            data_loader = DataLoader(level_set_t1[time], batch_size=64, shuffle=True, num_workers=6)
            level_set_entropy = torch.zeros((1, len(representative_t_1))).cuda()
            ## Determining the weight for each level set so that updates of each level equal affect
            weight = (max_entropy_weight/max_entropy_lev)
            ## Parsing through the level set
            for i, data_1 in enumerate(data_loader, 0):
                optimizer.zero_grad()
                states_batch = data_1
                states = states_batch.cuda().to(torch.float32)
                if states.shape[0] == 1:
                    states = states.repeat(2, 1)
                network_states = torch.hstack((states[:,0:2], torch.sin(states[:,2]).unsqueeze(1), torch.cos(states[:,2]).unsqueeze(1))).cuda()
                ## Determining the action probabilities
                pred_probability_distribution = model_t(network_states)
                ## Updating the count
                level_set_entropy = count_updation(level_set_entropy, pred_probability_distribution, representative_t_1, actions, states, velocity = velocity)
                ## Calcualting the entropy and multiplying it with weight
                loss = weight * entropy_maximization(level_set_entropy)
                ## Backward pass
                loss.backward()
                optimizer.step()
                ## Keeping the exisiting count of the occurences for the next batch
                level_set_entropy = level_set_entropy.detach()
            ## Storing the statistics
            running_loss += loss.item()
            running_entropy += loss.item()
            total_number_of_batches = i
            print("Model: " + str(time) + " Epoch: " + str(epoch) + " Level Set " + str(time)+ " Entropy Loss: " + str(loss.item()))
    with open(experiment_folder + 'training.log', 'a+') as f:
        f.write("Model: " + str(time) + " Epoch: " + str(epoch) + " Loss: " + str(running_loss) + " Entropy Loss: " + str(running_entropy) + "\n")
    ## Saving the best model
    if best_loss is not None:
        if (running_loss) < best_loss:
            torch.save(model_t, os.path.join(experiment_folder, "best_model.pt"))
            best_loss = running_loss
    else:
        torch.save(model_t, os.path.join(experiment_folder, "best_model.pt"))
        best_loss = running_loss
