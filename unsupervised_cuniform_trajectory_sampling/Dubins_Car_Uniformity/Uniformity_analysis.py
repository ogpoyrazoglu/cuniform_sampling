import pickle
import torch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)  # This sets the seed for all GPUs
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import scipy
import math
from scipy.stats import entropy
import torch.distributions as dist
import copy
import matplotlib.pyplot as plt
from classes.grid import *
from Model import *

def dynamics_KS_3d_steering_angle(state, action, dt = 0.2, v = 1):
    '''
    F1tenth dynamics
    '''
    x, y, theta = state[0], state[1], state[2]
    L_wb = 0.324 # wheelbase for F1Tenth
    ## Updating X,Y
    x_new = x + v * np.cos(theta) * dt 
    y_new = y + v * np.sin(theta) * dt
    ## Updating theta
    theta_new = theta + (v/L_wb) * np.tan(action) * dt
    return (x_new, y_new, theta_new) 

## Getting the config file
with open("/home/isleri/mahes092/C_Uniform/Datasets/C_Uniform_processed_disjoint_KS_3D_STEERING_ANGLE_perturb_2.01_slack_0.0_seed_2025_grid_0.03_0.03_6.00deg_t3.01_ts0.2vrange_1.0_1.0_steering_13.pkl", "rb") as f:
    data = pickle.load(f)

## Creating the model instance
model = torch.load("/home/isleri/mahes092/C_Uniform/Dubins_Car_Uniformity/Models/Kinematic_Unsupervised_All_Weight_Perturbation_Latest/best_model.pt")
model.eval()

## Getting the thresholds and create grid object
thresholds = data["config"]["thresholds"]
g = Grid(thresholds=thresholds)

## Creating the lists for uniformity analysis
level_set_t1 = list()
representative_list = list()
## Getting the representatives
for time in range(len(data["reachable_representative_across_LS"])):
    level_set_states = list()
    reach_rep_list = list(data["reachable_representative_across_LS"])
    level_set_bins = {}
    for each_state in range(len(reach_rep_list[time])):
        current_state = np.array(list(reach_rep_list[time])[each_state])
        if time != len(data["reachable_representative_across_LS"])- 1:
            level_set_states.append(current_state)
        ## Initializing the count as 0
        level_set_bins[g.get_index(current_state)] = 0
    if time != len(data["reachable_representative_across_LS"]) - 1:
        level_set_t1.append(level_set_states)
    representative_list.append(level_set_bins)

for each in range(len(level_set_t1)):
    current_level_set = level_set_t1[each]
    perturbed_states = torch.tensor(np.array(current_level_set)).cuda().to(torch.float32)
    next_level_set_bins = representative_list[each + 1]
    final_network_current_state = torch.hstack((perturbed_states[:,0:2], torch.sin(perturbed_states[:,2]).unsqueeze(1), torch.cos(perturbed_states[:,2]).unsqueeze(1))).to(torch.float32).cuda()
    ## Getting the action probability for each state
    current_probability_distribution_all = model(final_network_current_state)
    for each_state in range(len(current_level_set)):
        ## Treating 0th level set as special case for uniform probabilities
        if each == 0:
            current_probability_distribution = (torch.ones((final_network_current_state.shape[0],13))/13).detach().cpu().numpy()[0]
        else:
            current_probability_distribution = current_probability_distribution_all[each_state,:].detach().cpu().numpy()
        ## Calculating next states
        next_states = list()
        actions = data["config"]["actions"][:,0]
        for each_action in range(actions.shape[0]):
            next_state = dynamics_KS_3d_steering_angle(current_level_set[each_state], actions[each_action], dt = 0.2)
            next_states.append(next_state)
        ## Calcuating the count of the next representatives
        for each_next_state in range(len(next_states)):
            if g.get_index(next_states[each_next_state]) not in next_level_set_bins:
                pass
            else:
                next_level_set_bins[g.get_index(next_states[each_next_state])] += 1*current_probability_distribution[each_next_state]
    representative_list[each + 1] = next_level_set_bins

unformity_level_set_nn = list()
uniformity_uniform = list()
for each_level_set_bin in range(1, len(representative_list)):
    uniformity_measure_level_set_nn = list()
    current_level_set_bins = representative_list[each_level_set_bin]
    current_counts = np.array(list(current_level_set_bins.values()))
    ## Determine the entropy based on current determined counts
    current_counts = current_counts/np.sum(current_counts)
    model_entropy = entropy(current_counts, base=2)
    ## Assume uniform count and calculate maximum entropy
    uniformity_entropy = entropy(np.ones(len(list(current_level_set_bins.values())))/len(list(current_level_set_bins.values())), base=2)
    ## Determining the percentage uniformity based on the entropy ratio
    unformity_level_set_nn.append((model_entropy/uniformity_entropy)*100)

## Plotting the uniformity %
bar_width = 0.4
x = np.arange(1, len(representative_list))

# Create bar plots
plt.bar(x, unformity_level_set_nn, width=bar_width, label='NN', color='blue', alpha=0.7)
# Add labels, title, and legend
# Set y-axis increments to 5
y_min = 0
y_max = 100  # Adjust based on your data range
plt.yticks(np.arange(y_min, y_max + 1, 5))  # Increment by 5

plt.xlabel('Level Set')
plt.ylabel('Entropy % Uniformity')
plt.legend()
# Show the plot
plt.tight_layout()
plt.savefig("quantitative_analysis_NN_kinemtic_weight_perturb_latest.png")