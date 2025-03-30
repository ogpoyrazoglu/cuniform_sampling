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
import torch.distributions as dist
import copy
import matplotlib.pyplot as plt

        
def dynamics_random_walk(state, action):
    """
    Computes the new state of the Dubins car after applying a control action.

    Args:
        state (tuple): Current state of the Dubins car (x, y, theta).
        action (float): Angular velocity (control input in radians).

    Returns:
        tuple: New state (x, y, theta) after applying the action.
    """
    return state + action

with open("/home/isleri/mahes092/C_Uniform/Datasets/x_u_pair_na3_ni30_manually_assigned_probability.pkl", "rb") as f:
    data = pickle.load(f)
actions = np.array([[1,-1],[1,0],[1,1]])
level_sets = list()
bins_level_set = list()
level_sets_probabilities = list()
level_sets_actions = list()
print(len(data))
for time in range(len(data)):
    level_set_t1 = list()
    level_set_actions = list()
    level_set_probabilities = list()
    bins = {}
    for each_state in data[time].keys():
        current_state = np.array(each_state)
        level_set_t1.append(current_state)
        level_set_actions.append(actions)
        for each_action in range(actions.shape[0]):
            action = actions[each_action]
            next_state = dynamics_random_walk(current_state, action)
            bins[tuple(next_state)] = 0
        level_set_probabilities.append(data[time][each_state])
    level_sets_probabilities.append(level_set_probabilities)
    level_sets_actions.append(level_set_actions)
    level_sets.append(level_set_t1)
    bins_level_set.append(bins)

def calculate_entropy_from_bins(bin_counts):
    """
    Calculate the entropy from bin counts.

    Parameters:
        bin_counts (dict): A dictionary with keys as bin numbers and values as counts.

    Returns:
        float: The calculated entropy.
    """
    # Extract the counts from the dictionary
    counts = np.array(list(bin_counts.values()))
    # total_count = np.sum(counts)
    # probabilities = counts / total_count
    mean_of_counts = np.mean(counts)
    adjusted_counts = counts - mean_of_counts
    std_dev = np.std(adjusted_counts)
    return adjusted_counts

actions = torch.tensor(actions).cuda()
for each in range(len(level_sets)):
    current_level_set = level_sets[each]
    current_actions = level_sets_actions[each]
    current_probabilities = level_sets_probabilities[each]
    next_level_set_bins = bins_level_set[each]
    perturbed_states = torch.tensor(np.array(current_level_set)).cuda().to(torch.float32)
    for each_state in range(len(current_probabilities)):
        current_probability_distribution = np.array(current_probabilities[each_state])
        current_action_distribution = np.array(current_actions[each_state])
        current_state = perturbed_states[each_state,:].unsqueeze(0)
        num_samples = 5000
        current_probability_distribution = current_probability_distribution / np.sum(current_probability_distribution)
        next_states = dynamics_random_walk(current_state, actions[:,:]).detach().cpu().numpy()
        for each_next_state in range(next_states.shape[0]):
            next_level_set_bins[tuple(next_states[each_next_state])] += 1*current_probability_distribution[each_next_state]
    bins_level_set[each] = next_level_set_bins

entropy_level_set = list()
unformity_level_set = list()
level_set_index = list()
for each_level_set_bin in range(len(bins_level_set)):
    current_level_set_bins = bins_level_set[each_level_set_bin]
    level_set_std = calculate_entropy_from_bins(current_level_set_bins)

    level_set_index.append(each_level_set_bin + 1)
    entropy_level_set.append(level_set_std)

plt.clf()

plt.figure(figsize=(10, 6))  # Adjust figure size if needed
# plt.boxplot(entropy_level_set, positions=level_set_index, patch_artist=True, 
#             boxprops=dict(facecolor="lightblue", color="blue"), 
#             medianprops=dict(color="red"))

plt.boxplot(
    entropy_level_set, 
    positions=level_set_index, 
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    showmeans=True,  # Display the mean
    meanprops=dict(marker='o', markerfacecolor='green', markeredgecolor='black')
)

# Add labels, title, and ticks
plt.xlabel('Level Set Index', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('Box Plots of Distributions')
plt.xticks(level_set_index)  # Ensure all distributions are labeled on x-axis
plt.tight_layout()

# Save or show the plot
plt.savefig("box_plots_distributions_manual.png")
plt.show()