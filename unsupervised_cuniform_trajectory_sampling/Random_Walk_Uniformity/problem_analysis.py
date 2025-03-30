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


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 256, bias=True)
        self.bn_1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.bn_2 = nn.BatchNorm1d(256)

        self.out = nn.Linear(256, 3, bias=True)
        self.act = nn.ReLU()

    def forward(self, x_input):
        x = self.bn_1(self.act(self.fc1(x_input)))
        x = self.bn_2(self.act(self.fc2(x)))
        x = self.out(x)
        return x


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

with open("/home/isleri/mahes092/C_Uniform/Datasets/x_u_pair_na3_ni30_max_flow_probability.pkl", "rb") as f:
    data = pickle.load(f)
model = torch.load("/home/isleri/mahes092/C_Uniform/Random_Walk_Uniformity/NN_Random_Walk_All/single_probability_model_best.pt").cuda()
model.eval()
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
    return counts

actions = torch.tensor(actions).cuda()
for each in range(len(level_sets)):
    if each == 1:
        current_level_set = level_sets[each]
        current_actions = level_sets_actions[each]
        current_probabilities = level_sets_probabilities[each]
        next_level_set_bins = bins_level_set[each]

        perturbed_states = torch.tensor(np.array(current_level_set)).cuda().to(torch.float32)
        for each_state in range(len(current_probabilities)):
            current_probability_distribution = np.array(current_probabilities[each_state])
            current_action_distribution = np.array(current_actions[each_state])
            current_state = perturbed_states[each_state,:].unsqueeze(0)
            plt.clf()
            plt.bar([-1,0,1], current_probability_distribution)
            plt.xlabel('Actions')
            plt.ylabel('Probability')
            plt.title('probability NF')
            plt.ylim(0, 1)
            plt.savefig("NF_" + str(perturbed_states[each_state,:]) + ".png")
            num_samples = 5000
            current_probability_distribution = nn.Softmax(dim = 1)(model(current_state))
            plt.clf()
            plt.bar([-1,0,1], current_probability_distribution.detach().cpu().numpy()[0])
            plt.xlabel('Actions')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            plt.title('probability NN')
            plt.savefig("NN_" + str(perturbed_states[each_state,:]) + ".png")
            next_states = dynamics_random_walk(current_state[:,0:2], actions[:,:]).detach().cpu().numpy()

