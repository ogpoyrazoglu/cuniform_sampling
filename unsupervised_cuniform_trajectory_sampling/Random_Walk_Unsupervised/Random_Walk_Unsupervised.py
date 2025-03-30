import pickle
import torch
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
with open("/home/isleri/mahes092/C_Uniform/Datasets/x_u_pair_na3_ni30_max_flow_probability.pkl", "rb") as f:
    data = pickle.load(f)

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
        x = nn.Softmax(dim = 1)(self.out(x))
        return x

print()

actions = torch.tensor(np.array([[1,-1],[1,0],[1,1]])).cuda()
action_dim = 2
state_dim = 4
hidden_num = 512
experiment_folder = "/home/isleri/mahes092/C_Uniform/Random_Walk_Unsupervised/Unsupervised_NN/"
print("started epochs")
dataset = list()
model_t =MLP().cuda()
# exponential_test_level_set = [3,4,6,7,8]
# test_level_set = [1,3,5,7,9,11,13,15,17]

test_level_set = []

time_t_0 = 4
level_set_t1 = list()
representative_list = list()
for time in range(1, len(data)):
    level_set_states = list()
    rep_list = list()
    for each_state in data[time].keys():
        current_state = np.array(each_state).astype(np.float32)
        actual_probability = (data[time][each_state]).astype(np.float32)
        level_set_states.append(current_state)
        rep_list.append(torch.tensor(current_state))
    if time != len(data) - 1:
        level_set_t1.append(level_set_states)
    representative_list.append(rep_list)
    
print(len(dataset))
def dynamics_random_walk(state, action):
    """
    Computes the new state of the Dubins car after applying a control action.

    Args:
        state (tuple): Current state of the Dubins car (x, y, theta).
        action (float): Angular velocity (control input in radians).

    Returns:
        tuple: New state (x, y, theta) after applying the action.
    """
     # Reshape states to (N, 1, 2) for broadcasting
    states = state.unsqueeze(1)
    
    # Reshape actions to (1, M, 2) for broadcasting
    actions = action.unsqueeze(0)
    
    # Add states and actions to get new states
    new_states = states + actions
    
    return new_states
def count_updation(current_level_set_entropy, probability, representative_t_1, actions, real_states_batch):
    next_states = dynamics_random_walk(real_states_batch, actions).reshape(-1, 2)
    next_states_occurence = probability.reshape(-1).unsqueeze(1)
    decayed_distance = next_states_occurence * torch.exp(-10*torch.cdist(next_states, representative_t_1, p=2))
    probability = torch.sum(decayed_distance, axis = 0)
    current_level_set_entropy += probability
    return current_level_set_entropy

def entropy_maximization(current_level_set_entropy):
    probability = current_level_set_entropy / torch.sum(current_level_set_entropy)
    entropy_loss = torch.sum(probability * torch.log(probability + 1e-8)).mean()
    return entropy_loss

optimizer = torch.optim.Adam(model_t.parameters(), lr=1e-4)  
iterations = 100 
best_loss = None
for epoch in range(iterations+1):
    running_loss = 0.0
    running_entropy = 0.0
    model_t.train()
    for time in range(len(level_set_t1)):
        representative_t_1 = torch.stack(representative_list[time + 1]).cuda()
        max_entropy = np.array([1/len(representative_t_1) for i in range(len(representative_t_1))])
        print("Max Extropy", -np.sum(max_entropy * np.log(max_entropy)))
        data_loader = DataLoader(level_set_t1[time], batch_size=128, shuffle=True, num_workers=6)
        level_set_entropy = torch.zeros((1, len(representative_t_1))).cuda()
        for i, data_1 in enumerate(data_loader, 0):
            states_batch = data_1
            states = states_batch.cuda().to(torch.float32)
            pred_probability_distribution = model_t(states)
            level_set_entropy = count_updation(level_set_entropy, pred_probability_distribution, representative_t_1, actions, states)
        loss = entropy_maximization(level_set_entropy)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_entropy += loss.item()
        total_number_of_batches = i
        print("Model: " + str(time) + " Epoch: " + str(epoch) + " Level Set " + str(time)+ " Entropy Loss: " + str(loss.item()))
    with open(experiment_folder + 'training.log', 'a+') as f:
        f.write("Model: " + str(time) + " Epoch: " + str(epoch) + " Loss: " + str(running_loss) + " Entropy Loss: " + str(running_entropy) + "\n")
    
    if best_loss is not None:
        if (running_loss / (total_number_of_batches + 1)) < best_loss:
            torch.save(model_t, os.path.join(experiment_folder, "best_model.pt"))
            best_loss = running_loss / (total_number_of_batches + 1)
    else:
        torch.save(model_t, os.path.join(experiment_folder, "best_model.pt"))
        best_loss = running_loss / (total_number_of_batches + 1)