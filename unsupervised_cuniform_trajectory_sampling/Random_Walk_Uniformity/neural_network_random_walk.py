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
with open("/home/isleri/mahes092/C_Uniform/Datasets/x_u_pair_na3_ni30_max_flow_probability.pkl", "rb") as f:
    data = pickle.load(f)
## With Time
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

experiment_folder = "/home/isleri/mahes092/C_Uniform/Random_Walk_Uniformity/NN_Random_Walk_All/"
print("started epochs")
dataset = list()
model_t = MLP().cuda()
# exponential_test_level_set = [3,4,6,7,8]
# test_level_set = [2,4,6,8,10,12,14,16,18]
test_level_set = []
actions = np.array([[1,-1],[1,0],[1,1]])
max_samples = 61
for time in range(len(data)):
    if time not in test_level_set:
    # if time < 5:
        for each_state in data[time].keys():
            current_state = np.array(each_state).astype(np.float32)
            actual_probability = (data[time][each_state]).astype(np.float32)
            dataset.append([current_state, actual_probability, 1])
        current_number_samples = len(data[time].keys())
        # lev_states = list(data[time].keys())  # Get the keys as a list
        # min_value = min(np.array(lev_states)[:,1])
        # max_value = max(np.array(lev_states)[:,1])
        # extremity_measure = {}
        # for state in lev_states:
        #     midpoint = (min_value + max_value) / 2
        #     extremity_measure[state] =  1 / (1 + abs(state[1] - midpoint)) 
        # total_weight = sum(extremity_measure.values())
        # normalized_weights = [weight / total_weight for weight in extremity_measure.values()]
        # random_state = random.choices([each for each in range(len(lev_states))], weights=normalized_weights, k=max_samples - current_number_samples)
        # for sample_aug in range(len(random_state)):
        #     current_state = np.array(lev_states[random_state[sample_aug]]).astype(np.float32)
        #     actual_probability = (data[time][lev_states[random_state[sample_aug]]]).astype(np.float32)
        #     uniform_random = np.random.rand(1)
        #     # Scale and shift to the desired range
        #     scaled_random = -0.15 + (0.15 - -0.15) * uniform_random
        #     current_state[0] += scaled_random
        #     dataset.append([current_state, actual_probability, 1])
print(len(dataset))
data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)
optimizer = torch.optim.Adam(model_t.parameters(), lr=1e-4)  
iterations = 50
best_loss = None
for epoch in range(iterations+1):
    running_loss = 0.0
    running_normalization_loss = 0.0
    running_mse_loss = 0.0
    total_number_of_batches = 0
    model_t.train()
    for i, data_1 in enumerate(data_loader, 0):
        states_batch, action_prob_batch, weights = data_1[0], data_1[1], data_1[2]
        states = states_batch.cuda().to(torch.float32)
        pred_probability_distribution = model_t(states)
        probability_loss = nn.CrossEntropyLoss()(pred_probability_distribution, action_prob_batch.cuda().to(torch.float32))
        # probability_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(pred_probability_distribution, dim=1), action_prob_batch.cuda().to(torch.float32))
        loss = probability_loss
        running_loss += loss.item()
        running_mse_loss += probability_loss.item()
        loss.backward()
        optimizer.step()
        total_number_of_batches = i
    print("Model: " + str(time) + " Epoch: " + str(epoch) + " Loss: " + str(running_loss / (total_number_of_batches + 1)) + " Probability Loss: " + str(running_mse_loss / (total_number_of_batches + 1)))
    with open(experiment_folder + 'training.log', 'a+') as f:
        f.write("Model: " + str(time) + " Epoch: " + str(epoch) + " Loss: " + str(running_loss / (total_number_of_batches + 1)) + " Probability Loss: " + str(running_mse_loss / (total_number_of_batches + 1)) + "\n")
    
    if best_loss is None:
        torch.save(model_t, os.path.join(experiment_folder, "single_probability_model_best.pt"))
    elif best_loss > running_loss:
        torch.save(model_t, os.path.join(experiment_folder, "single_probability_model_best.pt"))
        best_loss = copy.deeopcopy(running_loss)