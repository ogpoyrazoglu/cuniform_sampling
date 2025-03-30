import torch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)  # This sets the seed for all GPUs
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, no_of_actions = 13):
        super().__init__()
        self.fc1 = nn.Linear(4, 256, bias=True)
        self.bn_1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.bn_2 = nn.BatchNorm1d(256)

        self.out = nn.Linear(256, no_of_actions, bias=True)
        self.act = nn.ReLU()

    def forward(self, x_input):
        x = self.bn_1(self.act(self.fc1(x_input)))
        x = self.bn_2(self.act(self.fc2(x)))
        x = nn.Softmax(dim = 1)(self.out(x))
        return x