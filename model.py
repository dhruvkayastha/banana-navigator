import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ANet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=42, h1_size=16, h2_size=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Total number of actions
            seed (int): Random seed
        """

        super(ANet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_size, h1_size)
        self.linear2 = nn.Linear(h1_size, h2_size)
        self.linear3 = nn.Linear(h2_size, action_size)        
        

    def forward(self, state):
        """Build a network that maps state -> action probabilities."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return F.softmax(x, dim=1)
        