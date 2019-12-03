import numpy as np
import random
from collections import namedtuple, deque

from model import ANet

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 3e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=42):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.policyNetwork = ANet(state_size, action_size, seed, 16, 16).to(device)
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=LR)
  
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policyNetwork.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        log_probs, discounted_rewards = experiences


        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        policy_gradient = torch.cat(policy_gradient).sum()
        

        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()
