import numpy as np
import yaml
import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory

# Configurations for the simulation.
class Config:
    # Define reward component functions -- simulation part 2 (correlated-states).
    @staticmethod
    def f_1(s,a):
        return((5 * np.sin(s[0]**2) + 5)*(a == 1) + (4*s[0] - 5)*(a == 0))
    
    @staticmethod
    def f_2(s,a):
        return((2*s[1]**3 - 5)*(a == 0) + (5*s[1]**2 + 5)*(a == 1))


    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            params = yaml.safe_load(file)

        self.__dict__.update(params)

# Construct out-of-sample bellman loss.
class Evaluation:
    def __init__(self, config):
        self.config = config

    def bellman_loss(self, model, dataset):
        # Extract state, action, reward, next_state, next_action from dataset.
        state = dataset[:, :self.config.STATE_DIM]
        action = dataset[:, self.config.STATE_DIM]
        reward = dataset[:, self.config.STATE_DIM + 1]
        next_state = dataset[:, self.config.STATE_DIM + 2:2*self.config.STATE_DIM + 2]
        next_action = dataset[:, 2*self.config.STATE_DIM + 2]

        # Compute bellman loss.
        bellman_loss = 0
        for i in range(len(state)):
            bellman_loss += (reward[i] + self.config.DISCOUNT * model.get_action_value(next_state[i], next_action[i]) - model.get_action_value(state[i], action[i]))**2

        bellman_loss /= len(state)
        return bellman_loss

# Custom nueral network with 64 hidden units.
class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()  # This line was added.
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0], 64)
        self.fc2 = nn.Linear(64, feature_size)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return h

    def get_feature_size(self):
        return self.feature_size

# Setup custom encoder factory.
class CustomEncoderFactory(EncoderFactory):
    TYPE = 'custom' 

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return CustomEncoder(observation_shape, self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}