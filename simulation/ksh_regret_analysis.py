import numpy as np
import pickle
import math
import sys
import torch
import torch.nn as nn
from d3rlpy.algos import DiscreteCQL, DQN
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import EncoderFactory
from models.ksh_lspi_algorithm import KSH_LSPI
from simulation.simulated_mdp import MDP

# Define parameters.
STATE_DIM = int(sys.argv[1])
DISCOUNT = float(sys.argv[2])
SIGMA = 0.3

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
    
# Define reward component functions.
def f_1(s,a):
    return((5 * np.sin(s**2) + 5)*(a == 1) + (4*s - 5)*(a == 0))

def f_2(s,a):
    return((2 * s**3 - 5)*(a == 0)  + (5*s**2 + 5)*(a == 1))

# Run models in environment and compute regret.
def generate_transitions(mdp, policy=None, max_steps=10, num_episodes=10):
    transitions = []    
    regrets = []
    rewards = []
    actions = []
    for _ in range(num_episodes):
        state = mdp.reset()
        action = policy(state)
        
        for _ in range(max_steps):
            next_state, reward, reward_components = mdp.step(action, components=True)
            actions.append(action)
            # Note the reward is binary. Compute reward of the alternative action.
            alternate_reward = mdp.compute_reward(next_state, 1 - action)[0]

            # Compute regret
            regret = max(reward, alternate_reward) - reward
            regrets.append(regret)
            rewards.append(reward)

            # Compute next action
            next_action = policy(next_state)
            transitions.append((state, action, reward, reward_components, next_state, next_action))
            action = next_action            
            state = next_state
        
    return np.array(transitions), np.array(regrets), np.array(rewards), np.array(actions)


def main():
    # Load dataset
    mdp_dataset = MDPDataset.load('mdp_dataset_state_dim_{}_discount_{}.h5'.format(STATE_DIM, DISCOUNT))

    # Setup and load DQN algorithm.
    dqn = DQN(encoder_factory=CustomEncoderFactory(feature_size=STATE_DIM),
            gamma=DISCOUNT,
            target_update_interval=500,
            use_gpu=False,)
    dqn.build_with_dataset(mdp_dataset)
    dqn.load_model('dqn_simulated_state_dim_{}_discount_{}.pt'.format(STATE_DIM, DISCOUNT))

    # Setup MDP environment.
    mdp = MDP(state_dim=STATE_DIM, sigma=SIGMA, f_1=f_1, f_2=f_2)
    
    # Define policy for DQN.
    def dqn_policy(state):
        return dqn.predict([state])[0]
    
    # Run DQN in MDP environment and compute regret. 
    _, regrets, rewards, actions = generate_transitions(
        mdp, dqn_policy, max_steps=10, num_episodes=1000
    )

    print('DQN Regret: {}'.format(np.mean(regrets)))
    print('DQN Reward: {}'.format(np.mean(rewards)))
    print('DQN Action: {}'.format(np.mean(actions)))


if __name__ == '__main__':
    main()