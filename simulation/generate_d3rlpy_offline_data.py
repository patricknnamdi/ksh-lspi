import numpy as np
import pickle
from d3rlpy.dataset import MDPDataset
from simulation_config import Config
import argparse


def main():
    parser = argparse.ArgumentParser(description='Update configuration.')
    parser.add_argument('config_file', type=str, help='configuration file name')
    parser.add_argument('--correlated_states', action='store_true', 
                        dest='correlated_states',
                        help='Use an MDP setup with correlated states')
    parser.set_defaults(correlated_states=False)
    args = parser.parse_args()

    # Load configuration file based on job id
    config = Config(args.config_file)

    # Use the correlated_states argument
    correlated_states = args.correlated_states

    # Load Monte Carlo simulation data.
    with open('cluster_wide_sweep/offline_dataset_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pkl'.format(config.STATE_DIM, 
                                                                    config.DISCOUNT, correlated_states, config.NUM_EPISODES,
                                                                                                                 config.MAX_STEPS), 'rb') as f:
        offline_dataset = pickle.load(f)

    # Make offline dataset into numpy array, with using the state, action, reward, next_state.
    rl_batch_dataset = np.array([])
    for (state, action, reward, _, next_state, next_action, _) in offline_dataset:
        row = np.concatenate((state, action, reward, next_state, next_action), axis=None)
        if rl_batch_dataset.size == 0:
            rl_batch_dataset = row
        else:
            rl_batch_dataset = np.vstack((rl_batch_dataset, row))

    # Get observatrions, actions, rewards, terminals from offline dataset.
    observations = rl_batch_dataset[:, :config.STATE_DIM]
    actions = rl_batch_dataset[:, config.STATE_DIM]
    rewards = rl_batch_dataset[:, config.STATE_DIM + 1]

    # Compute the terminal vector, where every MAX_STEPS steps is a terminal state.
    terminals = np.zeros(len(observations))
    for i in range(len(observations)):
        if i % config.MAX_STEPS == 0:
            terminals[i] = 1

    # Construct MDPDataset object.
    mdp_dataset = MDPDataset(observations, actions, rewards, terminals,
                             discrete_action=True)
    
    # Save as HDF5 file.
    mdp_dataset.dump('cluster_wide_sweep/mdp_dataset_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.h5'.format(config.STATE_DIM,
                                                                      config.DISCOUNT, correlated_states, config.NUM_EPISODES,
                                                                                                                 config.MAX_STEPS))

if __name__ == '__main__':
    main()