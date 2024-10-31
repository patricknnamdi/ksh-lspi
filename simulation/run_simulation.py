from simulated_mdp import MDP, CorrelatedStatesMDP, NonlinearTransitionMDP
from simulation_config import Config
import numpy as np
import multiprocessing as mp
import pickle
import argparse
import random
import os


# Generate an episode
def generate_transitions(mdp, policy=None, max_steps=10, num_episodes=10, sarsa=False):
    transitions = []    
    for _ in range(num_episodes):
        state = mdp.reset()
        
        if policy is None:
            action = np.random.randint(2)
        else:
            action = policy(state)
        
        for _ in range(max_steps):
            next_state, reward, reward_components = mdp.step(action, components=True)
            
            if sarsa and policy is not None:
                next_action = policy(next_state)
            else:
                next_action = np.random.randint(2) if policy is None else policy(next_state)

            if sarsa:
                transitions.append((state, action, reward, reward_components, next_state, next_action))
                action = next_action
            else:
                transitions.append((state, action, reward, reward_components, next_state))
            
            state = next_state
        
    return transitions

# Construct worker function for parallel Monte Carlo estimation.
def worker(args):
    state, action, mdp, gamma, num_episodes, max_steps = args
    # mdp = MDP(state_dim, sigma, Config.f_1, Config.f_2)
    mc_q_value = mdp.monte_carlo_q_estimate(state, action, gamma, num_episodes, 
                                            max_steps)
    return (tuple(state), action), mc_q_value

# Construst mc estimates via parallelization.
def parallel_monte_carlo_estimates(dataset, mdp, gamma, num_episodes, max_steps):
    # Prepare the arguments for the worker function
    worker_args = [(state, action, mdp, gamma, num_episodes, max_steps) for state, action, *rest in dataset]

    # Take a random sample of 100 state-action pairs
    # worker_args = random.sample(worker_args, 100)

    num_workers = mp.cpu_count()
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker, worker_args)

    return dict(results)

def main():
    parser = argparse.ArgumentParser(description='Update configuration.')
    parser.add_argument('config_file', type=str, help='configuration file name')
    parser.add_argument('--correlated_states', action='store_true', 
                        dest='correlated_states',
                        help='Use an MDP setup with correlated states')
    parser.add_argument('--nonlinear_states', action='store_true', 
                        dest='nonlinear_states',
                        help='Use an MDP setup with nonlinear state transitions')
    parser.set_defaults(correlated_states=False)
    parser.set_defaults(nonlinear_states=False)
    args = parser.parse_args()

    # Load configuration file based on job id
    config = Config(args.config_file)

    # Use the alternative MDP argument
    correlated_states = args.correlated_states
    nonlinear_states = args.nonlinear_states

    # Create MDP
    if correlated_states:
        mdp = CorrelatedStatesMDP(config.STATE_DIM, config.SIGMA, Config.f_1, 
                                  Config.f_2, config.B, config.DELTA)
    elif nonlinear_states:
        mdp = NonlinearTransitionMDP(config.STATE_DIM, config.NUM_ACTIONS, 
                                     config.SIGMA, Config.f_1, Config.f_2)
    else:
        mdp = MDP(config.STATE_DIM, config.SIGMA, Config.f_1, Config.f_2)

    # Generate an offline dataset
    offline_dataset = generate_transitions(mdp, max_steps=config.MAX_STEPS, 
                                           num_episodes=config.NUM_EPISODES, 
                                           sarsa=True)

    # Calculate the Monte Carlo Q-function estimates for each state-action pair in the dataset in parallel
    mc_q_estimates = parallel_monte_carlo_estimates(offline_dataset, mdp, config.GAMMA, 
                                                    config.MC_NUM_EPISODES, 
                                                    config.MC_MAX_STEPS)

    for state_action_pair, q_value in mc_q_estimates.items():
        print(f"State: {state_action_pair[0]}, Action: {state_action_pair[1]}, Q-value: {q_value}")

    # Append the Monte Carlo Q-function estimates to the dataset
    for i, (state, action, reward, reward_components, next_state, next_action) in enumerate(offline_dataset):
        offline_dataset[i] = (state, action, reward, reward_components, next_state, next_action, mc_q_estimates[(tuple(state), action)])

    # Make a seperate dataset for the Monte Carlo Q-function estimates and match them with the state-action pairs
    # mc_q_estimates_dataset = [(state, action, q_value) for (state, action), q_value in mc_q_estimates.items()]

    # # Save the Monte Carlo Q-function estimates using pickle
    # with open('datasets/mc_q_estimates_state_dim_{}_discount_{}_correlated_states_{}_nonlinear_states_{}_num_eps_{}_max_steps_{}_mc_max_steps_{}_mc_eps_{}.pkl'.format(config.STATE_DIM, config.GAMMA, 
    #                                                                                                              correlated_states, nonlinear_states, config.NUM_EPISODES,
    #                                                                                                              config.MAX_STEPS, config.MC_MAX_STEPS, config.MC_NUM_EPISODES), "wb") as f:
    #     pickle.dump(mc_q_estimates, f)

    # If folder datasets does not exist, create it
    if not os.path.exists('datasets'):
        os.makedirs('datasets')


    # Save the dataset using pickle
    with open('datasets/offline_dataset_state_dim_{}_discount_{}_correlated_states_{}_nonlinear_states_{}_num_eps_{}_max_steps_{}.pkl'.format(config.STATE_DIM, config.GAMMA, 
                                                                                                                 correlated_states, nonlinear_states, config.NUM_EPISODES,
                                                                                                                 config.MAX_STEPS), "wb") as f:
        pickle.dump(offline_dataset, f)

if __name__ == '__main__':
    main()