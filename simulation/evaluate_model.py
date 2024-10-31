from simulation_config import Config, CustomEncoderFactory
from simulation.simulated_mdp import CorrelatedStatesMDP
from d3rlpy.algos import DQN, NFQ, DoubleDQN, DiscreteCQL
from d3rlpy.dataset import MDPDataset
import pandas as pd 
import numpy as np 
import pickle
import argparse

correlated_states = True

# Run models in environment and compute regret.
def generate_transitions(mdp, policy=None, max_steps=10, num_episodes=100):
    states = []    
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
            states.append(state)
            action = next_action            
            state = next_state

    # Print simulate complete prompt
    print("Simulation complete")
    return np.array(states), np.array(regrets), np.array(rewards), np.array(actions)

def evaluate_ksh_models(config, idx, max_steps=10, num_episodes=1000):

    # Setup MDP.
    mdp = CorrelatedStatesMDP(state_dim=config.STATE_DIM, sigma=config.SIGMA, f_1=config.f_1, f_2=config.f_2,
                            b=config.B, delta=config.DELTA)

    # ----------------------
    # Evaluate KSH Algorithm    
    # ----------------------
    # Load model.
    with open('ksh_lspi_model_state_dim_{}_state_idx_{}_h_{}_num_z_{}_num_eps_{}_max_steps_{}.pkl'.format(config.STATE_DIM, idx, 
                                                                                                          config.H, config.NUM_Z, 
                                                                                                          config.NUM_EPISODES, 
                                                                                                          config.MAX_STEPS), 'rb') as f:
        ksh_simulated_model = pickle.load(f)

    # Run KSH in MDP environment and compute regret.
    _, ksh_regrets, ksh_rewards, ksh_actions = generate_transitions(
        mdp, ksh_simulated_model.select_action, max_steps=max_steps, num_episodes=num_episodes
    )

    # Create a vector that indicates the current episode for each transition, 
    # Note that than episode last for max_steps.
    episode = np.repeat(np.arange(1000), 10) + 1

    # Place results within a pandas dataset
    results = pd.DataFrame({
        'episode': episode,
        'ksh_regrets': ksh_regrets,
        'ksh_rewards': ksh_rewards,
        'ksh_actions': ksh_actions,
    })

    return results

def evaluate_nn_models(config, max_steps=10, num_episodes=1000):

    # ---------------------------
    # Evaluate NN-based Algorithm
    # ---------------------------

    correlated_states = True

    # Load dataset
    mdp_dataset = MDPDataset.load('cluster_wide_sweep/mdp_dataset_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.h5'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                                        correlated_states, config.NUM_EPISODES,
                                                                                                                    config.MAX_STEPS))

    # Setup and load DQN algorithm.
    dqn = DQN(encoder_factory=CustomEncoderFactory(feature_size=config.STATE_DIM),
            gamma=config.DISCOUNT,
            target_update_interval=500,
            use_gpu=False,)
    dqn.build_with_dataset(mdp_dataset)
    dqn.load_model('cluster_wide_sweep/dqn_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pt'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                            correlated_states, config.NUM_EPISODES,
                                                                                                                    config.MAX_STEPS))
    # Setup and load NFQ algorithm.
    nfq = NFQ(encoder_factory=CustomEncoderFactory(feature_size=config.STATE_DIM),
            gamma=config.DISCOUNT,
            use_gpu=False,)
    nfq.build_with_dataset(mdp_dataset)
    nfq.load_model('cluster_wide_sweep/nfq_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pt'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                            correlated_states, config.NUM_EPISODES, config.MAX_STEPS))

    # Setup and load DoubleDQN algorithm.
    ddqn = DoubleDQN(encoder_factory=CustomEncoderFactory(feature_size=config.STATE_DIM),
                gamma=config.DISCOUNT,
                target_update_interval=500,
                use_gpu=False,)
    ddqn.build_with_dataset(mdp_dataset)
    ddqn.load_model('cluster_wide_sweep/ddqn_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pt'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                                correlated_states, config.NUM_EPISODES, config.MAX_STEPS))

    # Setup and load NFQ algorithm.
    cql = DiscreteCQL(encoder_factory=CustomEncoderFactory(feature_size=config.STATE_DIM),
                gamma=config.DISCOUNT,
                target_update_interval=500,
                use_gpu=False,)
    cql.build_with_dataset(mdp_dataset)
    cql.load_model('cluster_wide_sweep/cql_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pt'.format(
            config.STATE_DIM, config.DISCOUNT, correlated_states, config.NUM_EPISODES, config.MAX_STEPS))


    # Setup MDP environment.
    mdp = CorrelatedStatesMDP(state_dim=config.STATE_DIM, sigma=config.SIGMA, f_1=config.f_1, f_2=config.f_2,
                            b=config.B, delta=config.DELTA)

    # Define policy for DQN.
    def dqn_policy(state):
        return dqn.predict([state])[0]

    # Define policy for NFQ.
    def nfq_policy(state):
        return nfq.predict([state])[0]

    # Define policy for DoubleDQN.
    def ddqn_policy(state):
        return ddqn.predict([state])[0]

    # Define policy for CQL.
    def cql_policy(state):
        return cql.predict([state])[0]

    # Run DQN in MDP environment and compute regret. 
    _, dqn_regrets, dqn_rewards, dqn_actions = generate_transitions(
        mdp, dqn_policy, max_steps=max_steps, num_episodes=num_episodes
    )

    # Run NFQ in MDP environment and compute regret.
    _, nfq_regrets, nfq_rewards, nfq_actions = generate_transitions(
        mdp, nfq_policy, max_steps=max_steps, num_episodes=num_episodes
    )

    # Run DoubleDQN in MDP environment and compute regret.
    _, ddqn_regrets, ddqn_rewards, ddqn_actions = generate_transitions(
        mdp, ddqn_policy, max_steps=max_steps, num_episodes=num_episodes
    )

    # Run CQL in MDP environment and compute regret.
    _, cql_regrets, cql_rewards, cql_actions = generate_transitions(
        mdp, cql_policy, max_steps=max_steps, num_episodes=num_episodes
    )


    # Create a vector that indicates the current episode for each transition, 
    # Note that than episode last for max_steps.
    episode = np.repeat(np.arange(1000), 10) + 1

    # Place results within a pandas dataset
    results = pd.DataFrame({
        'episode': episode,
        'dqn_regrets': dqn_regrets,
        'dqn_rewards': dqn_rewards,
        'dqn_actions': dqn_actions,
        'nfq_regrets': nfq_regrets,
        'nfq_rewards': nfq_rewards,
        'nfq_actions': nfq_actions,
        'ddqn_regrets': ddqn_regrets,
        'ddqn_rewards': ddqn_rewards,
        'ddqn_actions': ddqn_actions,
        'cql_regrets': cql_regrets,
        'cql_rewards': cql_rewards,
        'cql_actions': cql_actions,
    })

    return results

def main():
    parser = argparse.ArgumentParser(description='Update configuration.')
    parser.add_argument('config_file', type=str, help='configuration file name')
    parser.add_argument('--idx', dest='candidate_state_idx', type=int, required=False,
                        help='The index of the candidate state')
    args = parser.parse_args()
    idx = args.candidate_state_idx

    # Load configuration file based on job id
    config = Config(args.config_file)

    if idx is None:
        # Evaluate models
        results = evaluate_nn_models(config)

        # Save results
        results.to_csv('cluster_wide_sweep/nn_evaluation_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.csv'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                                                        True, config.NUM_EPISODES,
                                                                                                                        config.MAX_STEPS), 
                    index=False)

    else:
        # Evaluate models
        results = evaluate_ksh_models(config, idx)

        # Save results
        results.to_csv('ksh_lspi_evaluation__state_dim_{}_state_idx_{}_h_{}_num_z_{}_num_eps_{}_max_steps_{}.csv'.format(config.STATE_DIM, idx, 
                                                                                                            config.H, config.NUM_Z, 
                                                                                                            config.NUM_EPISODES, 
                                                                                                            config.MAX_STEPS), 
                    index=False)

if __name__ == '__main__':
    main()
