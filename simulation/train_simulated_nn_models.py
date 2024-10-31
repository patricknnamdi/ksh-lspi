"""
Train DQN, NFQ, DoubleDQN, and DiscreteCQL on simulated MDPs.
"""
from d3rlpy.algos import DQN, NFQ, DoubleDQN, DiscreteCQL
from d3rlpy.dataset import MDPDataset
from simulation_config import Config, CustomEncoderFactory
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


    # Load dataset
    mdp_dataset = MDPDataset.load('cluster_wide_sweep/mdp_dataset_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.h5'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                                        correlated_states, 
                                                                                                         config.NUM_EPISODES,
                                                                                                                 config.MAX_STEPS))

    # Setup DQN algorithm.
    dqn = DQN(encoder_factory=CustomEncoderFactory(feature_size=config.STATE_DIM),
            gamma=config.DISCOUNT,
            target_update_interval=500,
            use_gpu=False,)

    # Train DQN.
    dqn.fit(mdp_dataset, 
            n_epochs=20,
            n_steps_per_epoch=100,
            logdir='training_logs', 
            experiment_name='dqn_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                                 correlated_states, config.NUM_EPISODES,
                                                                                                                 config.MAX_STEPS),)

    # Save DQN.
    dqn.save_model('dqn_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pt'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                           correlated_states, config.NUM_EPISODES,
                                                                                                                 config.MAX_STEPS))
    
    # Setup NFQ algorithm.
    nfq = NFQ(encoder_factory=CustomEncoderFactory(feature_size=config.STATE_DIM),
            gamma=config.DISCOUNT,
            use_gpu=False,)

    # Train NFQ.
    nfq.fit(mdp_dataset, 
            n_epochs=20,
            n_steps_per_epoch=100,
            logdir='training_logs', 
            experiment_name='nfq_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}'.format(config.STATE_DIM, config.DISCOUNT,
                                                                            correlated_states, config.NUM_EPISODES, config.MAX_STEPS),)

    # Save NFQ.
    nfq.save_model('nfq_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pt'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                           correlated_states, config.NUM_EPISODES, config.MAX_STEPS))

    # Setup DoubleDQN algorithm.
    ddqn = DoubleDQN(encoder_factory=CustomEncoderFactory(feature_size=config.STATE_DIM),
            gamma=config.DISCOUNT,
            target_update_interval=500,
            use_gpu=False,)

    # Train DoubleDQN.
    ddqn.fit(mdp_dataset, 
            n_epochs=20,
            n_steps_per_epoch=100,
            logdir='training_logs', 
            experiment_name='dsqn_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}'.format(config.STATE_DIM, config.DISCOUNT,
                                                                             correlated_states, config.NUM_EPISODES, config.MAX_STEPS),)

    # Save DoubleDQN.
    ddqn.save_model('ddqn_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pt'.format(config.STATE_DIM, config.DISCOUNT,
                                                                                             correlated_states, config.NUM_EPISODES, config.MAX_STEPS))

    # Setup DiscreteCQL algorithm.
    cql = DiscreteCQL(encoder_factory=CustomEncoderFactory(feature_size=config.STATE_DIM),
            gamma=config.DISCOUNT,
            target_update_interval=500,
            use_gpu=False,)

    # Train DiscreteCQL.
    cql.fit(mdp_dataset, 
            n_epochs=20,
            n_steps_per_epoch=100,
            logdir='training_logs', 
            experiment_name='dqn_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}'.format(
                config.STATE_DIM, config.DISCOUNT, correlated_states, config.NUM_EPISODES, config.MAX_STEPS),)

    # Save DiscreteCQL.
    cql.save_model('cql_simulated_state_dim_{}_discount_{}_correlated_states_{}_num_eps_{}_max_steps_{}.pt'.format(
        config.STATE_DIM, config.DISCOUNT, correlated_states, config.NUM_EPISODES, config.MAX_STEPS))


if __name__ == '__main__':
    main()