import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import numpy as np
import seaborn as sns
import pickle
from models.ksh_lspi_algorithm import KSH_LSPI
from simulation_config import Config, Evaluation
import argparse
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Reorder state components to bring ith component to the first index.
def reorder_state_components(dataset, i, state_dim):
    if i < 0 or i >= state_dim - 1:
        raise ValueError("Invalid component index. Must be between 0 and STATE_DIM - 1.")
    
    reordered_dataset = dataset.copy()
    # Reorder state components
    reordered_dataset[:, :state_dim] = np.roll(dataset[:, :state_dim], -i, axis=1)
    # Reorder next_state components
    reordered_dataset[:, state_dim + 2 : 2 * state_dim + 2] = np.roll(dataset[:, state_dim + 2 : 2 * state_dim + 2], -i, axis=1)

    return reordered_dataset

def main():
    # Use argparse to initialize correlated states variable
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='configuration file name')
    parser.add_argument('--correlated_states', action='store_true', 
                        dest='correlated_states',
                        help='Use an MDP setup with correlated states')
    parser.add_argument('--nonlinear_states', action='store_true', 
                        dest='nonlinear_states',
                        help='Use an MDP setup with nonlinear state transitions')
    parser.set_defaults(correlated_states=False)
    parser.set_defaults(nonlinear_states=False)
    # parser.add_argument('--idx', dest='candidate_state_idx', type=int, required=True,
    #                     help='The index of the candidate state')
    args = parser.parse_args()
    correlated_states = args.correlated_states
    nonlinear_states = args.nonlinear_states

    # Load configuration file based on job id
    config = Config(args.config_file)
    idx = config.CANDIDATE_STATE_IDX
    
    # Initialize W&B with a unique name
    #wandb.init(project='ksh_lspi_simulations_testing')
    wandb.init(project='ksh_lspi_paper_simulations_ablation_mu_study')

    # Log config to W&B
    wandb.config.update(config)

    # Load Monte Carlo simulation data.
    dataset_path = 'datasets/offline_dataset_state_dim_{}_discount_{}_correlated_states_{}_nonlinear_states_{}_num_eps_{}_max_steps_{}.pkl'.format(config.STATE_DIM, config.GAMMA, 
                                                                                                                 correlated_states, nonlinear_states, config.NUM_EPISODES,
                                                                                                                 config.MAX_STEPS)
    with open(dataset_path, 'rb') as f:
        offline_dataset = pickle.load(f)

    # Shuffle dataset.
    np.random.seed(0)
    np.random.shuffle(offline_dataset)

    # Make offline dataset into numpy array, with using the state, action, reward, next_state.
    rl_batch_dataset = np.array([])
    mc_estimates = np.array([])
    for (state, action, reward, _, next_state, next_action, (mc_q_estimates, (f_1_component, f_2_component))) in offline_dataset:
        row = np.concatenate((state, action, reward, next_state, next_action), axis=None)
        mc_row = np.concatenate((state[0:2], action, mc_q_estimates, f_1_component, f_2_component), axis=None)
        if rl_batch_dataset.size == 0:
            rl_batch_dataset = row
            mc_estimates = mc_row
        else:
            rl_batch_dataset = np.vstack((rl_batch_dataset, row))
            mc_estimates = np.vstack((mc_estimates, mc_row))

    # Log datasets as W&B artifacts
    artifact = wandb.Artifact('offline_dataset', type='dataset')
    artifact.add_file(dataset_path)
    wandb.log_artifact(artifact)
    
    # Generate dataset for alternative state ordering.
    train_dataset_1 = reorder_state_components(rl_batch_dataset, idx, config.STATE_DIM)

    # Cross-validation setup
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=0)
    bellman_losses = []

    for train_index, val_index in kf.split(train_dataset_1):
        train_data, val_data = train_dataset_1[train_index], train_dataset_1[val_index]

        # Fit Lasso-KSH-LSPI model.
        z = np.linspace(np.min(train_data[:, 0]), np.max(train_data[:, 0]), config.NUM_Z)
        ksh_model = KSH_LSPI(data=train_data,
                num_actions=config.NUM_ACTIONS,
                state_dim=config.STATE_DIM,
                df=config.DF,
                bandwidth=config.H,
                mu=float(config.MU),
                degree=config.DEGREE,
                discount=config.GAMMA,
                max_iter=config.MAX_ITER,
                max_policy_iter=config.MAX_POLICY_ITER,
                lambda_reg=config.LAMBDA_REG,
                basis_type=config.BASIS_TYPE,
                local_centers=z,)
        for i in z:
            print("Fitting model for z = {}".format(i))
            ksh_model.learn(train_data, i, behavioral_init=True)

        # Perform policy iteration. 
        ksh_model.policy_iteration(train_data)

        # Evaluate model
        evaluation = Evaluation(config)
        bellman_loss = evaluation.bellman_loss(ksh_model, val_data)
        bellman_losses.append(bellman_loss)
        print("Bellman loss for fold: ", bellman_loss)

    avg_bellman_loss = np.mean(bellman_losses)
    print("Average Bellman loss: ", avg_bellman_loss)

    # Log average bellman loss to W&B
    wandb.log({'avg_bellman_loss': avg_bellman_loss})

    # Save final model trained on the full dataset
    ksh_model = KSH_LSPI(data=train_dataset_1,
            num_actions=config.NUM_ACTIONS,
            state_dim=config.STATE_DIM,
            df=config.DF,
            bandwidth=config.H,
            mu=float(config.MU),
            degree=config.DEGREE,
            discount=config.GAMMA,
            max_iter=config.MAX_ITER,
            max_policy_iter=config.MAX_POLICY_ITER,
            lambda_reg=config.LAMBDA_REG,
            local_centers=z,)
    for i in z:
        print("Fitting model for z = {}".format(i))
        ksh_model.learn(train_dataset_1, i, behavioral_init=True)
    ksh_model.policy_iteration(train_dataset_1)

    # Make results directory if it does not exist.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save model.
    model_path = 'results/ksh_lspi_model_state_dim_{}_state_idx_{}_h_{}_num_z_{}_num_eps_{}_max_steps_{}.pkl'.format(
        config.STATE_DIM, idx, config.H, config.NUM_Z, config.NUM_EPISODES, config.MAX_STEPS)
    with open(model_path, 'wb') as f:
        pickle.dump(ksh_model, f)
        
    # Log the model as W&B artifact
    artifact = wandb.Artifact('ksh_lspi_model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # Construct visualization of first and second component functions. 
    f_idx_0_estimate = []
    f_idx_1_estimate = []
    for sample in train_dataset_1:
        state = sample[:config.STATE_DIM]
        action = sample[config.STATE_DIM]
        if action == 0:
            f_idx_0_estimate.append(ksh_model.get_marginal_component(state, action))
        else:
            f_idx_1_estimate.append(ksh_model.get_marginal_component(state, action))

    # Get model weights for each action.
    weights = ksh_model.weights
    g_0 = ksh_model.weights[:,0]
    g_1 = ksh_model.weights[:,int(weights.shape[1]/2)]
    z = ksh_model.local_centers

    # Put weights into a dataframe.
    weights = pd.DataFrame(np.array([z, g_0, g_1]).T, columns = ["x", "g_0", "g_1"])

    # Put MC estimates into a dataframe.
    mc_estimates = pd.DataFrame(mc_estimates, 
                                columns = ["state_0", "state_1", 
                                        "action", "mc_q", "mc_f_1", "mc_f_2"])
    
    # Calculate shift parameter. Mean difference between MC and KSH estimates. Filter
    # by action.
    if idx == 0:
        mc_estimates_action_0 = mc_estimates[(mc_estimates['action'] == 0)]
        mc_estimates_action_1 = mc_estimates[(mc_estimates['action'] == 1)]
        shift_action_0 = np.mean(mc_estimates_action_0['mc_f_1'] - f_idx_0_estimate)
        shift_action_1 = np.mean(mc_estimates_action_1['mc_f_1'] - f_idx_1_estimate)

        # Shift KSH estimates.
        weights['g_0'] = weights['g_0'] + shift_action_0
        weights['g_1'] = weights['g_1'] + shift_action_1
    else:
        mc_estimates_action_0 = mc_estimates[(mc_estimates['action'] == 0)]
        mc_estimates_action_1 = mc_estimates[(mc_estimates['action'] == 1)]
        shift_action_0 = np.mean(mc_estimates_action_0['mc_f_2'] - f_idx_0_estimate)
        shift_action_1 = np.mean(mc_estimates_action_1['mc_f_2'] - f_idx_1_estimate)

        # Shift KSH estimates.
        weights['g_0'] = weights['g_0'] # + shift_action_0
        weights['g_1'] = weights['g_1'] # + shift_action_1

    # Add second y-axis to the plot.
    sns.set_theme(style="ticks", font_scale=1.7)
    sns.despine(top = False)
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax2 = ax1.twinx()

    # Use seaborn to plot the weights.
    sns.lineplot(x="x", y="g_0", data=weights, color = "tab:blue", ax = ax1, linewidth=2)
    sns.lineplot(x="x", y="g_1", data=weights, color = "tab:orange", ax = ax1, linewidth=2)

    # Use seaborn to plot the MC estimates for each action separetely.
    if idx == 0:
        sns.lineplot(x="state_0", y="mc_f_1", hue='action', data=mc_estimates, 
                    alpha = 0.5, color = "black", ax = ax1, linestyle='--', linewidth=2)
        # Add a density of the state values for both actions to the figures to second y-axis.
        sns.kdeplot(x='state_0', data=mc_estimates, ax = ax2, fill=True, color='Grey',
                    alpha=.05, linewidth=0.5)
    else:
        sns.lineplot(x="state_1", y="mc_f_2", hue='action', data=mc_estimates, 
                    alpha = 0.5, color = "black", ax = ax1, linestyle='--', linewidth=2)
        # Add a density of the state values for both actions to the figures to second y-axis.
        sns.kdeplot(x='state_1', data=mc_estimates, ax = ax2, fill=True, color='Grey',
                    alpha=.05, linewidth=0.5)
    ax2.set_ylim(0, 3.5)
    ax2.set_yticks([0,1])
    ax2.set(ylabel=None)
    ax1.set(xlabel=None)
    ax1.set(ylabel=None)

    # Remove legend from the first y-axis.
    ax1.legend_.remove()
    sns.despine(right=False)

    # Save plot pdf.
    run_name = wandb.run.name
    plot_path = 'results/ksh_lspi_model_state_dim_{}_state_idx_{}_h_{}_num_z_{}_num_eps_{}_max_steps_{}_run_{}.pdf'.format(
        config.STATE_DIM, idx, config.H, config.NUM_Z, 
        config.NUM_EPISODES, config.MAX_STEPS, run_name)
    plt.savefig(plot_path, bbox_inches='tight')
    
    # Save as png for W&B
    png_plot_path = plot_path.replace('.pdf', '.png')
    plt.savefig(png_plot_path, bbox_inches='tight')

    # Log the plot as W&B artifact
    artifact = wandb.Artifact('ksh_lspi_model_component_plot', type='plot')
    artifact.add_file(plot_path)
    wandb.log_artifact(artifact)

    # Log the plot as W&B image
    wandb.log({"ksh_lspi_model_component_plot": wandb.Image(png_plot_path)})

    # Log the shift parameters to W&B
    wandb.log({'shift_action_0': shift_action_0, 'shift_action_1': shift_action_1})
    
if __name__ == '__main__':
    main()