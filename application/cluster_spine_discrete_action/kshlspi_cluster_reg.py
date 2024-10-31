import numpy as np
import pandas as pd
import pickle
from models.lasso_ksh_lspi import group_lasso_ksh_lspi, \
    get_b_spline_means, get_b_spline_knots, b_spline_basis_state
import random
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from preprocess_data import prep_offline_data
import csaps
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import bs

# Import data.
direc = Path(Path().absolute())
data_dir = direc / "study_data_gps_gait_pain_response.csv" 
data = pd.read_csv(data_dir, index_col=0)

# Construct data according to (s,a,r,s') format.
state = ['home_time', 'dist_traveled', 'Age', 'days_since_surgery', 'radius',
         'num_sig_places', 'pause_time', 'max_dist_home', 'avg_cadence']
pos = int(sys.argv[1])
state.insert(0, state.pop(pos))
action = 'num_of_steps'
prepped_data = prep_offline_data(data, state, action, post_op_days=60, 
                                 continuous_action=False, sarsa=True)
mdp_data = prepped_data[0]
state_scaler = prepped_data[1]
action_scaler = prepped_data[2]

# Split into training and test set by patients.
random.seed(30)
train_set = 0.8
patients = list(set(mdp_data['Beiwe.ID']))
train_patients = random.sample(patients, int(train_set*len(patients)))
test_patients = list(set(patients) - set(train_patients))
train_data = mdp_data[mdp_data['Beiwe.ID'].isin(train_patients)]
test_data = mdp_data[mdp_data['Beiwe.ID'].isin(test_patients)]

# Shape data according to selected MDP.
reward = 'answer'
next_state = [i + '_next' for i in state]
mdp_train_data = np.array(train_data[state + ['action'] + [reward] + next_state + ['action_next']])
mdp_test_data = np.array(test_data[state + ['action'] + [reward] + next_state + ['action_next']])

# Initialize model parameters.
df = int(sys.argv[2])
h = float(sys.argv[3])
degree = int(sys.argv[4])
mu = float(sys.argv[5])
lambda_reg = float(sys.argv[6])
discount = float(sys.argv[7])
policy_max_iter = int(sys.argv[8])
nA = 2
sK = len(state)
k = (df*(sK-1) + 1) * nA
z = np.arange(0,1.01,0.01)
init_beta = np.random.uniform(-2,2,(len(z),k))

# run model at points z
act_val_0 = []
act_val_1 = []
spline_means = np.array([[]])
spline_knots = np.array([[]])

# Run local LSPI algorithm 
mod = group_lasso_ksh_lspi(
    data = mdp_train_data,
    discount = discount,
    sK = sK,
    degree = degree,
    init_beta = init_beta,
    nA = nA,
    z = z,
    h = h,
    lambda_reg = lambda_reg,
    df = df,
    mu = mu,
    mod_max_iter = 200, mod_eps=10**-3,
    policy_max_iter = policy_max_iter, policy_eps=10**-1,
    behavior_policy_init=True
    )
spline_knots = get_b_spline_knots(mdp_train_data, nA, sK, df, degree, padding = 10)
spline_means = get_b_spline_means(mdp_train_data, nA, sK, df, degree, spline_knots)

# get estimated action value at all points in z
act_val_0 = mod[:,0]
act_val_1 = mod[:,int(k/2)]

# predict directly using model output from cluster 
def predict_q_values(eval_data): 
    # compute predicted q-values 
    q_values = np.empty(0)
    for _, elem in eval_data.iterrows():
        # select model with nearest s_0 value
        index = np.argmin(np.abs(z - elem[state][0]))

        # compute dot product 
        q_val = mod[index,:].dot(b_spline_basis_state(
            elem[state], 
            int(elem['action']), nA, sK, df, degree, 
            spline_means, 
            spline_knots)
            )
        q_values = np.append(q_values, q_val)
    return(q_values)

# compute predicted q-values for training
tr_data = train_data[state + ['action'] + [reward] + next_state]
te_data = test_data[state + ['action'] + [reward] + next_state]

q_vals_train = predict_q_values(tr_data)
train_mse = np.mean((tr_data['answer'] - q_vals_train)**2)

# compute train and test mse
q_vals_test = predict_q_values(te_data)
test_mse = np.mean((te_data['answer'] - q_vals_test)**2)
print("Train MSE: {}".format(round(train_mse, 3)), 
        "Test MSE: {}".format(round(test_mse, 3)))

# save session objects
model_dict = {   
    'processed_data': prepped_data, 
    'train_data': train_data,
    'state': state,
    'state_interest': state[0],
    'action': action,
    'reward': reward,
    'df': df,
    'h': h,
    'discount': discount,
    'degree':degree,
    'mu': mu,
    'lambda_reg': lambda_reg,
    'nA': nA,
    'sK': sK,
    'z': z,
    'act_val_0':act_val_0,
    'act_val_1':act_val_1, 
    'opt_beta': mod,
    'spline_knots':spline_knots,
    'spline_means':spline_means,
    'train_mse': train_mse,
    'test_mse': test_mse
}
output_filename = 'results/ksh_lspi_dill_{}_degree_{}_discount_{}_h_{}_df_{}_mu_{}_lambda_{}_policy_iters_{}.p'.format(state[0], degree, 
    discount, h, df, mu, lambda_reg, policy_max_iter)
pickle.dump(model_dict, open(output_filename, "wb"))

# Compute joint state feature functions.
def f_j(s_j, s_0, a, j, z, nA, df, sK, degree, beta_matrix, spline_means, spline_knots):
    # select approriate weights
    pos = np.abs(z - s_0).argmin()
    k = (df*(sK-1) + 1) * nA
    indx = int(k/2)*a
    beta_j = beta_matrix[pos, 1+indx+(j-1)*df:1+indx+j*df]

    # basis expand state feature value 
    knots = spline_knots[a, int((j-1)*(df-degree+1)):int(j*(df-degree+1))]
    means = spline_means[a, int((j-1)*df):int(j*df)]
    basis = bs(s_j, degree=degree, knots=knots[:-2], 
               include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
        
    # compute value 
    val =  beta_j @ (basis[0] - basis.mean())
    return val 

def g(s_0, z, a, nA, df, sK, beta_matrix):
    # select approriate weights 
    k = (df*(sK-1) + 1) * nA
    pos = np.abs(z - s_0).argmin()
    val = beta_matrix[pos,a*int(k/2)]

    return val

def invTransform(scaler, data, col_index, num_features):
    temp = pd.DataFrame(np.zeros((len(data), num_features)))
    temp.iloc[:, col_index] = data
    temp = scaler.inverse_transform(temp)
    return temp[:, col_index]

# plot smooth contours for each state feature
pdf = matplotlib.backends.backend_pdf.PdfPages("results/smooth_discrete_action_contour_{}_degree_{}_discount_{}_h_{}_df_{}_mu_{}_lambda_{}_policy_iter_{}.pdf".format(state[0], degree, 
    discount, h, df, mu, lambda_reg, policy_max_iter))

# construct contour plot of joint state feature, action functions
diff = z[1]-z[0]
s_j = np.arange(0, 1 + 0.01, 0.01)

fig, ax = plt.subplots(8, 5, figsize = (22,32))
fig.subplots_adjust(hspace = 0.27)
ax = ax.ravel()

# plot f(a)
z_inverse = invTransform(state_scaler, z, 0, sK)
ax[0].plot(z_inverse, mod[:,0], label="0")
ax[0].plot(z_inverse, mod[:,int(k/2)], label="1")
ax[0].set_xlabel('z')
ax[0].set_ylabel('f_a')
fig.suptitle("Maginal Effect, f(a): s_0 = {},  df = {}, h = {}, degree = {}, train_mse = {}, test_mse = {}".format(state[0], df, h, degree, 
    round(train_mse,2), round(test_mse,2)))

ind = 1
for j in range(1, len(state)):
    # Compute matrix of f_j values for each (s_j, s_0) value pair under action a.
    for a in range(nA):
        func_grid = np.empty((len(z), len(s_j)))
        for i, z_val in enumerate(z):
            for m, s_val in enumerate(s_j):
                func_grid[i,m] = f_j(s_val, z_val, a, j, z, nA, df, sK, degree, mod, spline_means, spline_knots)

        # create contour plot 
        s_inverse = invTransform(state_scaler, s_j, j, sK)
        Z, S = np.meshgrid(z_inverse, s_inverse)
        X = [z, s_j]
        sp = csaps.NdGridCubicSmoothingSpline(X, func_grid, smooth=0.8)
        func_smooth = sp(X)
        cs = ax[ind].contourf(Z, S, func_smooth.T, cmap='hot_r')
        fig.colorbar(cs, ax=ax[ind], shrink=0.7)
        ax[ind].set_xlabel('z')
        ax[ind].set_ylabel('s_j')
        ax[ind].set_title("s_j = {}, a = {}".format(state[j], a))
        ind += 1

for j in range(1, len(state)):
    for a in range(nA):
        # compute matrix of f_j values for each (s_j, a) value pair 
        func_grid = np.empty((len(z), len(s_j)))
        for i, z_val in enumerate(z):
            for m, s_val in enumerate(s_j):
                func_grid[i,m] = f_j(s_val, z_val, a, j, z, nA, df, sK, degree, mod, spline_means, spline_knots) + g(z_val, z, a, nA, df, sK, mod)

        # create contour plot 
        s_inverse = invTransform(state_scaler, s_j, j, sK)
        Z, S = np.meshgrid(z_inverse, s_inverse)
        X = [z, s_j]
        sp = csaps.NdGridCubicSmoothingSpline(X, func_grid, smooth=0.8)
        func_smooth = sp(X)
        cs = ax[ind+2].contourf(Z, S, func_smooth.T, cmap='hot_r')
        fig.colorbar(cs, ax=ax[ind+2], shrink=0.7)
        ax[ind+2].set_xlabel('z')
        ax[ind+2].set_ylabel('s')
        ax[ind+2].set_title("s_j = {}, a = {}".format(state[j], a))
        ind += 1

pdf.savefig(fig)
pdf.close()
