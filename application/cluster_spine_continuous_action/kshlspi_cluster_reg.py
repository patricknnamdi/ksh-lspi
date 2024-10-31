import numpy as np
import pandas as pd
import pickle
from models.lasso_ksh_lspi_continous_action import group_lasso_ksh_lspi, \
    b_spline_basis, get_b_spline_means, predict_action_values, get_b_spline_knots
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
action = 'num_of_steps'
prepped_data = prep_offline_data(data, state, action, post_op_days=60, 
                                 continuous_action=True, sarsa=True)
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
mdp_train_data = np.array(train_data[state + ['action'] + [reward] + next_state])
mdp_test_data = np.array(test_data[state + ['action'] + [reward] + next_state])

# Initialize model parameters.
df = int(sys.argv[1])
h = float(sys.argv[2])
degree = int(sys.argv[3])
mu = float(sys.argv[4])
lambda_reg = float(sys.argv[5])
discount = float(sys.argv[6])
policy_max_iter = int(sys.argv[7])
sK = len(state)
k = df*sK + 1
z = np.arange(0,1.05,0.05)
init_beta = np.random.uniform(-2,2,(len(z),k))

# Run model at points z.
spline_means = np.array([[]])
spline_knots = np.array([[]])

# Run KSH-LSPI algorithm.
mod = group_lasso_ksh_lspi(
    data = mdp_train_data,
    discount = discount,
    sK = sK,
    degree = degree,
    init_beta = init_beta,
    z = z,
    h = h,
    lambda_reg = lambda_reg,
    df = df,
    mu = mu,
    mod_max_iter = 200, mod_eps=10**-3,
    policy_max_iter = policy_max_iter, policy_eps=10**-1,
    behavior_policy_init=True
    )

# Get estimated action value at all points in z.
act_val = mod[:,0]

# Get spline knots and means. 
spline_knots = get_b_spline_knots(mdp_train_data, sK, df, degree, padding = 10)
spline_means = get_b_spline_means(mdp_train_data, sK, df, degree, spline_knots)

# Compute predicted q-values for training.
train_phi_s = b_spline_basis(mdp_train_data[:,:sK], sK, df, degree)
q_vals_train = predict_action_values(train_phi_s, mdp_train_data[:,sK], mod, z)
train_mse = np.mean((train_data['answer'] - q_vals_train)**2)

# Compute predicted q-values for test.
test_phi_s = b_spline_basis(mdp_test_data[:,:sK], sK, df, degree, spline_means, 
                            spline_knots)
q_vals_test = predict_action_values(test_phi_s, mdp_test_data[:,sK], mod, z)
test_mse = np.mean((test_data['answer'] - q_vals_test)**2)

# Save session objects.
model_dict = {  
    'processed_data': prepped_data, 
    'train_data': mdp_train_data,
    'state': state,
    'action': action,
    'reward': reward,
    'df': df,
    'h': h,
    'discount': discount,
    'degree':degree,
    'mu': mu,
    'lambda_reg': lambda_reg,
    'sK': sK,
    'z': z,
    'act_val':act_val,
    'opt_beta': mod,
    'spline_knots':spline_knots,
    'spline_means':spline_means,
    'train_mse': train_mse,
    'test_mse': test_mse
}
output_filename = 'results/ksh_lspi_dill_degree_{}_discount_{}_h_{}_df_{}_mu_{}_lambda_{}_policy_iter_{}.p'.format(degree, 
    discount, h, df, mu, lambda_reg, policy_max_iter)
pickle.dump(model_dict, open(output_filename, "wb"))

# helper functions
# compute joint state feature, action function
def f_j(s, a, j, z, df, degree, beta_matrix, spline_means, spline_knots):
    # select approriate weights
    pos = np.abs(z - a).argmin()
    beta_j = beta_matrix[pos, 1+(j-1)*df:1+j*df]

    # basis expand state feature value 
    knots = spline_knots[int((j-1)*(df-degree+1)):int(j*(df-degree+1))]
    means = spline_means[int((j-1)*df):int(j*df)]
    basis = bs(s, degree=degree, knots=knots[:-2], 
               include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
        
    # compute value 
    val =  beta_j @ (basis[0] - basis.mean())
    return val 

def f(a, z, beta_matrix):
    # select approriate weights 
    pos = np.abs(z - a).argmin()
    val = beta_matrix[pos,0]

    return val

def invTransform(scaler, data, col_index, num_features):
    temp = pd.DataFrame(np.zeros((len(data), num_features)))
    temp.iloc[:, col_index] = data
    temp = scaler.inverse_transform(temp)
    return temp[:, col_index]

# plot smooth contours for each state feature
pdf = matplotlib.backends.backend_pdf.PdfPages("results/smooth_contour_degree_{}_discount_{}_h_{}_df_{}_mu_{}_lambda_{}_policy_iter_{}.pdf".format(degree, 
    discount, h, df, mu, lambda_reg, policy_max_iter))

# construct contour plot of joint state feature, action functions
diff = z[1]-z[0]
s_j = np.arange(0, 1 + 0.01, 0.01)

fig, ax = plt.subplots(4, 5, figsize = (22,16))
fig.subplots_adjust(hspace = 0.27)
ax = ax.ravel()

# plot f(a)
z_inverse = action_scaler.inverse_transform(z.reshape(-1, 1))[:,0]
ax[9].plot(z_inverse, mod[:,0], label="0")
ax[9].set_xlabel('z')
ax[9].set_ylabel('f(a)')
fig.suptitle("Maginal Effect, f(a) - df = {}, h = {}, degree = {}, train_mse = {}, test_mse = {}".format(df, h, degree, 
    round(train_mse,2), round(test_mse,2)))

for j in range(0, len(state)):
    # compute matrix of f_j values for each (s_j, a) value pair 
    pos = j

    func_grid = np.empty((len(z), len(s_j)))
    for i, z_val in enumerate(z):
        for m, s_val in enumerate(s_j):
            func_grid[i,m] = f_j(s_val, z_val, pos+1, z, df, degree, mod, spline_means, spline_knots)

    # create contour plot 
    s_inverse = invTransform(state_scaler, s_j, j, sK)
    Z, S = np.meshgrid(z_inverse, s_inverse)
    X = [z, s_j]
    sp = csaps.NdGridCubicSmoothingSpline(X, func_grid, smooth=0.8)
    func_smooth = sp(X)
    cs = ax[j].contourf(Z, S, func_smooth.T, cmap='hot_r')
    fig.colorbar(cs, ax=ax[j], shrink=0.7)
    ax[j].set_xlabel('z')
    ax[j].set_ylabel('s')
    ax[j].set_title("s = {}".format(state[j]))

for j in range(0, len(state)):
    # compute matrix of f_j values for each (s_j, a) value pair 
    func_grid = np.empty((len(z), len(s_j)))
    for i, z_val in enumerate(z):
        for m, s_val in enumerate(s_j):
            func_grid[i,m] = f_j(s_val, z_val, j+1, z, df, degree, mod, spline_means, spline_knots) + f(z_val, z, mod)

    # create contour plot 
    s_inverse = invTransform(state_scaler, s_j, j, sK)
    Z, S = np.meshgrid(z_inverse, s_inverse)
    X = [z, s_j]
    sp = csaps.NdGridCubicSmoothingSpline(X, func_grid, smooth=0.8)
    func_smooth = sp(X)
    cs = ax[j+10].contourf(Z, S, func_smooth.T, cmap='hot_r')
    fig.colorbar(cs, ax=ax[j+10], shrink=0.7)
    ax[j+10].set_xlabel('z')
    ax[j+10].set_ylabel('s')
    ax[j+10].set_title("s = {}".format(state[j]))

pdf.savefig(fig)
pdf.close()
