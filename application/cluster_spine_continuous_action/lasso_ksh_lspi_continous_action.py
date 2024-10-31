"""
An non-parametric, off-policy implementation of Least-Squares 
Policy Iteration (Lagoudakis & Parr 2003) using Group-Lasso regularization
"""
import numpy as np
import scipy.linalg
from numpy import linalg as LA
from patsy import bs
import random
import math
import itertools

def _R_compat_quantile(x, probs):
    probs = np.asarray(probs)
    quantiles = np.asarray([np.percentile(x, 100 * prob)
                            for prob in probs.ravel(order="C")])
    return quantiles.reshape(probs.shape, order="C")

def infer_knots(x, df, degree, padding = 0):
    # infer knots based on quantiles of x
    n_inner_knots = df - degree - 1 # include intercept
    if n_inner_knots < 0:
        raise ValueError("df=%r is too small for degree=%r"
                            % (df, degree))
    
    # Need to compute inner knots
    knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
    inner_knots = _R_compat_quantile(x, knot_quantiles)
    lower_bound = np.min(x) - padding
    upper_bound = np.max(x) + padding
    inner_knots = np.asarray(inner_knots)

    return(np.concatenate((inner_knots, [lower_bound, upper_bound])))


def get_b_spline_knots(model_matrix, sK, df, degree, padding = 0):
    state_matrix = model_matrix[:,:sK]
    state_knot = np.array([])
    for i in range(sK):
        s_vec = state_matrix[:,i]
        knot = infer_knots(s_vec, df, degree, padding)
        state_knot = np.append(state_knot, knot) 
    return(state_knot)

def get_b_spline_means(model_matrix, sK, df, degree, spline_knots): 
    """
    Compute B-Spline basis component means for state features (expect for first)
    
    Notes:
    - method uses the quantiles of state_feature to choose df-degree (internal) knots
    """
    state_matrix = model_matrix[:,:sK]
    state_mean = np.array([])
    iter = 0
    for i in range(sK):
        knots = spline_knots[int(iter*(df-degree+1)):int((iter + 1)*(df-degree+1))]
        s_matrix = state_matrix[:,i]
        mat_spline_comp = bs(s_matrix, degree=degree, knots=knots[:-2], 
            include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
        col_mean_spline = mat_spline_comp.mean(axis=0)
        state_mean = np.append(state_mean, col_mean_spline) 
        iter += 1 
    return(state_mean)

def b_spline_basis(state_matrix, sK, df, degree, spline_means=None, spline_knots=None, padding=0):
    """
    Implement hybrid B-spline basis expansion 
    """
    # set up and initialization
    n = state_matrix.shape[0]
    k = df*sK + 1
    basis_matrix = np.zeros((n,k))
    basis_matrix[:, 0] = 1

    # populate basis expansion matrix
    if spline_knots is not None and spline_means is not None:
        iter = 0
        for j in range(1, sK + 1): 
            knots = spline_knots[int(iter*(df-degree+1)):int((iter + 1)*(df-degree+1))]
            means = spline_means[int(iter*df):int((iter + 1)*df)]
            basis = bs(state_matrix[:,j-1], degree=degree, knots=knots[:-2], 
                include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
            basis_matrix[:, 1+(j-1)*df:1+j*df] = basis - basis.mean(axis=0)
            iter += 1 
    else:
        for j in range(1, sK + 1): 
            knots = infer_knots(state_matrix[:,j-1], df, degree, padding)
            basis = bs(state_matrix[:,j-1], degree=degree, knots=knots[:-2], 
                include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
            basis_matrix[:, 1+(j-1)*df:1+j*df] = basis - basis.mean(axis=0)

    return(basis_matrix)

def b_spline_basis_state(state, sK, df, degree, spline_means, spline_knots):
    """
    Implement B-spline basis function 
    """
    # Compute the basis function 
    temp = np.array([])
    k = df*sK + 1
    phi = np.zeros(k)
    iter = 0
    for i in state:
        knots = spline_knots[int(iter*(df-degree+1)):int((iter + 1)*(df-degree+1))]
        means = spline_means[int(iter*df):int((iter + 1)*df)]
        basis = bs(i, degree=degree, knots=knots[:-2], 
            include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
        temp = np.append(temp, basis - means)
        iter += 1 
    phi = np.append([1],temp)

    return(phi)

def nearest_z_policy(state_matrix, beta_matrix, z, sK, df, degree, padding=0):
    """
    For each point in z find nearest values in state_matrix, apply b_spline_policy and
    Calculate greedy policy defined as argmax_{a in A} Q^{pi}(s,a) for the hybrid b-spline 
    basis matrix (across multiple observations)

    returns: optimal actions (N x 1)
    -----------
    """
    # initize np array
    nrows = state_matrix.shape[0]
    optimal_act = np.zeros((nrows, 1))

    # get b-spline basis matrix
    basis_matrix = b_spline_basis(state_matrix, sK, df, degree, padding)

    # get index closest point in z for each data point
    q_value_matrix = basis_matrix @ beta_matrix.T

    # get position of optimal q-value for each observation
    maximal_pos = q_value_matrix.argmax(axis=1)

    # get the associated actions using the position of maximal q-value
    optimal_act = z[maximal_pos]

    return(optimal_act)

def soft_threshold(v, lambda_reg, mu):
    #if LA.norm(v,2) < lambda_reg * mu:
    #    return(v * 0)
    #else:
    #    return(v - ((v * lambda_reg * mu) / ( LA.norm(v,2))))
    norm = LA.norm(v,2)
    if norm == 0:
        norm_v = v*0
    else:
        norm_v = v/norm
    return((norm_v)*np.max((0,norm-lambda_reg * mu)))

def predict_action_values(phi_s, actions, beta, z):
    """
    Compute the action-values for a given basis expanded state matrix 
    """
    # intialize arrays
    nrows = phi_s.shape[0]
    q_values = np.zeros(nrows)
    num_z = len(z)

    # get index closest point in z for each data point
    repeated_action = np.transpose([actions] * num_z)
    repeated_z = np.tile(z, (phi_s.shape[0],1))
    index_vector = np.argmin(np.abs(repeated_action - repeated_z), axis = 1)
    
    # compute q-values 
    q_values = (phi_s * beta[index_vector,:]).sum(axis=1)
    
    return(q_values)

def ksh_lstdq(phi_s, phi_ns, curr_actions, opt_actions, r, d_z, discount, beta, reg, mu, pos, z, z_iter):
    """
    Implement Group LASSO  Kernel Sieve Hybrid LSTDQ to learn the state-action value function
    -----------
    - input: model_matrix, discount factor, degree of spline polynomial, policy
    - beta: group lasso regularization term
    """

    # copy beta
    updated_beta = np.array(beta)

    # compute current and next-state, action q-values
    current_q_val = predict_action_values(phi_s, curr_actions, updated_beta, z)
    next_q_val = predict_action_values(phi_ns, opt_actions, updated_beta, z)

    # update beta using soft-threshold operator
    val = updated_beta[z_iter, pos] - (mu * phi_s[:,pos].T  @ d_z @ ((current_q_val - discount * next_q_val)  - r))
    result = soft_threshold(np.array([val]), reg, mu)

    return(result)

def group_lasso_ksh_lspi(data, discount, sK, init_beta, z, mod_eps=10**-5, 
    mod_max_iter=1000, policy_eps=10**-3, policy_max_iter=10, lambda_reg=0, degree=1, 
    mu = 0.001, h = 0.4, df = 1, padding = 1, behavior_policy_init = False):
    """
    Implement Group LASSO  Kernel Sieve Hybrid- LSPI 
    -----------
    - data: np array with columns: state (n x sK), action (n x 1), reward (n x 1), next_state (n x sK)]
    - input: data, discount factor, degree of poly. basis, policy
    - z is a grid of points to evaluate the model on
    - lambda_reg: regularization constant for group LASSO
    - mu: step size
    - h: guassian kernel bandwidth parameter
    - df: number of B-Spline basis functions 
    - init_beta: matrix (|Z| x k) of initial beta values for each point in Z 
    """

    # initialize current policy
    beta_matrix = np.array(init_beta)

    # compute basis matrix 
    phi_s = b_spline_basis(data[:,:sK], sK, df, degree)

    # next state matrix
    ns_matrix = data[:,-sK:]

    if behavior_policy_init:
        # next state matrix. 
        ns_matrix = data[:,-(sK+1):-1]
    else:
        # next state matrix
        ns_matrix = data[:,-sK:]

    # current actions
    curr_actions = data[:,sK]

    # dimension of colspace phi
    k = df*sK + 1

    # policy iteration loop - continue until beta matrix converges
    policy_iter = 0 
    policy_dist = float('inf')
    while policy_dist > policy_eps and policy_iter < policy_max_iter:
        # for each point in the grid
        z_iter = 0
        for i in z:
            if behavior_policy_init and policy_iter == 0:
                # next actions 
                next_act = data[:,-1]
            else:
                # compute next state actions using current policy (i.e. beta_matrix)
                next_act = nearest_z_policy(ns_matrix, beta_matrix, z, sK, df, degree, padding=padding)

            # compute next state matrix 
            phi_ns = b_spline_basis(ns_matrix, sK, df, degree)

            # compute projection matrix 
            d_z = np.diag(np.exp(-(data[:,sK]-i)**2/(2*h)))

            r = data[:,sK + 1]
            # set model iteration to zero and z parameter dist to inf
            mod_iter = 0 
            mod_dist = float('inf')

            while mod_dist > mod_eps and mod_iter < mod_max_iter:
                #  select model coordinate
                for j in range(sK + 1):

                    # Check if coordinate corresponds to intercept.   
                    if j == 0:
                        pos = 0
                        reg = lambda_reg * math.sqrt(df)
                    else:
                        pos = [(j-1)*df+ 1 + k for k in range(0, df)]
                        reg = lambda_reg

                    # update beta
                    temp_beta = ksh_lstdq(phi_s, phi_ns, curr_actions, next_act, r, d_z, discount, beta_matrix, reg, \
                        mu, pos, z, z_iter)
                    mod_dist = np.linalg.norm(temp_beta - beta_matrix[z_iter, pos])
                    beta_matrix[z_iter,pos] = temp_beta

                # increment counter
                mod_iter += 1
 
            # increment z iter
            z_iter += 1
        
        # compare current policy to previous
        policy_dist = np.linalg.norm(init_beta - beta_matrix)
        init_beta = np.array(beta_matrix)
        print(policy_dist, policy_iter)
        policy_iter += 1

    return(beta_matrix)

