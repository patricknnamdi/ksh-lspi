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

    return np.concatenate((inner_knots, [lower_bound, upper_bound]))


def get_b_spline_knots(model_matrix, nA, sK, df, degree, padding = 0):
    """
    Notes:
    -  does not compute mean for first state feature 
    """
    state_matrix = model_matrix[:,1:sK+1]
    spline_sa_knots = np.array([])
    for j in range(nA):
        state_knot = np.array([])
        for i in range(sK-1):
            sa_vec = state_matrix[state_matrix[:,-1] == j,i]
            knot = infer_knots(sa_vec, df, degree, padding)
            state_knot = np.append(state_knot, knot) 
        spline_sa_knots = np.vstack([spline_sa_knots, state_knot]) if spline_sa_knots.size else state_knot
    return spline_sa_knots

def get_b_spline_means(model_matrix, nA, sK, df, degree, spline_knots): 
    """
    Compute B-Spline basis component means for state features (expect for first)
    
    Notes:
    - method uses the quantiles of state_feature to choose df-degree (internal) knots
    - does not compute mean for first state feature 
    """
    state_matrix = model_matrix[:,1:sK+1]
    spline_sa_means = np.array([])
    for j in range(nA): 
        state_mean = np.array([])
        iter = 0
        for i in range(sK-1):
            knots = spline_knots[j,int(iter*(df-degree+1)):int((iter + 1)*(df-degree+1))]
            sa_matrix = state_matrix[state_matrix[:,-1] == j,i]
            mat_spline_comp = bs(sa_matrix, degree=degree, knots=knots[:-2], 
                include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
            col_mean_spline = mat_spline_comp.mean(axis=0)
            state_mean = np.append(state_mean, col_mean_spline) 
            iter += 1 
        spline_sa_means = np.vstack([spline_sa_means, state_mean]) if spline_sa_means.size else state_mean
    return spline_sa_means 

def b_spline_basis(state_action_matrix, nA, sK, df, degree, padding=0):
    """
    Implement hybrid B-spline basis expansion 
    """
    # set up and initialization
    n = state_action_matrix.shape[0]
    k = (df*(sK-1) + 1) * nA
    basis_matrix = np.zeros((n,k))

    # populate basis expansion matrix
    for j in range(0, sK): 
        for i in range(nA):
            index = (state_action_matrix[:,-1]==i)
            if sum(index) != 0:
                if j == 0:
                    indx = int(k/2)*i
                    basis_matrix[index, indx] = 1
                else:
                    knots = infer_knots(state_action_matrix[index,j], df, degree, padding)
                    basis = bs(state_action_matrix[index,j], degree=degree, knots=knots[:-2], 
                        include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
                    indx = int(k/2)*i
                    basis_matrix[index, 1+indx+(j-1)*df:1+indx+j*df] = basis - basis.mean(axis=0)
    return basis_matrix

def b_spline_basis_state(state, action, nA, sK, df, degree, spline_means, spline_knots):
    """
    Implement B-spline basis function 
    """
    # Compute the basis function 
    temp = np.array([])
    phi_size = (df*(sK-1) + 1) * nA
    phi = np.zeros(phi_size)
    iter = 0
    for i in state[1:]:
        knots = spline_knots[action,int(iter*(df-degree+1)):int((iter + 1)*(df-degree+1))]
        means = spline_means[action,int(iter*df):int((iter + 1)*df)]
        basis = bs(i, degree=degree, knots=knots[:-2], 
            include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
        temp = np.append(temp, basis - means)
        iter += 1 
    sA_index = int((df*(sK-1) + 1) * action)
    phi[sA_index:sA_index + (df*(sK-1) + 1)] = np.append([1],temp)

    return phi

def b_spline_policy_state(state, w, nA, sK, df, degree, spline_means, spline_knots): 
    """
    Calculate greedy policy defined as argmax_{a in A} Q^{pi}(s,a)
    for the b-spline basis function
    -----------
    - use cat_cols = [] if no categorical columns 
    """
    # Calculate Q-value for each action a 
    q_values = [w.dot(b_spline_basis_state(state, action, nA, sK, df, 
        degree, spline_means, spline_knots)) for action in range(nA)]

    # Choose optimal action
    optimal_action = np.argmax(q_values)

    return optimal_action

def nearest_z_policy_state(state, beta_matrix, z, nA, sK, df, degree, spline_means, spline_knots):
    """ 
    Compute the optimal action for a given state vector 
    """
    # determine best local model 
    index = np.argmin(np.abs(state[0] - z))

    # select model from array of z points
    beta = beta_matrix[index,:]

    # get optimal action 
    optimal_action = b_spline_policy_state(state, beta, nA, sK, df, degree, spline_means, spline_knots)

    return optimal_action


def nearest_z_policy(state_matrix, beta_matrix, z, nA, sK, df, degree, padding=0):
    """
    For each point in z find nearest values in state_matrix and apply b_spline_policy
    returns: optimal actions (N x 1)
    -----------
    """
    # initize np array
    nrows = state_matrix.shape[0]
    optimal_act = np.zeros((nrows, 1))
    num_z = len(z)

    # get index closest point in z for each data point
    repeated_first_state = np.transpose([state_matrix[:,0]] * num_z)
    repeated_z = np.tile(z, (state_matrix.shape[0],1))
    index_vector = np.argmin(np.abs(repeated_first_state - repeated_z), axis = 1)

    for i in range(num_z):
        temp_act = b_spline_policy(state_matrix[index_vector == i,:], beta_matrix[i,:], 
            nA, sK, df, degree, padding)
        optimal_act[index_vector == i,0] = temp_act

    return optimal_act

def b_spline_policy(state_matrix, beta, nA, sK, df, degree, padding=0): 
    """
    Calculate greedy policy defined as argmax_{a in A} Q^{pi}(s,a)
    for the hybrid b-spline basis matrix (across multiple observations)
    -----------
    """
    # Calculate Q-value for each action over all n obs
    nrows = state_matrix.shape[0]
    q_values = np.empty((nrows, nA))

    for action in range(nA):
        # compute state-action matrix
        actions = np.full(nrows, action).reshape(nrows,1)
        state_action_matrix = np.append(state_matrix, actions, axis = 1)

        # get hybrid b-spline basis matrix
        basis_matrix = b_spline_basis(state_action_matrix, nA, sK, df, degree, padding)

        # compute q-values for all n
        q_values[:,action] = np.dot(basis_matrix, beta)

    # Choose optimal action
    optimal_actions = np.argmax(q_values, axis=1)

    return optimal_actions

# predict q-value 
def predict_action_values(state_action_matrix, beta_matrix, z, nA, sK, df, degree, padding=0): 
    """ 
    Compute the action-values for a given state-action matrix 
    """
     # initialize q-values
    nrows = state_action_matrix.shape[0]
    q_values = np.zeros(nrows)
    num_z = len(z)

    # get index closest point in z for each data point
    repeated_first_state = np.transpose([state_action_matrix[:,0]] * num_z)
    repeated_z = np.tile(z, (state_action_matrix.shape[0],1))
    index_vector = np.argmin(np.abs(repeated_first_state - repeated_z), axis = 1)
    
    # loop through all models 
    for i in range(num_z):
        # get hybrid b-spline basis matrix
        basis_matrix = b_spline_basis(state_action_matrix[index_vector == i,:], nA, sK, df, degree, padding)

        # compute action-values 
        q_values[index_vector == i] = np.dot(basis_matrix, beta_matrix[i,:])

    return q_values

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

def ksh_lstdq(phi_sa, phi_nsa, r, d_z, discount, df, degree, nA, sK, beta, lambda_reg, h, mu, j):
    """
    Implement Group LASSO  Kernel Sieve Hybrid LSTDQ to learn the state-action value function
    -----------
    - input: model_matrix, discount factor, degree of spline polynomial, policy
    - beta: group lasso regularization term
    """ 
    
    # check if coordinate is first feature or a spline feature 
    size = len(beta) # beta of size (df*(sK-1) + 1) * nA
    if j == 0:
        pos = [int(size / nA)*i for i in range(0,nA)]
        reg = lambda_reg * math.sqrt(df)
    else:
        pos = [[(int(size / nA)*i + (j-1)*df+1) + k for k in range(0, df)] for i in range(0,nA)]
        reg = lambda_reg

    # initialize new beta 
    updated_beta = np.array(beta)

    # select actions
    i = int(np.random.randint(nA, size = 1))
    #err = (psi @ updated_beta) - y

    # update beta using soft-threshold operator 
    #val = updated_beta[pos[i]] - ((mu/psi.shape[0]) * (psi[:,pos[i]].T @ d_z @ ((psi @ updated_beta) - y)))
    val = updated_beta[pos[i]] - ((mu) * phi_sa[:,pos[i]].T  @ d_z @ (((phi_sa - discount * phi_nsa) @ updated_beta) - r))
    updated_beta[pos[i]] = soft_threshold(np.array([val]), reg, mu)

    return updated_beta 

def group_lasso_ksh_lspi(data, discount, sK, nA, init_beta, z, mod_eps=10**-5, 
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
    phi_sa = b_spline_basis(data[:,:sK+1], nA, sK, df, degree)

    if behavior_policy_init:
        # next state-action matrix
        nsa_matrix = data[:,-(sK+1):]
        ns_matrix = data[:,-(sK+1):-1]
    else:
        # next state matrix
        ns_matrix = data[:,-sK:]

    # dimension of colspace phi
    k = (df*(sK-1) + 1) * nA

    # policy iteration loop - continue until beta matrix converges
    policy_iter = 0 
    policy_dist = float('inf')
    while policy_dist > policy_eps and policy_iter < policy_max_iter:
        # for each point in the grid
        z_iter = 0
        for i in z:

            if behavior_policy_init and policy_iter == 0:
                phi_nsa = b_spline_basis(nsa_matrix, nA, sK, df, degree)
            else: 
                # compute next state actions using current policy (i.e. beta_matrix)
                opt_act = nearest_z_policy(ns_matrix, beta_matrix, z, nA, sK, 
                    df, degree, padding=padding)

                # compute next state matrix 
                nsa_matrix = np.append(ns_matrix, opt_act, axis = 1)
                phi_nsa = b_spline_basis(nsa_matrix, nA, sK, df, degree)

            # compute projection matrix 
            d_z = np.diag(np.exp(-(data[:,0]-i)**2/(2*h)))

            r = data[:,sK + 1]
            # set model iteration to zero and z parameter dist to inf
            mod_iter = 0 
            mod_dist = float('inf')

            while mod_dist > mod_eps and mod_iter < mod_max_iter:
                #  select model coordinate 
                for j in range(sK): 
                    #j = random.randint(0, sK-1)
                    temp_beta = ksh_lstdq(phi_sa, phi_nsa, r, d_z, discount, df, degree, nA, sK, beta_matrix[z_iter,:], lambda_reg, \
                        h, mu, j)
                    mod_dist = np.linalg.norm(temp_beta - beta_matrix[z_iter,:])
                    beta_matrix[z_iter,:] = temp_beta
                    #print(mod_dist, j)
                mod_iter += 1
            
            # increment z iter
            z_iter += 1
        
        # compare current policy to previous
        policy_dist = np.linalg.norm(init_beta - beta_matrix)
        init_beta = np.array(beta_matrix)
        print(policy_dist, policy_iter)
        policy_iter += 1

    return beta_matrix

def l2_ksh_lspi(data, discount, sK, nA, init_beta, z, policy_eps=10**-3, policy_max_iter=10,
    lambda_reg=0, degree=1, h = 0.4, df = 1, precondition = 0.00001):
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
    phi_sa = b_spline_basis(data[:,:sK+1], nA, sK, df, degree)

    # next state matrix
    ns_matrix = data[:,-sK:]

    # dimension of colspace phi
    k = (df*(sK-1) + 1) * nA

    # policy iteration loop - continue until beta matrix converges
    policy_iter = 0 
    policy_dist = float('inf')
    while policy_dist > policy_eps and policy_iter < policy_max_iter:
        # for each point in the grid
        z_iter = 0
        for i in z:
            # compute next state actions using current policy (i.e. beta_matrix)
            opt_act = nearest_z_policy(ns_matrix, beta_matrix, z, nA, sK, df, degree)

            # compute next state matrix 
            nsa_matrix = np.append(ns_matrix, opt_act, axis = 1)
            phi_nsa = b_spline_basis(nsa_matrix, nA, sK, df, degree)

            # compute projection matrix 
            d_z = np.diag(np.exp(-(data[:,0]-i)**2/(2*h)))
            precond = np.zeros((k, k))
            np.fill_diagonal(precond, precondition)
            inv_gram = np.linalg.inv(precond + phi_sa.T @ d_z @ phi_sa) 
            proj_mat = phi_sa @ inv_gram @ phi_sa.T @ d_z

            # compute projected reward vector and Psi 
            y = proj_mat @ data[:,sK + 1]
            psi = phi_sa - discount * (proj_mat @ phi_nsa)

            # L2 solution 
            precond2 = np.zeros((k, k))
            np.fill_diagonal(precond2, precondition)
            inv_gram_psi = np.linalg.inv(precond2 + psi.T @ psi + np.diag(np.ones(k) + lambda_reg)) 
            l2_sol = inv_gram_psi @ psi.T @ y
            beta_matrix[z_iter,:] = l2_sol
            print(z_iter)
            # increment z iter
            z_iter += 1

        # compare current policy to previous
        policy_dist = np.linalg.norm(init_beta - beta_matrix)
        init_beta = np.array(beta_matrix)
        print(policy_dist, policy_iter)
        policy_iter += 1

    return beta_matrix 



def group_lasso_supervised(data, discount, sK, nA, init_beta, eps=10**-8, 
    max_iter=10, lambda_reg=0, degree=1, mu = 0.001, z = 0, h = 0.4, df = 1, precondition = 0.0001):
    """
    Implement Group LASSO - for normal supervised learning objective 
    -----------
    - data: np array with columns - data[state + [action] + [reward] + next_state]
    - input: data, discount factor, degree of poly. basis, policy
    - action: colname of data that represents actions 
    - lambda_reg: regularization constant for group LASSO
    """

    # initialize current policy
    beta = init_beta

    # update w using HKS-LSTDQ with group-lasso
    dist = float('inf')
    iter = 0 

    # subset model_matrix 
    # model_matrix = data[state + [action] + [reward] + next_state]

    # compute basis matrix 
    phi_sa = b_spline_basis(data[:,:sK+1], nA, sK, df, degree)

    # compute next state matrix (here matrix of random betas)
    

    # compute projection matrix 
    d_z = np.diag(np.exp(-(data[:,0]-z)**2/(2*h)))
    k = (df*(sK-1) + 1) * nA
    precond = np.zeros((k, k))
    np.fill_diagonal(precond, precondition)
    inv_gram = np.linalg.inv(precond + phi_sa.T @ d_z @ phi_sa) 
    proj_mat = phi_sa @ inv_gram @ phi_sa.T @ d_z

    # compute projected reward vector and Psi 
    y = proj_mat @ data[:,sK + 1] 
    psi = phi_sa #- discount * (proj_mat @ phi_nsa)

    while dist > eps and iter < max_iter:
        #  select model coordinate 
        j = random.randint(0, sK-1)
        temp_beta = ksh_lstdq(psi, y, d_z, discount, df, degree, nA, sK, beta, lambda_reg, \
             h, mu, j)
        dist = np.linalg.norm(temp_beta - beta)
        beta = temp_beta
        print(dist, j)
        iter += 1

    return beta

