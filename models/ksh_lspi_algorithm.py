import numpy as np
from patsy import bs
import multiprocessing as mp
import random
from numpy import linalg as LA
import math
from formulaic.transforms import basis_spline


class KSH_LSPI:
    """
    Implements Kernel Sieve Hybrid Least-Squares Policy Iteration (KSH-LSPI) for
    continuous state and continuous action spaces.
    """

    def __init__(self, data, num_actions, state_dim, df, degree,
                 local_centers, kernel=None, bandwidth=0.005, lambda_reg=0.1,
                 discount=0.9, epsilon=1e-6, max_iter=100, mu=0.001,
                 max_policy_iter=10, model_type='discrete-action', num_processes=None, 
                 basis_type='bspline'):
        """

        Args:
            data (numpy.ndarray): Array of data samples. Each sample is a tuple 
                of (state, action, reward, next_state).
            num_actions (int): Number of actions in the discrete action space.
            state_dim (int): Dimensionality of the state space.
            df (int): Degrees of freedom for the B-spline basis functions.
            degree (int): Degree of the B-spline basis functions.
            local_centers (array-like): Array of local centers for the 
                kernel-based model.
            kernel (callable, optional): Kernel function to use for computing 
                kernel weights. Defaults to the Gaussian kernel.
            bandwidth (float, optional): Bandwidth of the kernel function. 
                Defaults to 0.005.
            lambda_reg (float, optional): Regularization parameter for the group
                lasso penalty. Defaults to 0.1.
            discount (float, optional): Discount factor for future rewards. 
                Defaults to 0.9.
            epsilon (float, optional): Tolerance for convergence of the 
                coordinate descent algorithm. Defaults to 1e-6.
            max_iter (int, optional): Maximum number of iterations for 
                coordinate descent. Defaults to 100.
            mu (float, optional): Step size for coordinate descent algorithm. 
                Defaults to 0.001.
            max_policy_iter (int, optional): Maximum number of policy
                iterations. Defaults to 10.
            model_type (str, optional): Type of model, either 'discrete-action' 
                or 'continuous-action'. Defaults to 'discrete-action'.
            num_processes (int, optional): Number of processes to use for 
                parallelization. Defaults to one less than the number of CPU 
                cores.
            basis_type (str, optional): 'bspline' or 'trig' (defaults to 
                'bspline').
        """

        self.num_processes = mp.cpu_count() - 1 if num_processes is None else num_processes
        self.kernel = self.gaussian_kernel if kernel is None else kernel
        self.h = bandwidth
        self.discount = discount
        self.epsilon = epsilon
        self.mu = mu
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.max_policy_iter = max_policy_iter
        self.local_centers = local_centers
        self.model_type = model_type
        self.num_actions = num_actions if model_type == 'discrete-action' else 1
        self.state_dim = state_dim
        self.df = df
        self.degree = degree
        self.basis_type = basis_type

        if self.basis_type == 'bspline':
            self.spline_knots = self.get_b_spline_knots(data)
            self.spline_means = self.get_b_spline_means(data, self.spline_knots)
            self.feature_dim = (df * (state_dim - 1) + 1) * num_actions
        elif self.basis_type == 'trig':
            self.trig_means = self.get_trig_means(data)  # Pre-compute trig means
            self.feature_dim = ((2 * df) * (state_dim - 1) + 1) * num_actions  # Corrected feature_dim
        else:
            raise ValueError("Invalid basis type. Choose either 'bspline' or 'trig'.")

        self.weights = np.random.uniform(-2, 2, size=(len(local_centers),
                                                       self.feature_dim))
        self.basis_cache = {}

    
    def gaussian_kernel(self, x, y):
        """
        Computes the Gaussian kernel.

        Args:
            x (numpy.ndarray): First vector.
            y (numpy.ndarray): Second vector.

        Returns:
            float: Value of the Gaussian kernel.
        """
        return np.exp(-np.linalg.norm(x - y)**2 / (2 * self.h))

    def _R_compat_quantile(self, x, probs):
        """
        Computes quantiles of a given array.

        Args:
            x (numpy.ndarray): Array of values.
            probs (array-like): Array of probabilities.

        Returns:
            numpy.ndarray: Array of quantiles.
        """
        probs = np.asarray(probs)
        quantiles = np.asarray([np.percentile(x, 100 * prob)
                                for prob in probs.ravel(order="C")])
        return quantiles.reshape(probs.shape, order="C")

    def infer_knots(self, x, padding = 10):
        """
        Infers knots for the B-spline basis functions.

        Args:
            x (numpy.ndarray): Array of values.
            padding (float, optional): Padding for the knot boundaries. 
                Defaults to 10.

        Returns:
            numpy.ndarray: Array of knots.
        """
        # infer knots based on quantiles of x
        n_inner_knots = self.df - self.degree - 1 # include intercept
        if n_inner_knots < 0:
            raise ValueError("df=%r is too small for degree=%r"
                                % (self.df, self.degree))
        
        # Need to compute inner knots
        knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
        inner_knots = self._R_compat_quantile(x, knot_quantiles)
        lower_bound = np.min(x) - padding
        upper_bound = np.max(x) + padding
        inner_knots = np.asarray(inner_knots)

        return np.concatenate((inner_knots, [lower_bound, upper_bound]))

    def get_b_spline_knots(self, data, padding = 10):
        """
        Computes knots for the B-spline basis functions based on data.

        Args:
            data (numpy.ndarray): Array of data samples.
            padding (float, optional): Padding for the knot boundaries. 
                Defaults to 10.

        Returns:
            numpy.ndarray: Array of knots.

        Notes:
            -  does not compute mean for first state feature 
        """
        state_matrix = data[:,1:self.state_dim+1]
        spline_sa_knots = np.array([])
        for j in range(self.num_actions):
            state_knot = np.array([])
            for i in range(self.state_dim-1):
                sa_vec = state_matrix[state_matrix[:,-1] == j,i]
                knot = self.infer_knots(sa_vec, padding)
                state_knot = np.append(state_knot, knot) 
            spline_sa_knots = np.vstack([spline_sa_knots, state_knot]) if spline_sa_knots.size else state_knot
        return spline_sa_knots
    
    def get_b_spline_means(self, data, spline_knots): 
        """
        Compute B-Spline basis component means for state features (expect for first)
        
        Args:
            data (numpy.ndarray): Array of data samples.
            spline_knots (numpy.ndarray): Array of knots for the B-spline basis functions.

        Returns:
            numpy.ndarray: Array of means for the B-spline basis functions, 
                shape: (num_actions, (state_dim - 1) * df).

        Notes:
            - method uses the quantiles of state_feature to choose df-degree (internal) knots
            - does not compute mean for first state feature 
        """
        state_matrix = data[:,1:self.state_dim+1]
        spline_sa_means = np.array([])
        for j in range(self.num_actions): 
            state_mean = np.array([])
            iter = 0
            for i in range(self.state_dim-1):
                knots = spline_knots[j,int(iter*(self.df-self.degree+1)):int((iter + 1)*(self.df-self.degree+1))]
                sa_matrix = state_matrix[state_matrix[:,-1] == j,i]
                mat_spline_comp = bs(sa_matrix, degree=self.degree, knots=knots[:-2], 
                    include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
                col_mean_spline = mat_spline_comp.mean(axis=0)
                state_mean = np.append(state_mean, col_mean_spline) 
                iter += 1 
            spline_sa_means = np.vstack([spline_sa_means, state_mean]) if spline_sa_means.size else state_mean
        return spline_sa_means 

    # def get_features(self, state, action):
    #     """
    #     Implement B-spline basis function 
    #     """
    #     # Compute the basis function 
    #     temp = np.array([])
    #     phi_size = (self.df*(self.state_dim-1) + 1) * self.num_actions
    #     phi = np.zeros(phi_size)
    #     iter = 0
    #     for i in state[1:]:
    #         knots = self.spline_knots[int(action),int(iter*(self.df-self.degree+1)):int((iter + 1)*(self.df-self.degree+1))]
    #         means = self.spline_means[int(action),int(iter*self.df):int((iter + 1)*self.df)]
    #         basis = bs(i, degree=self.degree, knots=knots[:-2], 
    #             include_intercept=True, upper_bound=knots[-1], lower_bound=knots[-2])
    #         temp = np.append(temp, basis - means)
    #         iter += 1 
    #     sA_index = int((self.df*(self.state_dim-1) + 1) * int(action))
    #     phi[sA_index:sA_index + (self.df*(self.state_dim-1) + 1)] = np.append([1],temp)

    #     return phi

    def trig_basis(self, x, df):
        """
        Computes the trigonometric polynomial basis function values for a given x.

        Args:
            x (float): Input value.
            df (int): Degrees of freedom.

        Returns:
            numpy.ndarray: Array of basis function values.
        """
        # Note that the constant term is not included in the trigonometric basis
        k = np.arange(1, df + 1)
        sin_values = np.sin(np.outer(x, k))
        cos_values = np.cos(np.outer(x, k))
        return np.hstack((sin_values, cos_values))

    def get_trig_means(self, data):
        """
        Computes the means of the centered trigonometric basis functions for 
        each state dimension and action.

        Args:
            data (numpy.ndarray): Array of data samples.

        Returns:
            numpy.ndarray: Array of means, 
                shape: (num_actions, (state_dim - 1) * 2 * df).
        """
        state_matrix = data[:, 1:self.state_dim + 1]
        trig_means = np.zeros((self.num_actions, 
                               (self.state_dim - 1) * 2 * self.df))
        
        for j in range(self.num_actions):
            action_data = state_matrix[state_matrix[:, -1] == j] 
            for i in range(self.state_dim - 1):
                trig_mean = np.mean(self.trig_basis(action_data[:, i], 
                                                    self.df), axis=0)
                start_idx = i * 2 * self.df
                end_idx = start_idx + 2 * self.df
                trig_means[j, start_idx:end_idx] = trig_mean
        return trig_means

    def get_features(self, state, action):
        """
        Computes the feature vector for a given state and action pair using 
        the chosen basis function type (B-spline or trigonometric).

        For both types, the features are constructed as follows:
            1. The basis functions are computed for each state 
                dimension (excluding the first).
            2. The basis functions are then centered using pre-computed means 
                for the given action.
            3. A constant term (1) is added to the feature vector.
            4. The features are assembled into the phi vector, with each action 
                having a dedicated block of features.

        Args:
            state (numpy.ndarray): The state vector.
            action (int): The chosen action.

        Returns:
            numpy.ndarray: The feature vector representing the state-action pair.
        """
        action = int(action)
        state_action_key = (tuple(state), action)
        if state_action_key in self.basis_cache:
            return self.basis_cache[state_action_key]

        phi = np.zeros(self.feature_dim)
        if self.basis_type == 'bspline':
            temp = np.zeros((self.state_dim - 1, self.df))
            # phi_size = (self.df * (self.state_dim - 1) + 1) * self.num_actions
            # phi = np.zeros(phi_size)
            iter = 0
            for i in range(1, self.state_dim):
                knots = self.spline_knots[action,
                                            int(iter * (self.df - self.degree + 1)):int(
                    (iter + 1) * (self.df - self.degree + 1))]
                means = self.spline_means[action,
                                            int(iter * self.df):int((iter + 1) * self.df)]
                basis = basis_spline(state[i], degree=self.degree, knots=knots[:-2],
                                        include_intercept=True, 
                                        upper_bound=knots[-1], lower_bound=knots[-2])
                temp[iter] = np.fromiter(basis.values(), dtype=float) - means
                iter += 1
            sA_index = int((self.df * (self.state_dim - 1) + 1) * action)
            phi[sA_index:sA_index + (self.df * (self.state_dim - 1) + 1)] = np.append([1], temp.ravel())

        elif self.basis_type == 'trig':
            phi = np.zeros(self.feature_dim) 
            temp = np.array([])
            for i in range(1, self.state_dim):
                basis = self.trig_basis(state[i], self.df)
                start_idx = (i-1) * 2 * self.df
                end_idx = start_idx + 2 * self.df
                means = self.trig_means[action, start_idx: end_idx]
                centered_basis = basis - means
                temp = np.append(temp, centered_basis)
            sA_index = int(((2 * self.df) * (self.state_dim - 1) + 1) * action)
            phi[sA_index:sA_index + ((2 * self.df) * (self.state_dim - 1)) + 1] = np.append([1], temp)

        self.basis_cache[state_action_key] = phi
        return phi


    def get_action_value(self, state, action):
        """
        Computes the action value for a given state and action.

        Args:
            state (numpy.ndarray): State vector.
            action (int or float): Action value.

        Returns:
            float: Action value.
        """

        # Determine best local model 
        index = np.argmin(np.abs(state[0] - self.local_centers))

        # Select model from array of z points
        beta = self.weights[index,:]

        # Return Q-value
        return np.dot(beta, self.get_features(state, action))
    
    def get_marginal_component(self, state, action):
        """
        Computes the marginal component of the model for a given state and action.

        Args:
            state (numpy.ndarray): State vector.
            action (int or float): Action value.

        Returns:
            float: Marginal component.
        """
        # Determine best local model
        index = np.argmin(np.abs(state[0] - self.local_centers))

        # Select model from array of z points
        beta = self.weights[index,:]

        # Index beta by action
        beta = beta[int((self.df*(self.state_dim-1) + 1) * int(action)):int((self.df*(self.state_dim-1) + 1) * (int(action) + 1))]

        # Get first coefficSient of beta
        return beta[0]

    def get_kernel_weight(self, state, action, local_center):
        """
        Computes the kernel weight for a given state, action, and local center.

        Args:
            state (numpy.ndarray): State vector.
            action (int or float): Action value.
            local_center (float): Local center.

        Returns:
            float: Kernel weight.
        """
        # If disrete action model, compute kernel weight centered at local center
        # using the first state feature
        if self.model_type == 'discrete-action':
            return self.kernel(state[0], local_center)
        # If continuous action model, compute kernel weight centered at local center
        # using the continuous action
        else:
            return self.kernel(action, local_center)
    
    def select_action(self, state):
        """
        Selects the best action for a given state.

        Args:
            state (numpy.ndarray): State vector.

        Returns:
            int or float: Action value.
        """
        return np.argmax([self.get_action_value(state, a) for a in range(self.num_actions)])
    
    def compute_single_sample_A_b(self, sample, local_center, behavioral_init):
        """
        Computes A and b matrices for a single sample.

        Args:
            sample (numpy.ndarray): Single data sample.
            local_center (float): Local center.
            behavioral_init (bool): Whether to use the behavioral policy for the next action.

        Returns:
            tuple: A and b matrices.
        """
        # Compute A and b for a single sample for parallelization.
                        # Name variables: state, action, reward, next_state
        state = sample[:self.state_dim]
        action = sample[self.state_dim]
        reward = sample[self.state_dim + 1]
        next_state = sample[self.state_dim + 2:-1]
        
        kernel_weight = self.get_kernel_weight(state, action, local_center)
        features = self.get_features(state, action)
        next_action = self.select_action(next_state) if not behavioral_init else sample[-1]
        next_features = self.get_features(next_state, next_action)

        A = np.outer(features, kernel_weight * (features - self.discount * next_features)) 
        b = features * kernel_weight * reward

        return A, b
    
    def soft_threshold(self, v, lambda_reg, mu):
        """
        Applies soft thresholding to a vector.

        Args:
            v (numpy.ndarray): Vector to apply soft thresholding to.
            lambda_reg (float): Regularization parameter.
            mu (float): Step size.

        Returns:
            numpy.ndarray: Soft thresholded vector.
        """
        norm = LA.norm(v, 2)
        if norm == 0:
            return np.zeros_like(v)
        else:
            return np.maximum(0, norm - lambda_reg * mu) * (v / norm)

    def learn(self, samples, local_center, behavioral_init=False, debiasing=False):
        """
        Learns the model for a given local center.

        Args:
            samples (numpy.ndarray): Array of data samples.
            local_center (float): Local center.
            behavioral_init (bool, optional): Whether to use the behavioral 
                policy for the next action. Defaults to False.
            debiasing (bool, optional): Whether to perform debiasing after 
                learning. Defaults to False.
        """
        # Determine local model to update 
        index = np.argmin(np.abs(local_center - self.local_centers))
        
        # Initialize A and b
        A = np.zeros((self.feature_dim, self.feature_dim))
        b = np.zeros(self.feature_dim)

        # # Setup multiprocessing pool
        # with mp.Pool(processes=self.num_processes) as pool:
        #     results = pool.starmap(self.compute_single_sample_A_b, 
        #                            [(sample, local_center, behavioral_init) for sample in samples])
        
        # # Sum up A and b
        # for result in results:
        #     A += result[0]
        #     b += result[1]

        # Compute A and b for all samples without parallelization.
        for sample in samples:
            single_A, single_b = self.compute_single_sample_A_b(sample, local_center, behavioral_init)
            A += single_A
            b += single_b

        # Implement group lasso via randomized coordinate descent via soft thresholding.
        new_weights = np.copy(self.weights[index, :])
        prev_weights = np.copy(new_weights)
        for i in range(self.max_iter):
            # Randomly shuffle the indices of actions and state features
            action_indices = np.random.permutation(self.num_actions)
            state_feature_indices = np.random.permutation(self.state_dim)

            for a_idx in action_indices:
                for s_idx in state_feature_indices:
                    # Select regularization parameter and index of features to regularize
                    if self.basis_type == 'bspline':
                        if s_idx == 0:
                            pos = [int(self.feature_dim / self.num_actions) * i for i in
                                    range(self.num_actions)]
                            reg = self.lambda_reg * math.sqrt(self.df)
                        else:
                            pos = [(int(self.feature_dim / self.num_actions) * a_idx + (
                                        s_idx - 1) * self.df + 1) + k for k in
                                    range(self.df)]
                            reg = self.lambda_reg

                    elif self.basis_type == 'trig':
                        if s_idx == 0:
                            pos = [int(self.feature_dim / self.num_actions) * i for i in
                                    range(self.num_actions)]
                            reg = self.lambda_reg * math.sqrt(2 * self.df)
                        else:
                            pos = [int(self.feature_dim / self.num_actions) * a_idx + (s_idx - 1) * (
                                        2 * self.df) + 1 + k for k in range(2 * self.df)]
                            reg = self.lambda_reg

                    # Compute update
                    new_weights[pos] = new_weights[pos] - self.mu * (
                        np.dot(A[pos, :], new_weights) - b[pos])

                    # Compute soft thresholded weights
                    new_weights[pos] = self.soft_threshold(np.atleast_1d(new_weights[pos]), reg, self.mu)

            if np.linalg.norm(new_weights - prev_weights) < self.epsilon:
                break
            prev_weights = np.copy(new_weights)

        # Perform a debiasing step, perform least squares on the support of the weights (i.e., features that are non-zero)
        if debiasing:
            support = np.nonzero(new_weights)[0]
            A_reduced = A[np.ix_(support, support)]
            b_reduced = b[support]
            new_weights_debiased = np.zeros_like(new_weights)
            new_weights_debiased[support] = np.linalg.lstsq(A_reduced, b_reduced, rcond=None)[0]
            self.weights[index, :] = new_weights_debiased
        else:
            self.weights[index, :] = new_weights

    def policy_iteration(self, samples, debiasing=False):
        """
        Performs policy iteration on the model.

        Args:
            samples (numpy.ndarray): Array of data samples.
            debiasing (bool, optional): Whether to perform debiasing after each policy iteration. Defaults to False.
        """
        for i in range(self.max_policy_iter):
            # Update each local model
            for local_center in self.local_centers:
                print("Fitting model for z = {}, policy iteration step {}".format(local_center, i))
                self.learn(samples, local_center, False, debiasing)
        
    def save(self, path):
        """
        Saves the model weights to a file.

        Args:
            path (str): Path to save the weights to.
        """
        np.save(path, self.global_weights)