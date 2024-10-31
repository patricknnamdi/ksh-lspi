import numpy as np

class MDP:
    def __init__(self, state_dim, sigma, f_1, f_2):
        self.state_dim = state_dim
        self.sigma = sigma
        self.f_1 = f_1
        self.f_2 = f_2
        self.reset()

    def reset(self, init_state=None):
        if init_state is not None:
            self.state = init_state
        else:
            self.state = np.zeros(self.state_dim)
        return self.state

    def step(self, action, components=False):
        if action not in (0, 1):
            raise ValueError("Invalid action. Action should be either 0 or 1.")

        next_state = self.transition(self.state, action)
        reward = self.compute_reward(self.state, action)
        self.state = next_state

        if components:
            return next_state, reward[0], reward[1]
        else:
            return next_state, reward

    def transition(self, state, action):
        eps = np.random.normal(0, self.sigma, size=self.state_dim)
        new_state = state + (2 * action - 1) * eps
        return new_state

    def compute_reward(self, state, action):
        reward_1 = self.f_1(state, action)
        reward_2 = self.f_2(state, action)
        total_reward = reward_1 + reward_2
        return total_reward, (reward_1, reward_2)
    
    def monte_carlo_q_estimate(self, state, action, gamma, num_episodes, max_steps):
        total_return = 0
        total_rewards_components = np.zeros(2)

        for _ in range(num_episodes):
            self.reset(init_state=state)
            rewards = []
            rewards_components = []

            for _ in range(max_steps):
                _, reward, reward_components = self.step(action, components=True)
                rewards.append(reward)
                rewards_components.append(reward_components)

            g = 0
            g_components = np.zeros(2)
            for i in reversed(range(len(rewards))):
                r = rewards[i]
                r_components = rewards_components[i]
                
                g = r + gamma * g
                g_components = r_components + gamma * g_components

            total_return += g
            total_rewards_components += g_components

        return total_return / num_episodes, total_rewards_components / num_episodes


class CorrelatedStatesMDP:
    def __init__(self, state_dim, sigma, f_1, f_2, b = 0.1, delta=0.1):
        self.state_dim = state_dim
        self.sigma = sigma
        self.f_1 = f_1
        self.f_2 = f_2
        self.b = b
        self.delta=delta
        self.reset()

    def reset(self, init_state=None):
        if init_state is not None:
            self.state = init_state
        else:
            # Generate initial state vector
            self.U = np.random.uniform(0, 1, 1)
            self.X = np.random.uniform(0, 2, self.state_dim)
            self.state = np.random.uniform(0, 1, self.state_dim)
        return self.state

    def step(self, action, components=False):
        # Binary actions only
        if action not in (0, 1):
            raise ValueError("Invalid action. Action should be either 0 or 1.")\
        
        next_state = self.transition(self.state, action)
        reward = self.compute_reward(self.state, action)
        self.state = next_state

        if components:
            return next_state, reward[0], reward[1]
        else:
            return next_state, reward[0]

    def transition(self, state, action):
        # Note that b determines the fixed correlation between state components,
        # when a is 0 then the signs of the state components change, 
        # when a is 1 then the signs of the state components stay the same
        next_state = (self.X + self.b * self.U) / (1 + self.b) + self.delta * state
        if action == 0:
            return np.random.normal(np.negative(next_state), self.sigma)
        else:
            return np.random.normal(next_state, self.sigma)

    def compute_reward(self, state, action):
        reward_1 = self.f_1(state, action)
        reward_2 = self.f_2(state, action)
        total_reward = reward_1 + reward_2
        return total_reward, (reward_1, reward_2)
    
    def monte_carlo_q_estimate(self, state, action, gamma, num_episodes, max_steps):
        total_return = 0
        total_rewards_components = np.zeros(2)

        for _ in range(num_episodes):
            self.reset(init_state=state)
            rewards = []
            rewards_components = []

            for _ in range(max_steps):
                _, reward, reward_components = self.step(action, components=True)
                rewards.append(reward)
                rewards_components.append(reward_components)

            g = 0
            g_components = np.zeros(2)
            for i in reversed(range(len(rewards))):
                r = rewards[i]
                r_components = rewards_components[i]
                
                g = r + gamma * g
                g_components = r_components + gamma * g_components

            total_return += g
            total_rewards_components += g_components

        return total_return / num_episodes, total_rewards_components / num_episodes


class NonlinearTransitionMDP:
    def __init__(self, state_dim, num_actions, sigma, f_1, f_2):
        if state_dim < 2:
            raise ValueError("State dimension must be at least 2 for the given reward function.")
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.sigma = sigma
        self.f_1 = f_1
        self.f_2 = f_2
        self.reset()
        
    def reset(self, init_state=None):
        if init_state is not None:
            self.state = init_state
        else:
            self.state = np.random.uniform(0, 1, size=self.state_dim)  # Uniform initialization
        return self.state

    def step(self, action, components=False):
        if action not in (0, 1):
            raise ValueError("Invalid action. Action should be either 0 or 1.")
        next_state = self.transition(self.state, action)
        reward = self.compute_reward(self.state, action)
        self.state = next_state
        if components:
            return next_state, reward[0], reward[1]
        else:
            return next_state, reward

    def transition(self, state, action):
        x = np.random.uniform(0, 2, size=self.state_dim)  # Uniform random variable
        u = np.random.uniform(0, 1, size=self.state_dim)  # Uniform random variable
        eps = np.random.normal(0, self.sigma, size=self.state_dim)
        new_state = (-1)**action * (np.sin(x + action * u) + 0.1 * state**2) + eps
        return new_state

    def compute_reward(self, state, action):
        reward_1 = self.f_1(state, action)
        reward_2 = self.f_2(state, action)
        total_reward = reward_1 + reward_2
        return total_reward, (reward_1, reward_2)

    def monte_carlo_q_estimate(self, state, action, gamma, num_episodes, max_steps, policy=None):
        total_return = 0
        total_rewards_components = np.zeros(2)

        for _ in range(num_episodes):
            self.reset(init_state=state)
            rewards = []
            rewards_components = []
        
            _, reward, reward_components = self.step(action, components=True)
            rewards.append(reward)
            rewards_components.append(reward_components)

            for _ in range(max_steps - 1):
                if policy:
                    next_action = policy(self.state)
                else:
                    next_action = np.random.randint(self.num_actions) 
                _, reward, reward_components = self.step(next_action, components=True)
                rewards.append(reward)
                rewards_components.append(reward_components)

            g = 0
            g_components = np.zeros(2)
            for i in reversed(range(len(rewards))):
                r = rewards[i]
                r_components = rewards_components[i]
                
                g = r + gamma * g
                g_components = r_components + gamma * g_components

            total_return += g
            total_rewards_components += g_components

        return total_return / num_episodes, total_rewards_components / num_episodes
