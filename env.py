import numpy as np

class StochasticFrozenLake:

    def __init__(self, grid_size, no_hole_prob, horizon):
        assert grid_size >= 2, "Grid size must be greater than or equal to 2"
        self.grid_size = grid_size
        self.no_hole_prob = no_hole_prob
        self.horizon = horizon
        self.A = 5  # 0: Stay, 1: Up, 2: Down, 3: Left, 4: Right
        self._compute_rewards()
        self.reset()

    def _compute_rewards(self):
        self.rewards_table = np.zeros((self.grid_size*self.grid_size, self.A))
        self.rewards_table[-1,0] = 1
        self.rewards_table[-2,4] = 1
        self.rewards_table[-self.grid_size-1,2] = 1

    def reset(self):
        self.agent_pos = [0, 0]
        self._generate_lake()
        self._compute_allowed_actions()

    def _generate_lake(self):
        self.lake = np.random.rand(self.grid_size, self.grid_size) < self.no_hole_prob # True means safe
        self.lake[0, 0] = True  # Start is safe
        self.lake[-1, -1] = True  # Goal is safe
        self.lake[self.agent_pos[0], self.agent_pos[1]] = True # Current agent position is safe

    def _compute_allowed_actions(self):
        self.allowed_actions = [0]
        if self.agent_pos[0] > 0:  # Up
            if self.lake[self.agent_pos[0] - 1, self.agent_pos[1]]:
                self.allowed_actions.append(1)
        if self.agent_pos[0] < self.grid_size - 1:  # Down
            if self.lake[self.agent_pos[0] + 1, self.agent_pos[1]]:
                self.allowed_actions.append(2)
        if self.agent_pos[1] > 0:  # Left
            if self.lake[self.agent_pos[0], self.agent_pos[1] - 1]:
                self.allowed_actions.append(3)
        if self.agent_pos[1] < self.grid_size - 1:  # Right
            if self.lake[self.agent_pos[0], self.agent_pos[1] + 1]:
                self.allowed_actions.append(4)

    def getAllowedActions(self):
        return self.allowed_actions

    def getCurrentState(self):
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]
    
    def getRewards(self):
        return self.rewards_table
    
    def computeOptimalValueFunction(self, n_sim=100000):
        if self.no_hole_prob == 1:
            return self.horizon - 2 * (self.grid_size - 1) + 1
        else:
            mc_reward = 0
            for i in range(n_sim):
                self.reset()
                for h in range(self.horizon):
                    if self.agent_pos == [self.grid_size-1, self.grid_size-1]: # in goal
                        action = 0
                    elif self.agent_pos[0] >= self.agent_pos[1]: # on or below diagonal
                        if 4 in self.allowed_actions:
                            action = 4
                        elif 2 in self.allowed_actions:
                            action = 2
                        else:
                            action = 0
                    else: # above diagonal
                        if 2 in self.allowed_actions:
                            action = 2
                        elif 4 in self.allowed_actions:
                            action = 4
                        else:
                            action = 0
                    _, r = self.step(action)
                    mc_reward += r
            return mc_reward/n_sim

    def step(self, action):
        assert isinstance(action, int) or isinstance(action, np.int64), "Error in action: it must be int"
        assert action >= 0 and action < self.A, "Action not consistent"
        reward = 0
        if action in self.allowed_actions:
            if action == 1:  # Up
                self.agent_pos[0] = self.agent_pos[0] - 1
            elif action == 2:  # Down
                self.agent_pos[0] = self.agent_pos[0] + 1
            elif action == 3:  # Left
                self.agent_pos[1] = self.agent_pos[1] - 1
            elif action == 4:  # Right
                self.agent_pos[1] = self.agent_pos[1] + 1
            if self.agent_pos == [self.grid_size - 1, self.grid_size - 1]:
                reward = 1
        self._generate_lake()
        self._compute_allowed_actions()
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1], reward
