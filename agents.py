import numpy as np



class UCBVI:


    def __init__(self, S, A, H, K, R):
        assert isinstance(S, int) and isinstance(A, int) and isinstance(H, int) and isinstance(K, int), "Error in S, A, H or K: they must be all int"
        assert S > 0 and A > 0 and H > 0 and K > 0, "Error in S, A, H or K: they must be all positive"
        assert isinstance(R, np.ndarray), "Error in R: it must be instance of np.ndarray"
        assert R.ndim == 2, "Error in R.ndim"
        assert R.shape == (S, A), "Error R shape"
        self.S = S
        self.A = A
        self.H = H
        self.K = K
        self.R = R
        self.L = np.log(5 * self.S * self.A * self.K * self.H)
        self.Nxay = np.zeros((self.S, self.A, self.S))
        self.Nxa = np.zeros((self.S, self.A))
        self.Phat = np.zeros((self.S, self.A, self.S))
        self.Nxah = np.zeros((self.S, self.A, self.H+1))
        self.emp_cost = 10000 * pow(self.H, 3) * pow(self.S, 2) * self.A * pow(self.L, 2)
        self.Qstar = H * np.ones((self.S, self.A, self.H))
        self.newEpisode()


    def newEpisode(self):
        self.state = None
        self.lastaction = None
        self.Vstar = np.zeros((self.S, self.H+1))
        for h in range(self.H-1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    if self.Nxa[s, a] > 0:
                        aux = 0
                        for s_prime in range(self.S):
                            aux = aux + self.Phat[s, a, s_prime] * min(pow(self.H, 2), self.emp_cost / max(1, np.sum(self.Nxah[s_prime, : , h+1])))
                        bound = (14 / 3) * self.H * self.L / self.Nxa[s, a] + np.sqrt(8 * self.L * pow(np.std(self.Phat[s, a, :] * self.Vstar[:, h+1]), 2) / self.Nxa[s, a]) + np.sqrt(8 * np.sum(aux) / self.Nxa[s, a])
                        self.Qstar[s, a, h] = min(self.Qstar[s, a, h], self.R[s, a] + self.Phat[s, a, :].reshape(1, self.S) @ self.Vstar[:, h+1].reshape(self.S, 1) + bound)
                self.Vstar[s, h] = max(self.Qstar[s, :, h])


    def choose(self, state, stage):
        self.state = state
        self.stage = stage
        self.lastaction = np.argmax(self.Qstar[self.state, :, stage])
        if isinstance(self.lastaction, np.ndarray):
            self.lastaction = np.random.choice(self.lastaction)
        return self.lastaction


    def update(self, newstate, reward): # reward is given, so this input is ignored
        assert isinstance(newstate, int) or isinstance(newstate, np.int64), "Error in update(): newstate must be an integer"
        assert newstate >= 0 and newstate < self.S, "Error in update(): newstate must be s.t.: 0<= state < self.S"
        if self.state is not None and self.lastaction is not None:
            self.Nxay[self.state, self.lastaction, newstate] = self.Nxay[self.state, self.lastaction, newstate] + 1
            self.Nxa[self.state, self.lastaction] = self.Nxa[self.state, self.lastaction] + 1
            self.Phat[self.state, self.lastaction, :] = self.Nxay[self.state, self.lastaction, :] / self.Nxa[self.state, self.lastaction]
            self.Nxah[self.state, self.lastaction, self.stage] = self.Nxah[self.state, self.lastaction, self.stage] + 1
        self.state = newstate