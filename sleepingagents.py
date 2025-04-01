import numpy as np


def argmax_subset(values, subset):
    idx_max = subset[0]
    for elem in subset:
        if values[elem] > values[idx_max]: 
            idx_max = elem
    return idx_max


def int_to_list(n):
    return [i for i in range(n.bit_length()) if n & (1 << i)]


def list_to_int(lst):
    val = 0
    for elem in lst:
        val |= (1 << elem)
    return val


def list_to_listbool(lst, N):
    vect = [False] * N
    for num in lst:
        vect[num] = True
    return np.array(vect)


class SleepingUCBVI:

    
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
        self.L = np.log(80 * pow(self.S, 2) * self.A * pow(self.K, 2) * pow(self.H, 3))
        self.Nxay = np.zeros((self.S, self.A, self.S))
        self.Nxa = np.zeros((self.S, self.A))
        self.Phat = np.zeros((self.S, self.A, self.S))
        self.Nxah = np.zeros((self.S, self.A, self.H+1))
        self.Nx = np.zeros(self.S)
        self.NxB = np.zeros((self.S, pow(2, self.A)))
        self.Chat = np.zeros((self.S, pow(2, self.A)))
        self.emp_cost_Q = pow(2900, 2) * pow(self.H, 3) * pow(self.S, 3) * self.A * pow(2, self.A) * pow(self.L, 3)
        self.emp_cost_V = pow(1350, 2) * pow(self.H, 3) * pow(self.S, 3) * self.A * pow(2, self.A) * pow(self.L, 3)
        self.Qstar = H * np.ones((self.S, self.A, self.H))
        self.action_power_set = [int_to_list(i) for i in range(pow(2, self.A))]
        self.action_power_set_bools = [list_to_listbool(self.action_power_set[i], self.A) for i in range(pow(2, self.A))]
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
                            aux += self.Phat[s, a, s_prime] * min(pow(self.H, 2), self.emp_cost_Q / max(1, np.sum(self.Nxah[s_prime, : , h+1])))
                        bonusQ = 7 / 3 * self.H * self.L / self.Nxa[s, a] + np.sqrt(4 * self.L * pow(np.std(self.Phat[s, a, :] * self.Vstar[:, h+1]), 2) / self.Nxa[s, a]) + np.sqrt(4 * np.sum(aux) / self.Nxa[s, a])
                        self.Qstar[s, a, h] = min(self.Qstar[s, a, h], self.R[s, a] + self.Phat[s, a, :].reshape(1, self.S) @ self.Vstar[:, h+1].reshape(self.S, 1) + bonusQ)
                max_vect = np.zeros(pow(2, self.A))
                over_b = np.zeros(pow(2, self.A))
                for i in range(1, pow(2, self.A)):
                    max_vect[i] = max(self.Qstar[s, self.action_power_set_bools[i], h]) 
                    over_b[i] = min(pow(self.H, 2), self.emp_cost_V / max(1, self.Nxa[s,
                        argmax_subset(self.Qstar[s, :, h], self.action_power_set[i])]))
                aux = self.Chat[s, 1:] * max_vect[1:]
                self.Vstar[s, h] = np.sum(aux)
                term1 = np.sqrt(4 * self.L * pow(np.std(aux), 2) / max(1, self.Nx[s]))
                term2 = 7 * self.H * self.L / max(1, self.Nx[s])
                term3 = np.sqrt(4 * np.sum(self.Chat[s, 1:] * over_b[1:]) / max(1, self.Nx[s]))
                bonusV = term1 + term2 + term3
                self.Vstar[s, h] = min(self.H, self.Vstar[s, h] + bonusV)

    
    def choose(self, state, stage, availableactionlist):
        self.state = state
        self.stage = stage
        lst_id = list_to_int(availableactionlist)
        self.Nx[state] = self.Nx[state] + 1
        self.NxB[state, lst_id] = self.NxB[state, lst_id] + 1
        self.Chat[state, :] = self.NxB[state, :] / np.maximum(self.Nx[state], 1)
        self.lastaction = argmax_subset(self.Qstar[self.state, :, stage], availableactionlist)
        assert self.lastaction in availableactionlist, "Error in action selection"
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