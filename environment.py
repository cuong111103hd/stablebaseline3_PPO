import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

class STAR(gym.Env):
    def __init__(self,
                 num_antennas,
                 num_star_elements,
                 num_users,
                 num_d2d_pairs,
                 channel_est_error=False,
                 awgn_var=1e-2,
                 channel_noise_var=1e-2):

        self.M = num_antennas
        self.N = num_star_elements
        self.K = num_users
        self.D = num_d2d_pairs

        self.power = 1000
        self.power_d2d = 1
        self.power_d2d_matrix = np.ones(self.D) * np.sqrt(self.power_d2d)

        self.channel_est_error = channel_est_error
        self.awgn_var = awgn_var
        self.channel_noise_var = channel_noise_var

        self.action_dim = 2 * self.M * self.K + 2 * self.N + self.D + 1
        self.state_dim = self.action_dim + 2 * (self.M * self.K + self.M * self.N + self.N * self.K +
                                                2 * self.M * self.D + self.D ** 2 + 2 * self.N * self.D + 2 * self.D * self.K)

        # Action Space
        L1 = np.sqrt(self.power)
        L2 = 1.0
        L3 = np.sqrt(self.power_d2d)
        L4 = self.N

        low = np.concatenate([
            -L1 * np.ones(2 * self.M * self.K),
            -L2 * np.ones(2 * self.N),
            np.zeros(self.D),
            np.zeros(1)
        ])
        high = np.concatenate([
            L1 * np.ones(2 * self.M * self.K),
            L2 * np.ones(2 * self.N),
            L3 * np.ones(self.D),
            np.array([L4])
        ])
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.G = np.random.randn(self.M, self.K) + 1j * np.random.randn(self.M, self.K)
        trace_GGH = np.trace(self.G @ self.G.conj().T)
        scaling_factor = np.sqrt((self.power - 1) / trace_GGH)
        self.G = scaling_factor * self.G
        self.Phi = np.eye(self.N, dtype=complex)

        self.state = None
        self.done = False
        self.episode_t = None

    def stack_matrix(self, matrix):
        return np.hstack((np.real(matrix).ravel(), np.imag(matrix).ravel()))

    def compute_energy(self, matrix):
        return np.sum(np.abs(matrix)**2)

    def compute_reward(self, G, Phi, power_d2d_matrix, number_of_transmit):
        diag_Phi = np.diag(Phi)
        diag_Phi1 = np.zeros((self.N,), dtype=complex)
        diag_Phi2 = np.zeros((self.N,), dtype=complex)
        diag_Phi1[:number_of_transmit] = diag_Phi[:number_of_transmit]
        diag_Phi2[number_of_transmit:] = diag_Phi[number_of_transmit:]
        Phi1 = np.diag(diag_Phi1)
        Phi2 = np.diag(diag_Phi2)

        reward = 0
        opt_reward = 0.1
        min_R_d2d = 100

        G_power = self.compute_energy(G)
        if G_power > self.power:
            return 0, opt_reward

        for k in range(self.K):
            bs_user_k = self.bs_users[:, k]
            star_user_k = self.star_users[:, k]
            G_remove = np.delete(self.G, k, 1)
            d2d_user_k = self.d2d_users[:self.D, k]
            Phi_k = Phi1 if k < self.K // 2 else Phi2

            x = self.compute_energy((bs_user_k.T + star_user_k.conj().T @ Phi_k @ self.bs_star.T) @ self.G[:, k])
            inter_users = self.compute_energy((bs_user_k.T + star_user_k.conj().T @ Phi_k @ self.bs_star.T) @ G_remove)
            inter_d2d = self.compute_energy((d2d_user_k.T + star_user_k.conj().T @ Phi_k @ self.star_d2d[:, :self.D]) @ power_d2d_matrix)
            rho_k = x / (inter_users + inter_d2d)
            reward += np.log2(1 + rho_k)
            opt_reward += np.log2(1 + self.K / 2)

        for j in range(self.D):
            d2d_remove = np.delete(self.d2d_d2d, j, 0)
            d2d_star_remove = np.delete(self.star_d2d[:, :self.D], j, 1)
            power_d2d_matrix_remove = np.delete(power_d2d_matrix, j, 0)
            Phi_j = Phi1 if j < self.D // 2 else Phi2

            x = self.compute_energy((self.d2d_d2d[j, j] + self.star_d2d[:, self.D + j] @ self.Phi @ self.star_d2d[:, j].T) * power_d2d_matrix[j])
            inter_users = self.compute_energy((self.bs_d2d[:, j + self.D].T + self.star_d2d[:, self.D + j].conj().T @ Phi_j @ self.bs_star.T) @ self.G)
            inter_d2d = self.compute_energy((d2d_remove[:, j].T + self.star_d2d[:, self.D + j].conj().T @ Phi_j @ d2d_star_remove) @ power_d2d_matrix_remove)
            rho_j = x / (inter_users + inter_d2d)
            min_R_d2d = min(min_R_d2d, np.log2(1 + rho_j))

        return reward, opt_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_t = 0

        self.bs_users = np.random.randn(self.M, self.K) + 1j * np.random.randn(self.M, self.K)
        self.bs_star = np.random.randn(self.M, self.N) + 1j * np.random.randn(self.M, self.N)
        self.star_users = np.random.randn(self.N, self.K) + 1j * np.random.randn(self.N, self.K)
        self.number_of_transmit = np.random.uniform(0, self.N)

        init_action_G = np.hstack((np.real(self.G).ravel(), np.imag(self.G).ravel()))
        init_action_Phi = np.hstack((np.real(np.diag(self.Phi)), np.imag(np.diag(self.Phi))))
        init_action_power_d2d = self.power_d2d_matrix
        init_action = np.hstack((init_action_G, init_action_Phi, init_action_power_d2d, self.number_of_transmit))

        self.bs_d2d = np.random.randn(self.M, 2 * self.D) + 1j * np.random.randn(self.M, 2 * self.D)
        self.d2d_d2d = np.random.randn(self.D, self.D) + 1j * np.random.randn(self.D, self.D)
        self.star_d2d = np.random.randn(self.N, 2 * self.D) + 1j * np.random.randn(self.N, 2 * self.D)
        self.d2d_users = np.random.randn(2 * self.D, self.K) + 1j * np.random.randn(2 * self.D, self.K)

        self.state = np.hstack((init_action,
                                self.stack_matrix(self.bs_users),
                                self.stack_matrix(self.bs_star),
                                self.stack_matrix(self.star_users),
                                self.stack_matrix(self.bs_d2d),
                                self.stack_matrix(self.d2d_d2d),
                                self.stack_matrix(self.star_d2d),
                                self.stack_matrix(self.d2d_users))).astype(np.float32)

        return self.state, {}

    def step(self, action):
        self.episode_t += 1
        action = action.astype(np.float64)

        number_of_transmit = round(action[-1])
        power_d2d_matrix = np.clip(action[-self.D - 1:-1], 0.0, np.sqrt(self.power_d2d))
        G_real = action[:self.M * self.K]
        G_imag = action[self.M * self.K:2 * self.M * self.K]
        Phi_real = action[-2 * self.N - self.D - 1:-self.N - self.D - 1]
        Phi_imag = action[-self.N - self.D - 1:-self.D - 1]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)
        Phi_complex = Phi_real + 1j * Phi_imag
        Phi_complex /= np.abs(Phi_complex) + 1e-8
        self.Phi = np.diag(Phi_complex)

        reward, opt_reward = self.compute_reward(self.G, self.Phi, power_d2d_matrix, number_of_transmit)

        terminated = False
        truncated = self.episode_t >= 1000

        self.state = np.hstack((action,
                                self.stack_matrix(self.bs_users),
                                self.stack_matrix(self.bs_star),
                                self.stack_matrix(self.star_users),
                                self.stack_matrix(self.bs_d2d),
                                self.stack_matrix(self.d2d_d2d),
                                self.stack_matrix(self.star_d2d),
                                self.stack_matrix(self.d2d_users))).astype(np.float32)

        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass