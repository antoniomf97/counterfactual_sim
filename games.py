from scipy.special import binom
import numpy as np
import matplotlib.pyplot as plt


class NSH:
    """
    N-Player Stag Hunt Game
    """

    def __init__(self, Z, N, M, F, c):
        self.Z: int = Z
        self.N: int = N
        self.M: int = M
        self.F: float = F
        self.c: float = c

    def __str__(self):
        return f"n{self.N}m{self.M}f{self.F}c{self.c}".replace(".", "")

    def payoff(self, strategy, k):
        k += strategy
        # print(strategy, k)
        group_strategy = 0 if k < self.M else 1
        payoff_matrix = [
            [0, k * self.F * self.c / self.N],
            [-self.c, k * self.F * self.c / self.N - self.c],
        ]

        return payoff_matrix[strategy][group_strategy]

    def fitness(self, strategy, k, cost=0):
        fitness = 1.0 / binom(self.Z - 1, self.N - 1)
        if strategy:
            s = sum(
                [
                    binom(k - 1, i)
                    * binom(self.Z - k, self.N - i - 1)
                    * self.payoff(strategy, i + 1)
                    for i in range(self.N)
                ]
            )
        else:
            s = sum(
                [
                    binom(k, i)
                    * binom(self.Z - k - 1, self.N - i - 1)
                    * self.payoff(strategy, i)
                    for i in range(self.N)
                ]
            )
        return fitness * s - cost


class NSG:
    """
    N-Player Snowdrift Game
    """

    def __init__(self, Z, N, M, b, c):
        self.Z: int = Z
        self.N: int = N
        self.M: int = M
        self.b: float = b
        self.c: float = c

    def __str__(self):
        return f"n{self.N}m{self.M}f{self.b}c{self.c}".replace(".", "")

    def payoff(self, strategy, k):
        k += strategy

        if k < self.M:
            if strategy:
                return -self.c / self.M
            else:
                return 0
        else:
            if strategy:
                return self.b - self.c / k
            else:
                return self.b
        # group_strategy = 0 if k < self.M else 1
        # payoff_matrix = [[0, self.b],[-self.c/self.M, self.b-self.c/k]]

        # return payoff_matrix[strategy][group_strategy]


class P2:
    """
    2-Player Game
    """

    def __init__(self, Z, R, S, T, P):
        self.Z: int = Z
        self.N: int = 2
        self.R: float = R
        self.S: float = S
        self.T: float = T
        self.P: float = P

    def __str__(self):
        return f"r{self.R}s{self.S}t{self.T}p{self.P}"

    def payoff(self, A_strategy, B_strategy):
        payoff_matrix = [[self.P, self.T], [self.S, self.R]]
        return payoff_matrix[A_strategy][B_strategy]

    def fitness(self, strategy, k, cost=0):
        if strategy:
            return ((k - 1) * self.R + (self.Z - k) * self.S) / (self.Z - 1) - cost
        else:
            return (k * self.T + (self.Z - k - 1) * self.P) / (self.Z - 1) - cost


# class CRD:
#     def __init__(self, N, M, b, c, r):
#         self.N: int = N
#         self.M: int = M
#         self.b: float = b
#         self.c: float = c
#         self.r: float = r

#     def payoff(self, s, k):
#         if s == 0:
#             if 1 <= k < self.M:
#                 return self.b * (1 - self.r)
#             else:
#                 return self.b
#         else:
#             if 1 <= k < self.M:
#                 return self.b * (1 - self.r - self.c)
#             else:
#                 return self.b * (1 - self.c)


# if __name__ == "__main__":
#     Z = 10
#     N = 6
#     M = 3
#     F = 6.1
#     c = 1.
#     beta = 5.

#     nsh = NSH(Z, N, M, 6.1, 1)

#     t_minus = np.array([k/Z*((Z-k)/(Z-1) * 1./(1.0 + np.exp(beta * (nsh.fitness(1,k) - nsh.fitness(0,k))))) for k in range(0, Z+1)])
#     t_plus = np.array([(Z-k)/Z*(k/(Z-1) * 1./(1.0 + np.exp(beta * (nsh.fitness(0,k) - nsh.fitness(1,k))))) for k in range(0, Z+1)])

#     G = t_plus - t_minus

#     print(G)

#     plt.plot(G)
#     plt.grid(True)
#     plt.plot(np.zeros(Z+1))
#     plt.xlim(0,Z)
#     plt.show()
