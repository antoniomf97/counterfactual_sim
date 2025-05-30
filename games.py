from scipy.special import binom


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
        if strategy == 1:
            if 0 <= k < self.M:
                return -self.c
            else:
                return k * self.F * self.c / self.N - self.c
        else:
            if 0 <= k < self.M:
                return 0
            else:
                return k * self.F * self.c / self.N

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


# class SH:
#     """
#     2-Player Stag Hunt Game
#     """

#     def __init__(self, Z, R, S, T, P):
#         self.Z: int = Z
#         self.R: float = R
#         self.S: float = S
#         self.T: float = T
#         self.P: float = P

#     def __str__(self):
#         return f"r{self.R}s{self.S}t{self.T}p{self.P}"

#     def fitness(self, strategy, k, cost=0):
#         if strategy:
#             return ((k - 1) * self.R + (self.Z - k) * self.S) / (self.Z - 1) - cost
#         else:
#             return (k * self.T + (self.Z - k - 1) * self.P) / (self.Z - 1) - cost


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
