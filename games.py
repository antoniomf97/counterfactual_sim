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
        self.n_games = round(self.Z / (self.N - 1))

    def payoff(self, strategy, k):
        k += strategy
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
                    for i in range(round(self.N))
                ]
            )
        else:
            s = sum(
                [
                    binom(k, i)
                    * binom(self.Z - k - 1, self.N - i - 1)
                    * self.payoff(strategy, i)
                    for i in range(round(self.N))
                ]
            )
        return fitness * s - cost


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
        self.n_games = self.Z

    def payoff(self, A_strategy, B_strategy):
        payoff_matrix = [[self.P, self.T], [self.S, self.R]]
        return payoff_matrix[A_strategy][B_strategy]
    
    def fitness(self, strategy, k):
        if strategy:
            return ((k - 1) * self.R + (self.Z - k) * self.S) / (self.Z - 1)
        else:
            return (k * self.T + (self.Z - k - 1) * self.P) / (self.Z - 1)
