class NSH:
    def __init__(self, N, M, F, c):
        self.N: int = N
        self.M: int = M
        self.F: float = F
        self.c: float = c

    def payoff(self, s, k):
        if s == 0:
            if 1 <= k < self.M:
                return 0
            else:
                return k * self.F * self.c / self.N
        else:
            if 1 <= k < self.M:
                return -self.c
            else:
                return k * self.F * self.c / self.N - self.c
