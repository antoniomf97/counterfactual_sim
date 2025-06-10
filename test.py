import random, sys
from yaml import safe_load
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


class Player:
    id = itertools.count()

    def __init__(self, strategy = None):
        self.id: int = next(Player.id)
        self.strategy: int = strategy if strategy is not None else random.choice([0, 1])

    def mutate(self):
        self.strategy = 1 - self.strategy


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


def read_arguments():
    try:
        file_name: str = "inputs/" + str(sys.argv[1]) + ".yaml"
    except:
        raise ValueError(
            "No filename provided. Please run as 'python main.py <filename>'"
        )

    # Open and parse the YAML file
    with open(file_name, "r") as f:
        data = safe_load(f)

    try:
        return data["running"], (data["simulation"], data[data["simulation"]["game"]])
    except:
        raise ValueError(f"Game '{data['population']['game']}' not defined.")


class Simulator:
    def __init__(self, simulation, parameters):
        self.population: list[Player] = []
        self.k: int = 0

        self.Z = simulation["population_size"]
        self.game = P2(self.Z, **parameters)
        self.beta: float = simulation["selection_strength"]
        self.mutation: float = simulation["mutation_rate"] / self.game.n_games
        self.N = self.game.N

        self.gens: int = simulation["generations"]

        self.initialize_population()
        self.distribution = np.zeros(self.Z + 1)

    def initialize_population(self, init_k=None):
        if init_k is not None:
            strats = np.zeros(self.Z, dtype=int)
            indices = np.random.choice(self.Z, init_k, replace=False)
            strats[indices] = 1
            self.population = [Player(strategy=strats[i]) for i in range(self.Z)]
        else:
            self.population = [Player() for _ in range(self.Z)]

        self.k = 0
        for player in self.population:
            self.k += player.strategy

    def reset(self, init_k=None):
        self.initialize_population(init_k)
        self.distribution = np.zeros(self.Z + 1)

    def imitate(self, player_A, player_B):
        # player A plays n_games
        player_A.fitness = 0
        for _ in range(self.game.n_games):
            n_cooperators = sum(
                [p.strategy for p in random.sample(self.population, k=round(self.N) - 1)]
            )
            player_A.fitness += self.game.payoff(player_A.strategy, n_cooperators)

        # player B plays n_games
        player_B.fitness = 0
        for _ in range(self.game.n_games):
            n_cooperators = sum(
                [p.strategy for p in random.sample(self.population, k=round(self.N) - 1)]
            )
            player_B.fitness += self.game.payoff(player_B.strategy, n_cooperators)

        # normalize
        player_A.fitness /= self.game.n_games
        player_B.fitness /= self.game.n_games

        # leaning step
        p_Fermi = 1.0 / (
            1.0 + np.exp(self.beta * (player_A.fitness - player_B.fitness))
        )

        if random.random() < p_Fermi:
            player_A.strategy = player_B.strategy

    def evolutionary_step(self):
        player_A, player_B = random.sample(self.population, k=2)

        i_strategy = player_A.strategy

        if random.random() < self.mutation:
            player_A.mutate()
        else:
            self.imitate(player_A, player_B)

        self.k += player_A.strategy - i_strategy

        self.distribution[self.k] += 1

    def run_one_generation(self):
        for _ in range(self.Z):
            self.evolutionary_step()

    def run(self):
        for _ in tqdm(range(self.gens)):
            self.run_one_generation()

        self.distribution = self.distribution / (self.Z * self.gens)

        return self.distribution

    def run_n_times(self, runs):
        dist = np.zeros(self.Z + 1)
        k_values = np.linspace(0, self.Z, runs, dtype=int)
        print(k_values)
        for i in range(runs):
            self.reset(init_k=k_values[i])
            dist += self.run()
        
        dist /= runs

        self.plot_stationary_distribution(dist)

    def plot_stationary_distribution(self, dist):
        plt.plot(dist)
        plt.xlim(0, self.Z)
        plt.show()

def run_simulation(run_args, sim_args):
    simulation, parameters = sim_args
    sim = Simulator(simulation, parameters)
    sim.run_n_times(runs=run_args["runs"])


if __name__ == "__main__":
    run_args, sim_args = read_arguments()
    run_simulation(run_args, sim_args)
