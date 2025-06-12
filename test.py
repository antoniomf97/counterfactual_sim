import random, sys
from yaml import safe_load
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid, os
import pickle
import multiprocessing as mp
from copy import copy


class Player:
    id = itertools.count()

    def __init__(self, strategy=None):
        self.id: int = next(Player.id)
        self.strategy: int = strategy if strategy is not None else random.choice([0, 1])
        self.actions: list = []
        self.context: list = []

    def mutate(self):
        self.strategy = 1 - self.strategy

    def get_past_actions(self, depth: int):
        return self.actions[-depth:]

    def get_past_context(self, depth: int):
        return self.context[-depth:]


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
    games = {"NSH": NSH, "P2": P2}

    def __init__(self, simulation, parameters):
        self.population: list[Player] = []
        self.k: int = 0

        self.Z = simulation["population_size"]
        self.game_label = simulation["game"]
        self.game = self.games[self.game_label](self.Z, **parameters)

        self.N = self.game.N
        self.beta: float = simulation["selection_strength"]
        self.mutation: float = simulation["mutation_rate"]

        self.depth_dist: list[int] = simulation["depth_distribution"]

        self.gens: int = simulation["generations"]
        self.output: str = simulation["output"]

        self.context = self.set_context(
            simulation["use_context"], simulation["context"]
        )

        self.cost = 0

        self.initialize_population(simulation["init_k"])
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

    def set_context(self, use_context, context):
        if not use_context:
            return None
        elif context["label"] == "default":
            return self.default_context
        elif context["label"] == "sample":
            self.sample_size = context["sample_size"]
            return self.sampling_context
        elif context["label"] == "gaussian":
            self.sigma = context["uncertainty"]
            return self.gaussian_context
        else:
            raise ValueError("Invalid context provided.")

    def default_context(self):
        return self.k

    def sampling_context(self):
        k = 0
        for player in random.sample(self.population, round(self.sample_size)):
            k += player.strategy
        return round(k / self.sample_size * self.Z)

    def gaussian_context(self):
        return round(np.random.normal(self.k, self.sigma, 1)[0])

    def imitate(self, player_A, player_B):
        # player A plays n_games
        player_A.fitness = 0
        for _ in range(self.game.n_games):
            n_cooperators = sum(
                [
                    p.strategy
                    for p in random.sample(self.population, k=round(self.N) - 1)
                ]
            )
            player_A.fitness += self.game.payoff(player_A.strategy, n_cooperators)

        # player B plays n_games
        player_B.fitness = 0
        for _ in range(self.game.n_games):
            n_cooperators = sum(
                [
                    p.strategy
                    for p in random.sample(self.population, k=round(self.N) - 1)
                ]
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

        perceived_k = self.k

        if random.random() < self.mutation:
            player_A.mutate()
        else:
            depth = np.random.choice(
                np.arange(0, len(self.depth_dist)), p=self.depth_dist
            ) * 2
            if depth == 0:
                self.imitate(player_A, player_B)
            else:
                past_actions = player_A.get_past_actions(depth)
                past_context = player_A.get_past_context(depth)

                # 1. only compare with strategies that are different
                other_strat_idx = [
                    i
                    for i, action in enumerate(past_actions)
                    if action != player_A.strategy
                ]
                past_actions = [past_actions[i] for i in other_strat_idx]
                past_context = [
                    perceived_k if self.context is None else past_context[i]
                    for i in other_strat_idx
                ]

                # 2. compute past actions fitnesses
                fitness_now = self.game.fitness(
                    player_A.strategy, perceived_k, self.cost
                )
                fitness_past = [
                    self.game.fitness(
                        past_actions[-i], past_context[-i], self.cost * (i + 1)
                    )
                    for i in range(0, len(other_strat_idx))
                ]

                # 3. compare the differences between the contexts and weight all fitnesses

                # 4. chose max value of past fitness
                index_max = np.argmax(fitness_past)
                p_Fermi = 1.0 / (
                    1 + np.exp(self.beta * (fitness_now - fitness_past[index_max]))
                )
                if random.random() < p_Fermi:
                    player_A.strategy = past_actions[-index_max]

        self.k += player_A.strategy - i_strategy
        if player_A.strategy != i_strategy:
            player_A.context.append(perceived_k)
            player_A.actions.append(player_A.strategy)

        self.distribution[self.k] += 1

    def run_one_generation(self):
        for _ in range(self.Z):
            self.evolutionary_step()

    def run(self):
        for _ in range(self.gens):
            self.run_one_generation()

        self.distribution = self.distribution / (self.Z * self.gens)

        self.write_outputs()
    
    def write_outputs(self):
        filename = self.output + f"/{uuid.uuid1()}.pickle"
        with open(filename, 'wb') as handle:
            pickle.dump(self.distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_stationary_distribution(Z, dist, path, save_fig):
    plt.plot(dist)
    plt.xlim(0, Z)
    if save_fig:
        print("Figure saved.")
        plt.savefig(f"{path}.png")
    plt.show()


def prepare_configurations(runs, sim_args):
    simulation, parameters = sim_args
    size = simulation["population_size"]
    path = f"./outputs/{uuid.uuid1()}"
    os.mkdir(path)
    simulation["output"] = path

    k_values = np.linspace(0, size, runs, dtype=int)

    configurations = []
    for i in range(runs):
        simulation["init_k"] = k_values[i]
        configurations.append((copy(simulation), parameters))

    return path, size, configurations


def collapse_results(path, size, runs, save_fig):
    dist = np.zeros(size+1)

    for file in os.listdir(path):
        if not file.endswith(".pickle"):
            continue
         
        with open(f"{path}/{file}", 'rb') as handle:
            dist += pickle.load(handle)

    dist /= runs

    new_file = f"{path}.pickle" 
    with open(new_file, 'wb') as handle:
        pickle.dump(dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
    os.rmdir(path)
    
    print(f"Saved cleaned data to {new_file}")

    plot_stationary_distribution(size, dist, path, save_fig)


def run_simulation(args):
    simulation, parameters = args
    sim = Simulator(simulation, parameters)

    sim.run()


def run_simulations(run_args, sim_args, save_fig=False):
    runs, cores = run_args["runs"], run_args["cores"]

    path, size, configurations = prepare_configurations(runs, sim_args)

    num_cores = mp.cpu_count() - 1 if cores == "all" else cores

    print("=" * 10 + f" Running {runs} simulations in {cores} cores " + "=" * 10)

    print("Pooling processes...")
    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap(run_simulation, configurations),
                total=run_args["runs"],
            )
        )

    print("Simulations done. Processing results...")

    collapse_results(path, size, runs, save_fig)


if __name__ == "__main__":
    run_args, sim_args = read_arguments()
    run_simulations(run_args, sim_args,save_fig=True)