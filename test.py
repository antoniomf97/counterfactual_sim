import random, sys
from yaml import safe_load
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid, os
import pickle
import multiprocessing as mp
from copy import deepcopy
from scipy.special import binom
from math import floor


class Player:
    id = itertools.count()

    def __init__(self, strategy=None):
        self.id: int = next(Player.id)
        self.strategy: int = strategy if strategy is not None else random.choice([0, 1])
        self.context: list = []
        self.actions: list = []

    def mutate(self):
        self.strategy = 1 - self.strategy

    def get_past_context(self, depth: int):
        min_len = min([floor(len(self.context)/2), depth]) + 1
        return [self.context[-2*i] for i in range(1, min_len)]


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

        self.use_context = simulation["use_context"]
        self.perception = self.set_perception(simulation["perception"])

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

    def set_perception(self, perception):
        if perception["label"] == "default":
            return self.default_perception
        elif perception["label"] == "sample":
            self.sample_size = perception["sample_size"]
            return self.sampling_perception
        elif perception["label"] == "gaussian":
            self.sigma = perception["uncertainty"]
            return self.gaussian_perception
        else:
            raise ValueError("Invalid perception was provided.")

    def default_perception(self):
        return self.k

    def sampling_perception(self):
        k = 0
        for player in random.sample(self.population, round(self.sample_size)):
            k += player.strategy
        return round(k / self.sample_size * self.Z)

    def gaussian_perception(self):
        return round(np.random.normal(self.k, self.sigma, 1)[0])

    def calculate_efc(self):
        return sum([self.distribution[k] * k / self.Z for k in range(self.Z + 1)])

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

        perceived_k = self.perception()

        if random.random() < self.mutation:
            player_A.mutate()
        else:
            depth = np.random.choice(
                np.arange(0, len(self.depth_dist)), p=self.depth_dist
            )

            if depth == 0:
                self.imitate(player_A, player_B)
                
            elif len(player_A.context) >= 2:
                if not self.use_context:
                    past_action = 1 - player_A.strategy
                    fitness_now = self.game.fitness(player_A.strategy, perceived_k)
                    fitness_past = self.game.fitness(past_action, perceived_k)        

                    # learning step
                    p_Fermi = 1.0 / (
                        1 + np.exp(self.beta * (fitness_now - fitness_past))
                    )
                    if random.random() < p_Fermi:
                        player_A.strategy = past_action
                else:
                    past_action = 1 - player_A.strategy
                    # 1. get context on which past actions were taken
                    past_context = player_A.get_past_context(depth)

                    # 2. compute past actions fitnesses
                    fitness_now = self.game.fitness(player_A.strategy, perceived_k)
                    fitness_past = [
                        self.game.fitness(past_action, context)
                        for context in past_context
                    ]

                    # 3. compare the differences between the contexts and weight all fitnesses
                    


                    # 4. chose max value of past fitness
                    max_past_fitness = np.max(fitness_past)

                    # 5. learning step
                    p_Fermi = 1.0 / (
                        1 + np.exp(self.beta * (fitness_now - max_past_fitness))
                    )
                    if random.random() < p_Fermi:
                        player_A.strategy = past_action

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
            pickle.dump(self.distribution + [self.calculate_efc()], handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_stationary_distribution(Z, dist, path, save_fig, plot_show):
    plt.plot(dist)
    plt.xlim(0, Z)
    if save_fig:
        print("Figure saved.")
        plt.savefig(f"{path}.png")
    if plot_show:
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
        configurations.append((deepcopy(simulation), parameters))

    return path, size, configurations


def collapse_results(path, size, runs, save_data, save_fig, plot_show):
    dist = np.zeros(size+1)

    for file in os.listdir(path):
        if not file.endswith(".pickle"):
            continue
         
        with open(f"{path}/{file}", 'rb') as handle:
            dist += pickle.load(handle)

    dist /= runs

    if save_data:
        new_file = f"{path}.pickle" 
        with open(new_file, 'wb') as handle:
            pickle.dump(dist, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved cleaned data to {new_file}")

    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
    os.rmdir(path)

    if plot_show or save_fig:
        plot_stationary_distribution(size, dist[:-1], path, save_fig, plot_show)
    
    return dist[-1]


def run_simulation(args):
    simulation, parameters = args
    sim = Simulator(simulation, parameters)

    sim.run()


def run_simulations(run_args, sim_args, save_data=False, save_fig=False, plot_show=False):
    runs, cores = run_args["runs"], run_args["cores"]

    path, size, configurations = prepare_configurations(runs, sim_args)

    num_cores = mp.cpu_count() - 1 if cores == "all" else cores

    print("=" * 10 + f" Running {runs} simulations in {num_cores} cores " + "=" * 10)

    print("Pooling processes...")
    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap(run_simulation, configurations),
                total=run_args["runs"],
            )
        )

    print("Simulations done. Processing results...")

    return collapse_results(path, size, runs, save_data, save_fig, plot_show)


def e1_depth():
    run_args, sim_args = read_arguments()
    depths = [[1.],[0.9,0.1],[0.9,0,0.1],[0.9,0,0,0.1],
              [0.9,0,0,0,0.1],[0.9,0,0,0,0,0.1],[0.9,0,0,0,0,0,0.1],[0.9,0,0,0,0,0,0,0.1],
              [0.9,0,0,0,0,0,0,0,0.1],[0.9,0,0,0,0,0,0,0,0,0.1],[0.9,0,0,0,0,0,0,0,0,0,0.1]]
    labels = ['default', 'sample', 'gaussian']

    sim_args_all = []

    for label in labels:
        aux = []
        for depth in depths:
            sim_args[0]["depth_distribution"] = depth
            sim_args[0]["perception"]["label"] = label
            aux.append(deepcopy(sim_args))
        sim_args_all.append(aux)
 
    efcs = []
    for i, label in enumerate(labels):
        aux = []
        for j, depth in enumerate(depths):
            print("=" * 8 + f" Simulations with label={label} depth={depth} ")
            aux.append(run_simulations(run_args, sim_args_all[i][j]))
        efcs.append(aux)

    print(efcs)

    with open(f"./outputs/e1_depth/e1_depth_{len(labels)}_{len(depths)}.pickle", 'wb') as handle:
        pickle.dump(efcs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fig, ax = plt.subplots(1,1)
    plt.imshow(efcs, vmin=0, vmax=1)
    plt.colorbar()
    ax.set_xticks(ticks=range(len(depths)))
    ax.set_yticks(ticks=[0,1,2])
    ax.set_yticklabels(labels=['default', 'sample', 'gaussian'])
    plt.show()


if __name__ == "__main__":
    e1_depth()
