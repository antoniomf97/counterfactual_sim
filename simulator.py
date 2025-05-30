import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from player import Player
from games import NSH


class Simulator:
    games = {"NSH": NSH}

    def __init__(self, simulation, parameters):
        self.population: list[Player] = []
        self.k: int = 0

        self.N = parameters["N"]
        self.Z = simulation["population_size"]
        self.beta: float = simulation["selection_strength"]
        self.mutation: float = simulation["mutation_rate"]
        self.depth_dist: list[int] = simulation["depth_distribution"]
        self.cost: float = simulation["cost"]

        self.game_label = simulation["game"]
        self.game = self.games[self.game_label](self.Z, **parameters)
        self.gens: int = simulation["generations"]

        self.context = self.set_context(simulation["use_context"], simulation["context"])

        self.initialize_population()
        self.distribution = np.zeros(self.Z + 1)

    def initialize_population(self):
        self.population = [Player() for _ in range(self.Z)]

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
        return k / self.sample_size * self.Z

    def gaussian_context(self):
        k = round(np.random.normal(self.k, self.sigma, 1)[0])
        return k

    def calculate_efc(self):
        return sum([self.distribution[k]*k/self.Z for k in range(self.Z + 1)])

    def imitate(self, player_A, player_B):
        """
        Imitation Process

        Args:
            player_A (Player): player A
            player_B (Player): player B
        """
        n_games = round(self.Z/self.N)

        # player A plays n_games
        player_A.fitness = 0
        for _ in range(n_games):
            n_cooperators = sum([p.strategy for p in random.sample(self.population, k=self.N-1)])
            n_cooperators += player_A.strategy
            player_A.fitness += self.game.payoff(player_A.strategy, n_cooperators)

        # player B plays n_games
        player_B.fitness = 0
        for _ in range(n_games):
            n_cooperators = sum([p.strategy for p in random.sample(self.population, k=self.N-1)])
            n_cooperators += player_B.strategy
            player_B.fitness += self.game.payoff(player_B.strategy, n_cooperators)
        
        # normalize
        player_A.fitness /= n_games
        player_B.fitness /= n_games

        # leaning step
        p_Fermi = 1.0 / (1.0 + np.exp(self.beta * (player_A.fitness - player_B.fitness)))

        if random.random() < p_Fermi:
            player_A.strategy = player_B.strategy

    def evolutionary_step(self):
        player_A, player_B = random.sample(self.population, k=2)

        i_strategy = player_A.strategy

        perceived_k = self.context() if self.context else self.k

        if random.random() < self.mutation:
            player_A.mutate()
        else:
            depth = np.random.choice(
                np.arange(0, len(self.depth_dist)), p=self.depth_dist
            )
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
                    past_context[i] if self.context else perceived_k
                    for i in other_strat_idx
                ]

                # 2. if there are no past actions with different strategy, uses immitation
                if len(past_actions) == 0:
                    self.imitate(player_A, player_B)
                else:
                    fitness_now = self.game.fitness(player_A.strategy, perceived_k, self.cost)
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

        # if player_A.strategy != i_strategy:
        self.k += player_A.strategy - i_strategy
        if player_A.strategy != i_strategy:
            player_A.context.append(perceived_k)
            player_A.actions.append(player_A.strategy)

        self.distribution[self.k] += 1

    def run_one_generation(self):
        for _ in range(self.Z):
            self.evolutionary_step()

    def run(self, write_output=True, plot=True):
        for _ in tqdm(range(self.gens)):
            self.run_one_generation()

        self.distribution = self.distribution / (self.Z * self.gens)

        # write filename
        depth = "".join(
            e for e in str(self.depth_dist).replace("0.", "") if e.isalnum()
        )
        filename = f"{self.game_label}n{self.Z}b{self.beta}{self.game}c{self.cost}d{depth}".replace(
            ".", ""
        ).lower()

        # plot or write output
        if plot:
            self.plot_stationary_distribution()
        elif write_output:
            self.write_output(filename)

    def write_output(self, filename):
        path = f"./outputs/{filename}.csv"
        with open(path, "a") as file:
            for i, v in enumerate(self.distribution):
                file.write(f"{i},{v}\n")

    def plot_stationary_distribution(self):
        plt.plot(self.distribution)
        plt.show()
    