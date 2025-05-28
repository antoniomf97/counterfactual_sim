import sys
import uuid
import random

import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load
from tqdm import tqdm

from player import Player
from games import NSH, CRD, SH


class Simulator:
    games = {"NSH": NSH, "CRD": CRD, "SH": SH}

    def __init__(self, simulation, parameters):
        self.population: list[Player] = []
        self.k: int = 0

        self.N = parameters["N"]
        self.Z = simulation["population_size"]
        self.beta: float = simulation["selection_strength"]
        self.mutation: float = simulation["mutation_rate"] / self.Z
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
        return self.k / self.Z

    def sampling_context(self):
        k = 0
        for player in random.sample(self.population, round(self.sample_size)):
            k += player.strategy
        return k / self.sample_size

    def gaussian_context(self):
        k = round(np.random.normal(self.k, self.sigma, 1)[0])
        return k / self.Z

    def calculate_efc(self):
        return sum([self.distribution[k]*k/self.Z for k in range(self.Z + 1)])

    def imitate(self, player_A, player_B, k):
        fitness_A = 0
        for _ in range(round(self.Z/self.N)):
            n_cooperators = sum([p.strategy for p in random.sample(self.population, k=self.N-1)])
            fitness_A += self.game.payoff(player_A.strategy, n_cooperators)

        fitness_B = 0
        for _ in range(round(self.Z/self.N)):
            n_cooperators = sum([p.strategy for p in random.sample(self.population, k=self.N-1)])
            fitness_B += self.game.payoff(player_B.strategy, n_cooperators)
        
        p_Fermi = 1.0 / (1.0 + np.exp(self.beta * (fitness_A - fitness_B)))

        if random.random() <= p_Fermi:
            player_A.strategy = player_B.strategy

    def evolutionary_step(self):
        player_A, player_B = random.sample(self.population, k=2)

        i_strategy = player_A.strategy

        if random.random() < self.mutation:
            player_A.mutate()
        else:
            depth = np.random.choice(
                np.arange(0, len(self.depth_dist)), p=self.depth_dist
            )
            if depth == 0:
                self.imitate(player_A, player_B, self.k)
            else:
                print("other app")

        if player_A.strategy != i_strategy:
            self.k += player_A.strategy - i_strategy

        self.distribution[self.k] += 1

    def run_one_generation(self):
        for _ in range(self.Z):
            self.evolutionary_step()

    def run(self, write_output=True, plot=True):
        for _ in tqdm(range(self.gens)):
            self.run_one_generation()

        self.distribution = self.distribution / (self.Z * self.gens)

        depth = "".join(
            e for e in str(self.depth_dist).replace("0.", "") if e.isalnum()
        )
        filename = f"{self.game_label}n{self.Z}b{self.beta}{self.game}c{self.cost}d{depth}".replace(
            ".", ""
        ).lower()

        if plot:
            self.plot_stationary_distribution()

        if write_output:
            self.write_output(filename)

    def write_output(self, filename):
        path = f"./outputs/{filename}.csv"
        with open(path, "a") as file:
            for i, v in enumerate(self.distribution):
                file.write(f"{i},{v}\n")

    def plot_stationary_distribution(self):
        plt.plot(self.distribution)
        plt.show()
    