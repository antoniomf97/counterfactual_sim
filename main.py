import sys
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from yaml import safe_load

from player import Player
from games import NSH


def read_arguments():
    if sys.argv[1]:
        file_name: str = "inputs/" + str(sys.argv[1]) + ".yaml"
    else:
        raise ValueError(
            "No filename provided. Please run as 'python main.py <filename>'"
        )

    # Open and parse the YAML file
    with open(file_name, "r") as f:
        data = safe_load(f)

    try:
        return {
            k: data[k] for k in ("simulation", "population", data["population"]["game"])
        }
    except:
        raise ValueError(f"Game '{data['population']['game']}' not defined.")


class Simulator:
    def __init__(self, simulation, population, **kwargs):
        self.population: list[Player] = []
        self.gens = simulation["generations"]
        self.Z = population["pop_size"]
        self.mutation = population["mutation"]
        self.depth_dist = population["depth_dist"]
        self.beta = population["beta"]
        if population["game"] == "NSH":
            self.game = NSH(**kwargs["NSH"])
        self.k = 0

        if population["context"] == "default":
            self.context = self.default_context
        elif population["context"] == "sampling":
            self.context = self.sampling_context
            self.sample_size = population["context_sample_size"]
        elif population["context"] == "gaussian":
            self.context = self.gaussian_context
            self.sigma = population["context_uncertainty"]
        else:
            raise ValueError("Invalid context provided.")

        self.distribution = {i: [0 for _ in range(0, self.Z + 1)] for i in range(len(self.depth_dist))}

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

    def initialize(self):
        self.population = [
            Player(
                depth=np.random.choice(
                    np.arange(0, len(self.depth_dist)), p=self.depth_dist
                )
            )
            for _ in range(self.Z)
        ]

        for player in self.population:
            self.k += player.strategy

        self.distribution

    def imitate(self, player_A, player_B, k):
        fitness_A = self.game.payoff(player_A.strategy, k)
        fitness_B = self.game.payoff(player_B.strategy, k)
        p_Fermi = (1 + np.exp(self.beta * (fitness_A - fitness_B))) ** (-1)
        if random.random() < p_Fermi:
            player_A.strategy = player_B.strategy

    def evolution_step(self, player_A: Player):
        player_B = random.choice(self.population)
        while player_B.id == player_A.id:
            player_B = random.choice(self.population)

        self.k -= player_A.strategy

        perceived_k = round(self.context() * self.Z)

        if random.random() < self.mutation:
            player_A.mutate()
        elif player_A.depth == 0:
            self.imitate(player_A, player_B, perceived_k)
        else:
            past_actions = player_A.get_past_actions()
            past_context = player_A.get_past_context()

            if len(past_actions) == 0:
                self.imitate(player_A, player_B, perceived_k)
            else:
                fitness_now = self.game.payoff(player_A.strategy, perceived_k)
                fitness_past = [
                    self.game.payoff(1-past_actions[-i], past_context[-i])
                    for i in range(0, len(past_actions))
                ]
                index_max = np.argmax(fitness_past)

                p_Fermi = (1 + np.exp(self.beta * (fitness_now - fitness_past[index_max]))) ** (-1)
                if random.random() < p_Fermi:
                    player_A.strategy = past_actions[-index_max]

        player_A.context.append(perceived_k)
        player_A.actions.append(player_A.strategy)
        self.k += player_A.strategy

        self.distribution[player_A.depth][self.k] += 1

    def run_one_generation(self):
        for player in self.population:
            self.evolution_step(player)

    def run(self):
        self.initialize()

        for _ in tqdm(range(self.gens)):
            self.run_one_generation()

        return self.distribution


def run():
    args = read_arguments()

    distribuion = None

    for _ in range(args["simulation"]["runs"]):
        s = Simulator(**args)
        distribuion = s.run()

    plt.plot(distribuion[0])
    plt.show()


if __name__ == "__main__":
    run()
