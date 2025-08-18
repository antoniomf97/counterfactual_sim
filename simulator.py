import random

import numpy as np
import uuid
import pickle

from player import Player
from games import NSH, P2


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
            pickle.dump(np.append(self.distribution, self.calculate_efc()), handle)
