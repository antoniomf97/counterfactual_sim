import random

import numpy as np
from tqdm import tqdm

from player import Player
from games import NSH, CRD, SH


class Simulator:
    games = {"NSH": NSH, "CRD": CRD, "SH": SH}

    def __init__(self, gens, population, context, **kwargs):
        self.population: list[Player] = []
        self.k: int = 0
        self.Z: int = population["pop_size"]
        self.mutation: float = population["mutation"]
        self.beta: float = population["beta"]
        self.game_label = population["game"]
        self.game = self.games[self.game_label](self.Z, **kwargs[self.game_label])
        self.gens: int = gens

        self.use_context = population["use_context"]
        self.set_context(context)

        self.depth_dist: list[int] = population["depth_dist"]
        self.cost: float = population["cost"]

        self.initialize_population()

        self.distribution = np.zeros(self.Z + 1)

    def set_context(self, context):
        if not self.use_context:
            self.context = self.default_context
        elif context["label"] == "default":
            self.context = self.default_context
        elif context["label"] == "sample":
            self.context = self.sampling_context
            self.sample_size = context["sample_size"]
        elif context["label"] == "gaussian":
            self.context = self.gaussian_context
            self.sigma = context["uncertainty"]
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

    def initialize_population(self):
        self.population = [Player() for _ in range(self.Z)]

        for player in self.population:
            self.k += player.strategy

    def imitate(self, player_A, player_B, k):
        fitness_A = self.game.fitness(player_A.strategy, k)
        fitness_B = self.game.fitness(player_B.strategy, k)

        p_Fermi = 1.0 / (1.0 + np.exp(self.beta * (fitness_A - fitness_B)))

        if random.random() <= p_Fermi:
            player_A.strategy = player_B.strategy

    def evolution_step(self):
        player_A, player_B = random.sample(self.population, k=2)

        i_strategy = player_A.strategy

        perceived_k = self.context() * self.Z

        if random.random() < self.mutation:
            player_A.mutate()
        else:
            # apply the depth distribution statistically
            depth = np.random.choice(
                np.arange(0, len(self.depth_dist)), p=self.depth_dist
            )
            if depth == 0:
                self.imitate(player_A, player_B, perceived_k)
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
                    past_context[i] if self.use_context else perceived_k
                    for i in other_strat_idx
                ]

                # 2. if there are no past actions with different strategy, uses immitation
                if len(past_actions) == 0:
                    self.imitate(player_A, player_B, perceived_k)
                else:
                    fitness_now = self.game.fitness(player_A.strategy, perceived_k, 0)
                    fitness_past = [
                        self.game.fitness(
                            past_actions[-i], past_context[-i], self.cost * (i + 1)
                        )
                        for i in range(0, len(other_strat_idx))
                    ]

                    # 3. compare the differences between the contexts and weight all fitnesses
                    weight_decay = 1 / self.Z
                    weight_factor = np.array(
                        [
                            np.exp(
                                -weight_decay * np.abs(perceived_k - past_context[-i])
                            )
                            for i in range(0, len(other_strat_idx))
                        ]
                    )
                    fitness_past *= weight_factor

                    # 4. chose max value of past fitness
                    index_max = np.argmax(fitness_past)
                    p_Fermi = 1.0 / (
                        1 + np.exp(self.beta * (fitness_now - fitness_past[index_max]))
                    )
                    if random.random() < p_Fermi:
                        player_A.strategy = past_actions[-index_max]

        if player_A.strategy != i_strategy:
            self.k += player_A.strategy - i_strategy
            player_A.context.append(perceived_k)
            player_A.actions.append(player_A.strategy)

        self.distribution[self.k] += 1

    def run_one_generation(self):
        for _ in self.population:
            self.evolution_step()

    def run(self):

        for _ in tqdm(range(self.gens)):
            self.run_one_generation()

        self.distribution = self.distribution / (self.Z * self.gens)

        depth = "".join(
            e for e in str(self.depth_dist).replace("0.", "") if e.isalnum()
        )

        filename = f"{self.game_label}n{self.Z}b{self.beta}{self.game}c{self.cost}d{depth}".replace(
            ".", ""
        ).lower()

        return self.distribution, filename
