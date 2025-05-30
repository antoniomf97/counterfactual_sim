import random
import itertools


class Player:
    id = itertools.count()

    def __init__(self):
        self.id: int = next(Player.id)
        self.strategy: int = random.choice([0, 1])
        self.actions: list = []
        self.context: list = []
        self.fitness: float = 0.

    def mutate(self):
        self.strategy = 1 - self.strategy

    def get_past_actions(self, depth: int):
        return self.actions[-depth:]

    def get_past_context(self, depth: int):
        return self.context[-depth:]

    def __str__(self):
        return f"Player(ID={self.id}; S={self.strategy})"
