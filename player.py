import random
import itertools


class Player:
    id = itertools.count()

    def __init__(self, depth: int):
        self.id: int = next(Player.id)
        self.strategy: int = random.choice([0, 1])
        self.actions: list = []
        self.context: list = []
        self.depth: int = depth

    def mutate(self):
        self.strategy = random.choice([0, 1])

    def get_past_actions(self):
        return self.actions[-self.depth :]

    def get_past_context(self):
        return self.context[-self.depth :]

    def __str__(self):
        return f"Player(ID={self.id}; S={self.strategy}; D={self.depth})"
