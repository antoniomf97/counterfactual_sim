import random
import itertools
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