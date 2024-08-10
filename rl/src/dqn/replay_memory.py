from collections import namedtuple, deque
from torch import Tensor
import random
import logging

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Saves a transition."""
        for arg in args:
            if not isinstance(arg, Tensor):
                raise TypeError(f"Expected type torch.Tensor, but got {type(arg)}")

        if (
            len(self.memory) > 0
            and Transition(*args).state.shape != self.memory[0].state.shape
        ):
            logging.warn(
                f"Expected state to have shape {self.memory[0].state.shape}, but got {Transition(*args).state.shape}"
            )
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
