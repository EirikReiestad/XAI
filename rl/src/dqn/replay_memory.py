from collections import namedtuple, deque
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
