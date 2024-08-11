import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """A memory buffer to store and sample transitions."""

    def __init__(self, capacity: int) -> None:
        """Initialize ReplayMemory with a fixed capacity.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args) -> None:
        """Store a transition in memory.

        Args:
            state (Tensor): The current state.
            action (Tensor): The action taken.
            next_state (Tensor): The resulting next state.
            reward (Tensor): The reward received.

        Raises:
            TypeError: If any argument is not of type torch.Tensor.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        """Randomly sample a batch of transitions from memory.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            list[Transition]: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Return the current size of memory."""
        return len(self.memory)
