from rl.src.dqn.base import ReplayMemoryBase
from rl.src.dqn.components.memory import PrioritizedReplayMemory, ReplayMemory
import wandb


class MemoryManager:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size

        if wandb.run is None:
            return
        wandb.config.memory_size = memory_size

    def initialize(self, memory_type: str = "prioritized") -> ReplayMemoryBase:
        if memory_type == "replay":
            return ReplayMemory(self.memory_size)
        elif memory_type == "prioritized":
            return PrioritizedReplayMemory(self.memory_size)
        raise ValueError(f"Unknown memory type: {memory_type}")

    def update_priorities(self, memory: ReplayMemoryBase, *args) -> None:
        if isinstance(memory, PrioritizedReplayMemory):
            memory.update_priorities(*args)
        else:
            raise ValueError("Memory is not prioritized")
