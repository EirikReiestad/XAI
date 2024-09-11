from rl.src.dqn.components.memory import PrioritizedReplayMemory


class MemoryManager:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size

    def initialize(self) -> PrioritizedReplayMemory:
        return PrioritizedReplayMemory(self.memory_size)