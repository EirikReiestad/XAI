class MemoryManager:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)

    def initialize(self):
        pass
