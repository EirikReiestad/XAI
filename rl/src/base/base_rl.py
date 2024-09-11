from abc import ABC, abstractmethod


class BaseRL(ABC):
    @abstractmethod
    def learn(self, total_timesteps: int):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError
