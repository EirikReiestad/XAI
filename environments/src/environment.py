from abc import ABC, abstractmethod
import numpy as np
import pygame


class Environment(ABC):
    @abstractmethod
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.screen = None

    @abstractmethod
    def set_screen(self, screen: pygame.Surface):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass
