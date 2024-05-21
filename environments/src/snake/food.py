import pygame
import random


class Food:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.color = (255, 0, 0)
        self.size = 10

    def render(self, screen: pygame.Surface, cell_width: float, cell_height: float):
        pygame.draw.rect(screen, self.color, (self.x * cell_width,
                                              self.y * cell_height, cell_width, cell_height))

    def randomize_position(self, width: int, height: int):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
