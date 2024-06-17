import pygame as pg


class Goal:
    def __init__(self, pos_x: int, pos_y: int):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def __eq__(self, other):
        return self.pos_x == other.pos_x and self.pos_y == other.pos_y

    def render(self, screen: pg.Surface, cell_width: float, cell_height: float, color: (int, int, int) = (0, 255, 0)):
        pg.draw.rect(screen, color, (self.pos_x * cell_width,
                                     self.pos_y * cell_height, cell_width, cell_height))
