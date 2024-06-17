import pygame as pg
from ..direction import Direction


class Agent:
    def __init__(self, x: int, y: int):
        self.pos_x = int(x)
        self.pos_y = int(y)

    def action(self):
        pass

    def move(self, action: Direction):
        match action:
            case Direction.UP:
                self.pos_y -= 1
            case Direction.DOWN:
                self.pos_y += 1
            case Direction.LEFT:
                self.pos_x -= 1
            case Direction.RIGHT:
                self.pos_x += 1

    def reset(self):
        pass

    def render(self, screen: pg.Surface, cell_width: float, cell_height: float, color: (int, int, int) = (255, 0, 0)):
        pg.draw.rect(screen, color, (self.pos_x * cell_width,
                                     self.pos_y * cell_height, cell_width, cell_height))
