"""
We will generate a maze, and store it as a text file for later use.
"""
import pygame as pg
import enum


@enum.unique
class Direction(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def to_tuple(self):
        if self == Direction.UP:
            return (0, -1)
        if self == Direction.RIGHT:
            return (1, 0)
        if self == Direction.DOWN:
            return (0, 1)
        if self == Direction.LEFT:
            return (-1, 0)


class GenerateMaze():
    def __init__(self, width: int, height: int, name: str = "maze.txt"):
        self.width = width
        self.height = height

        self.screen_width = 800
        self.screen_height = 800
        self.cell_size = self.screen_width // self.width
        self.screen = pg.display.set_mode(
            (self.screen_width, self.screen_height))
        self.clock = pg.time.Clock()

        self.action = None
        self.current_square = (0, 0)
        self.obstacle_drawing = False

    def generate(self):
        maze = [[0 for _ in range(self.width)] for _ in range(self.height)]

        while True:
            self.screen.fill((255, 255, 255))

            self._draw_maze(maze)

            self._get_input()

            self._move_current_square()

            self.action = None

            if self.obstacle_drawing:
                maze[self.current_square[1]][self.current_square[0]] = 1
            else:
                maze[self.current_square[1]][self.current_square[0]] = 0

            pg.display.flip()

    def _save_maze(self, maze, name):
        with open(name, "w") as f:
            for row in maze:
                f.write("".join(map(str, row)) + "\n")

    def _get_input(self):
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self._save_maze(maze, "maze.txt")
                    pg.quit()
                    return
                if event.key == pg.K_RETURN:
                    self._save_maze(maze, "maze.txt")
                    return

                if event.key == pg.K_UP:
                    self.action = Direction.UP
                if event.key == pg.K_DOWN:
                    self.action = Direction.DOWN
                if event.key == pg.K_LEFT:
                    self.action = Direction.LEFT
                if event.key == pg.K_RIGHT:
                    self.action = Direction.RIGHT

                if event.key == pg.K_SPACE:
                    self.obstacle_drawing = not self.obstacle_drawing

    def _draw_maze(self, maze):
        for y in range(self.height):
            for x in range(self.width):
                if maze[y][x] == 1:
                    self.draw_cell(x, y, (0, 0, 0))
                else:
                    self.draw_cell(x, y, (255, 255, 255))
        self.draw_cell(self.current_square[0],
                       self.current_square[1], (50, 50, 50))
        if self.obstacle_drawing:
            self.draw_cell(
                self.current_square[0], self.current_square[1], (0, 0, 0))

    def draw_cell(self, x, y, color):
        pg.draw.rect(self.screen, color, (x * self.cell_size, y *
                     self.cell_size, self.cell_size, self.cell_size))

    def _move_current_square(self):
        if self.action is None:
            return

        new_square = (self.current_square[0] + self.action.to_tuple()
                      [0], self.current_square[1] + self.action.to_tuple()[1])
        if new_square[0] >= 0 and new_square[0] < self.width and new_square[1] >= 0 and new_square[1] < self.height:
            self.current_square = new_square


if __name__ == "__main__":
    maze = GenerateMaze(10, 10)
    maze.generate()
