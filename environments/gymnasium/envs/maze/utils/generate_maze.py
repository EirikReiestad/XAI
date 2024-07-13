"""
We will generate a maze, and store it as a text file for later use.
"""
import os
import sys
import logging
import pygame as pg
from ..enums import Direction, MazeDrawMode as DrawMode, MazeTileType as TileType

logging.basicConfig(level=logging.INFO)

MAZE_DATA_DIR = "environments/gymnasium/data/maze/"

if not os.path.exists(MAZE_DATA_DIR):
    logging.info(f"Current directory: {os.getcwd()}")
    raise FileNotFoundError(
        f"Directory {MAZE_DATA_DIR} does not exist. Create it and try again.")


class GenerateMaze():
    def __init__(self, width: int, height: int):
        pg.init()
        pg.font.init()

        self.width = width
        self.height = height

        # Header
        self.header_size = 100
        self.screen_width = 800
        self.screen_height = 800 + self.header_size

        self.header = pg.Surface((self.screen_width, self.header_size))

        self.cell_size = self.screen_width // self.width
        self.screen = pg.display.set_mode(
            (self.screen_width, self.screen_height))
        self.clock = pg.time.Clock()

        self.action = None
        self.current_square = (0, 0)
        self.draw_mode = DrawMode.NOTHING
        self.placement_mode = DrawMode.NOTHING

        self.maze = None
        self.maze_index = 0
        self._get_maze(self.maze_index)

    def generate(self):
        """Let the user generate a maze by drawing obstacles"""

        while True:
            self.screen.fill((255, 255, 255))

            self.render()

            self._get_input()

            self._move_current_square()

            self._update_maze()

            self.action = None

            pg.display.flip()

    def render(self):
        self._render_maze()
        self._render_header()

    def _render_header(self):
        self.header.fill((255, 255, 255))
        font = pg.font.Font(None, 36)
        text = font.render(
            f"Current maze: {self.maze_index}", True, (0, 0, 0))
        self.header.blit(text, (10, 5))

        text = font.render(
            f"Current mode: {self.draw_mode}", True, (0, 0, 0))
        self.header.blit(text, (10, 30))

        self.screen.blit(self.header, (0, 0))

    def _get_maze(self, index: int):
        """
        The maze is stored as a text file. We read it here.
        filename format: mazes/maze-{index}-{height}-{width}.txt

        Params:
            index: int
        """
        filename = MAZE_DATA_DIR + \
            f"maze-{index}-{self.height}-{self.width}.txt"

        if not os.path.exists(filename):
            logging.info(
                f"File {filename} does not exist. Generating a new maze.")
            self.maze = [[0 for _ in range(self.width)]
                         for _ in range(self.height)]

            with open(filename, "w") as f:
                for row in self.maze:
                    f.write("".join(map(str, row)) + "\n")

        with open(filename, "r") as f:
            maze = f.readlines()
            maze = [list(map(int, list(row.strip()))) for row in maze]

            if maze in [None, [], ""]:
                logging.error(
                    f"No data: Failed to read maze from file {filename}.")
                return

            if len(maze) != self.height or len(maze[0]) != self.width:
                logging.error(
                    f"Invalid maze size. Expected {self.height}x{self.width}, got {len(maze)}x{len(maze[0])}")
                return

            self.maze = maze

    def _update_maze(self):
        match self.draw_mode:
            case DrawMode.OBSTACLE:
                self.maze[self.current_square[1]
                          ][self.current_square[0]] = TileType.OBSTACLE.value
            case DrawMode.ERASE:
                self.maze[self.current_square[1]
                          ][self.current_square[0]] = TileType.EMPTY.value

        match self.placement_mode:
            case DrawMode.START:
                for y in range(self.height):
                    for x in range(self.width):
                        if self.maze[y][x] == TileType.START.value:
                            self.maze[y][x] = TileType.EMPTY.value
                self.maze[self.current_square[1]
                          ][self.current_square[0]] = TileType.START.value
            case DrawMode.END:
                for y in range(self.height):
                    for x in range(self.width):
                        if self.maze[y][x] == TileType.END.value:
                            self.maze[y][x] = TileType.EMPTY.value
                self.maze[self.current_square[1]
                          ][self.current_square[0]] = TileType.END.value

    def _save_maze(self):
        filename = MAZE_DATA_DIR + \
            f"maze-{self.maze_index}-{self.height}-{self.width}.txt"
        with open(filename, "w") as f:
            for row in self.maze:
                f.write("".join(map(str, row)) + "\n")

    def _get_input(self):
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self._save_maze()
                    pg.quit()
                    sys.exit()
                    return
                if event.key == pg.K_RETURN:
                    self._save_maze()
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
                    self.draw_mode += 1

                if event.key == pg.K_s:
                    self.placement_mode = DrawMode.START
                if event.key == pg.K_e:
                    self.placement_mode = DrawMode.END

                if event.key == pg.K_n:
                    self.maze_index += 1
                    self._get_maze(self.maze_index)
                if event.key == pg.K_b:
                    self.maze_index = max(0, self.maze_index - 1)
                    self._get_maze(self.maze_index)

    def _render_maze(self):
        for y in range(self.height):
            for x in range(self.width):
                match self.maze[y][x]:
                    case 0:
                        self.draw_cell(x, y, (255, 255, 255))
                    case 1:
                        self.draw_cell(x, y, (0, 0, 0))
                    case 2:
                        self.draw_cell(x, y, (0, 255, 0))
                    case 3:
                        self.draw_cell(x, y, (255, 0, 0))

        match self.draw_mode:
            case DrawMode.NOTHING:
                self.draw_cell(self.current_square[0],
                               self.current_square[1], (200, 200, 200))
            case DrawMode.OBSTACLE:
                self.draw_cell(self.current_square[0],
                               self.current_square[1], (75, 75, 75))
            case DrawMode.ERASE:
                self.draw_cell(self.current_square[0],
                               self.current_square[1], (175, 175, 175))

        match self.placement_mode:
            case DrawMode.START:
                self.draw_cell(self.current_square[0],
                               self.current_square[1], (0, 200, 0))
            case DrawMode.END:
                self.draw_cell(self.current_square[0],
                               self.current_square[1], (200, 0, 0))

    def draw_cell(self, x, y, color):
        pg.draw.rect(self.screen, color, (x * self.cell_size, y *
                     self.cell_size + self.header_size, self.cell_size, self.cell_size))

    def _move_current_square(self):
        if self.action is None:
            return

        self.placement_mode = DrawMode.NOTHING

        new_square = (self.current_square[0] + self.action.to_tuple()
                      [0], self.current_square[1] + self.action.to_tuple()[1])
        if new_square[0] >= 0 and new_square[0] < self.width and new_square[1] >= 0 and new_square[1] < self.height:
            self.current_square = new_square


if __name__ == "__main__":
    maze = GenerateMaze(10, 10)
    maze.generate()
