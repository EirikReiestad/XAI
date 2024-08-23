"""
We will generate a maze, and store it as a text file for later use.
"""

import logging
import os
import sys

import pygame as pg

from environments.gymnasium.envs.maze.utils import (
    MazeTileType as TileType,
)
from environments.gymnasium.utils import Direction
from tools.maze_generator.utils import MazeDrawMode as DrawMode

logging.basicConfig(level=logging.INFO)

MAZE_DATA_DIR = "environments/gymnasium/data/maze/"

if not os.path.exists(MAZE_DATA_DIR):
    logging.info(f"Current directory: {os.getcwd()}")
    raise FileNotFoundError(
        f"Directory {MAZE_DATA_DIR} does not exist. Create it and try again."
    )


class GenerateMaze:
    def __init__(self, width: int, height: int):
        pg.init()
        pg.font.init()

        self.width = width
        self.height = height
        self.header_size = 100
        self.screen_width = 800
        self.screen_height = 800 + self.header_size
        self.cell_size = self.screen_width // self.width

        self.header = pg.Surface((self.screen_width, self.header_size))
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pg.time.Clock()

        self.action = None
        self.current_square = (0, 0)
        self.draw_mode = DrawMode.NOTHING
        self.placement_mode = DrawMode.NOTHING

        self.maze: list[list[int]] = []
        self.maze_index = 0
        self._load_maze(self.maze_index)

    def generate(self):
        """Let the user generate a maze by drawing obstacles"""

        while True:
            self.screen.fill((255, 255, 255))
            self.render()
            self._handle_input()
            self._move_current_square()
            self._update_maze()
            self.action = None
            pg.display.flip()

    def render(self):
        self._render_maze()
        self._render_header()

    def _render_header(self):
        self.header.fill((255, 255, 255))
        self._draw_text(self.header, f"Current maze: {self.maze_index}", (10, 5))
        self._draw_text(self.header, f"Current maze: {self.draw_mode}", (10, 30))
        self.screen.blit(self.header, (0, 0))

    def _draw_text(self, surface: pg.Surface, text: str, position: tuple[int, int]):
        font = pg.font.Font(None, 36)
        text_surface = font.render(text, True, (0, 0, 0))
        surface.blit(text_surface, position)

    def _load_maze(self, index: int):
        filename = MAZE_DATA_DIR + f"maze-{index}-{self.height}-{self.width}.txt"

        if not os.path.exists(filename):
            logging.info(f"File {filename} does not exist. Generating a new maze.")
            self.maze = [[0 for _ in range(self.width)] for _ in range(self.height)]
            self._save_maze(filename)
        self._read_maze(filename)

    def _read_maze(self, filename: str):
        with open(filename, "r") as f:
            maze = [list(map(int, line.strip())) for line in f]
            if len(maze) != self.height or any(len(row) != self.width for row in maze):
                logging.error(f"No data: Failed to read maze from file {filename}.")
                return
            self.maze = maze

    def _save_maze(self, filename: str):
        with open(filename, "w") as f:
            for row in self.maze:
                f.write("".join(map(str, row)) + "\n")

    def _update_maze(self):
        if self.draw_mode in {DrawMode.OBSTACLE, DrawMode.ERASE}:
            self._update_tile(self.draw_mode)
        if self.placement_mode in {DrawMode.START, DrawMode.END}:
            self._update_placement()

    def _update_tile(self, draw_mode: DrawMode):
        value = (
            TileType.OBSTACLE.value
            if draw_mode == DrawMode.OBSTACLE
            else TileType.EMPTY.value
        )
        self.maze[self.current_square[1]][self.current_square[0]] = value

    def _update_placement(self):
        value = (
            TileType.START.value
            if self.placement_mode == DrawMode.START
            else TileType.END.value
        )
        for y in range(self.height):
            for x in range(self.width):
                if (
                    self.placement_mode == DrawMode.START
                    and self.maze[y][x] == TileType.START.value
                ):
                    self.maze[y][x] = TileType.EMPTY.value
                if (
                    self.placement_mode == DrawMode.END
                    and self.maze[y][x] == TileType.END.value
                ):
                    self.maze[y][x] = TileType.EMPTY.value
        self.maze[self.current_square[1]][self.current_square[0]] = value

    def _handle_input(self):
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self._exit_program()
                if event.key in {pg.K_UP, pg.K_DOWN, pg.K_LEFT, pg.K_RIGHT}:
                    self.action = Direction(pg.K_UP - event.key)
                if event.key == pg.K_SPACE:
                    self.draw_mode += 1
                if event.key == pg.K_s:
                    self.placement_mode = DrawMode.START
                if event.key == pg.K_e:
                    self.placement_mode = DrawMode.END
                if event.key == pg.K_n:
                    self.maze_index += 1
                    self._load_maze(self.maze_index)
                if event.key == pg.K_p:
                    self.maze_index = max(0, self.maze_index - 1)
                    self._load_maze(self.maze_index)

    def _exit_program(self):
        filename = os.path.join(
            MAZE_DATA_DIR, f"maze-{self.maze_index}-{self.height}-{self.width}.txt"
        )
        self._save_maze(filename=filename)
        pg.quit()
        sys.exit()

    def _render_maze(self):
        for y in range(self.height):
            for x in range(self.width):
                color = self._get_color(self.maze[y][x])
                self._draw_cell(x, y, color)
        self._render_mode_highlight()

    def _get_color(self, tile_type: int):
        return {
            0: (255, 255, 255),
            1: (0, 0, 0),
            2: (0, 255, 0),
            3: (255, 0, 0),
        }.get(tile_type, (255, 255, 255))

    def _render_mode_highlight(self):
        if self.draw_mode == DrawMode.NOTHING:
            color = (200, 200, 200)
        elif self.draw_mode == DrawMode.OBSTACLE:
            color = (75, 75, 75)
        elif self.draw_mode == DrawMode.ERASE:
            color = (175, 175, 175)
        else:
            color = (255, 255, 255)

        if self.placement_mode == DrawMode.START:
            color = (0, 200, 0)
        elif self.placement_mode == DrawMode.END:
            color = (200, 0, 0)

        self._draw_cell(self.current_square[0], self.current_square[1], color)

    def _draw_cell(self, x, y, color):
        pg.draw.rect(
            self.screen,
            color,
            (
                x * self.cell_size,
                y * self.cell_size + self.header_size,
                self.cell_size,
                self.cell_size,
            ),
        )

    def _move_current_square(self):
        if self.action is None:
            return

        self.placement_mode = DrawMode.NOTHING

        new_square = (
            self.current_square[0] + self.action.tuple[0],
            self.current_square[1] + self.action.tuple[1],
        )
        if (
            new_square[0] >= 0
            and new_square[0] < self.width
            and new_square[1] >= 0
            and new_square[1] < self.height
        ):
            self.current_square = new_square
