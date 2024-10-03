"""
We will generate an env, and store it as a text file for later use.
"""

import logging
import os
import sys

import pygame as pg

from environments.gymnasium.envs.tag.utils import TileType
from utils.src.tag_generator.utils import DrawMode

logging.basicConfig(level=logging.INFO)

DATA_DIR = "gui/src/assets/"

if not os.path.exists(DATA_DIR):
    logging.info(f"Current directory: {os.getcwd()}")
    raise FileNotFoundError(
        f"Directory {DATA_DIR} does not exist. Create it and try again."
    )


class EnvHandler:
    def __init__(self, width: int, height: int):
        pg.init()
        pg.font.init()

        self.width = width
        self.height = height
        self.screen_width = 800
        self.screen_height = 800
        self.cell_size = self.screen_width // self.width

        self.surface = pg.Surface((self.screen_width, self.screen_height))
        self.clock = pg.time.Clock()

        self.action = None
        self._current_square = (0, 0)
        self.draw_mode = DrawMode.NOTHING
        self.placement_mode = DrawMode.NOTHING

        self.env: list[list[int]] = []
        self.env_index = 0
        self._load_env()

    def generate(self):
        """Let the user generate an env by drawing obstacles"""
        self.surface.fill((255, 255, 255))
        self._update_env()
        self._save_env("gui/src/assets/env.txt")
        self.action = None
        self.render()
        self.save_image("gui/src/assets/image.png")

    def render(self):
        self._render_env()

    def save_image(self, filename: str):
        pg.image.save(self.surface, filename)

    def _load_env(self):
        filename = DATA_DIR + "env.txt"

        if not os.path.exists(filename):
            logging.info(f"File {filename} does not exist. Generating a new tag.")
            self.env = [[0 for _ in range(self.width)] for _ in range(self.height)]
            self._save_env(filename)
        self._read_env(filename)

    def _read_env(self, filename: str):
        with open(filename, "r") as f:
            env = [list(map(int, line.strip())) for line in f]
            if len(env) != self.height or any(len(row) != self.width for row in env):
                logging.error(f"No data: Failed to read tag from file {filename}.")
                return
            self.env = env

    def _save_env(self, filename: str):
        with open(filename, "w") as f:
            for row in self.env:
                f.write("".join(map(str, row)) + "\n")

    def _update_env(self):
        if self.draw_mode in {DrawMode.OBSTACLE, DrawMode.ERASE, DrawMode.BOX}:
            self._update_tile(self.draw_mode)
        if self.placement_mode in {DrawMode.SEEKER, DrawMode.HIDER}:
            self._update_placement()

    def _update_tile(self, draw_mode: DrawMode):
        value = TileType.EMPTY.value
        match draw_mode:
            case DrawMode.OBSTACLE:
                value = TileType.OBSTACLE.value
            case DrawMode.BOX:
                value = TileType.BOX.value
        self.env[self.current_square[1]][self.current_square[0]] = value

    def _update_placement(self):
        value = (
            TileType.SEEKER.value
            if self.placement_mode == DrawMode.SEEKER
            else TileType.HIDER.value
        )
        for y in range(self.height):
            for x in range(self.width):
                if (
                    self.placement_mode == DrawMode.SEEKER
                    and self.env[y][x] == TileType.SEEKER.value
                ):
                    self.env[y][x] = TileType.EMPTY.value
                if (
                    self.placement_mode == DrawMode.HIDER
                    and self.env[y][x] == TileType.HIDER.value
                ):
                    self.env[y][x] = TileType.EMPTY.value
        self.env[self.current_square[1]][self.current_square[0]] = value

    def _exit_program(self):
        filename = os.path.join(DATA_DIR, "env.txt")
        self._save_env(filename=filename)
        pg.quit()
        sys.exit()

    def _render_env(self):
        for y in range(self.height):
            for x in range(self.width):
                tile_type = TileType(self.env[y][x])
                color = DrawMode.from_tile_type(tile_type).color
                self._draw_cell(x, y, color)

    def _draw_cell(self, x: int, y: int, color: tuple[int, int, int]):
        pg.draw.rect(
            self.surface,
            color,
            (
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )

    @property
    def current_square(self) -> tuple[int, int]:
        return self._current_square

    @current_square.setter
    def current_square(self, value: tuple[int, int]):
        if not (0 <= value[0] < self.width and 0 <= value[1] < self.height):
            return
        self._current_square = value
