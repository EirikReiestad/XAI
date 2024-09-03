"""
We will generate an env, and store it as a text file for later use.
"""

import logging
import os
import sys

import pygame as pg

from environments.gymnasium.envs.tag.utils import TileType
from environments.gymnasium.utils import Direction
from utils.src.tag_generator.utils import DrawMode

logging.basicConfig(level=logging.INFO)

TAG_DATA_DIR = "environments/gymnasium/data/tag/"

if not os.path.exists(TAG_DATA_DIR):
    logging.info(f"Current directory: {os.getcwd()}")
    raise FileNotFoundError(
        f"Directory {TAG_DATA_DIR} does not exist. Create it and try again."
    )


class GenerateEnv:
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

        self.env: list[list[int]] = []
        self.env_index = 0
        self._load_env(self.env_index)

    def generate(self):
        """Let the user generate an env by drawing obstacles"""

        while True:
            self.screen.fill((255, 255, 255))
            self.render()
            self._handle_input()
            self._move_current_square()
            self._update_env()
            self.action = None
            pg.display.flip()

    def render(self):
        self._render_env()
        self._render_header()

    def _render_header(self):
        self.header.fill((255, 255, 255))
        self._draw_text(self.header, f"Current tag: {self.env_index}", (10, 5))
        self._draw_text(self.header, f"Current tag: {self.draw_mode}", (10, 30))
        self.screen.blit(self.header, (0, 0))

    def _draw_text(self, surface: pg.Surface, text: str, position: tuple[int, int]):
        font = pg.font.Font(None, 36)
        text_surface = font.render(text, True, (0, 0, 0))
        surface.blit(text_surface, position)

    def _load_env(self, index: int):
        filename = TAG_DATA_DIR + f"tag-{index}-{self.height}-{self.width}.txt"

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

    def _handle_input(self):
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self._exit_program()
                if event.key in {pg.K_UP, pg.K_DOWN, pg.K_LEFT, pg.K_RIGHT}:
                    self.action = Direction(pg.K_UP - event.key)
                if event.key == pg.K_SPACE:
                    self.draw_mode += 1
                if event.key == pg.K_0:
                    self.placement_mode = DrawMode.SEEKER
                if event.key == pg.K_1:
                    self.placement_mode = DrawMode.HIDER
                if event.key == pg.K_n:
                    self.env_index += 1
                    self._load_env(self.env_index)
                if event.key == pg.K_p:
                    self.env_index = max(0, self.env_index - 1)
                    self._load_env(self.env_index)

    def _exit_program(self):
        filename = os.path.join(
            TAG_DATA_DIR, f"tag-{self.env_index}-{self.height}-{self.width}.txt"
        )
        self._save_env(filename=filename)
        pg.quit()
        sys.exit()

    def _render_env(self):
        for y in range(self.height):
            for x in range(self.width):
                tile_type = TileType(self.env[y][x])
                color = DrawMode.from_tile_type(tile_type).color
                self._draw_cell(x, y, color)
        self._render_mode_highlight()

    def _render_mode_highlight(self):
        color = self.placement_mode.get_highlight_color()
        if color is None:
            color = self.draw_mode.get_highlight_color()
        if color is None:
            color = (200, 200, 200)

        self._draw_cell(self.current_square[0], self.current_square[1], color)

    def _draw_cell(self, x: int, y: int, color: tuple[int, int, int]):
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
