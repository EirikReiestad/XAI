import logging
from typing import Optional

import numpy as np
import pygame as pg

from environments.gymnasium.envs.tag.utils import TileType
from environments.gymnasium.utils import Position
from utils import Color

from .utils import ActionType


class TagRenderer:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, width: int, height: int, screen_width: int, screen_height: int):
        self._width = width
        self._height = height
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._render_mode = None

        self._init_render()
        self._post_init_screen = False
        self._post_init_surface = False

        self._direct_sight_positions = []
        self.seeker_action = ActionType.UP
        self.hider_action = ActionType.UP
        self._scale = (
            self._screen_height // self._height,
            self._screen_width // self._width,
        )

        self._load_sprites()

    def _load_sprites(self):
        try:
            seeker_sprite = pg.image.load("assets/sprites/tom.png")
            hider_sprite = pg.image.load("assets/sprites/jerry.png")
            box_sprite = pg.image.load("assets/sprites/cheese.png")
            obstacle_sprite = pg.image.load("assets/sprites/bush.png")

            scaled_seeker_sprite = pg.transform.scale(seeker_sprite, self._scale)
            scaled_hider_sprite = pg.transform.scale(hider_sprite, self._scale)
            scaled_box_sprite = pg.transform.scale(box_sprite, self._scale)
            obstacle_sprite = pg.transform.scale(obstacle_sprite, self._scale)
            self._seeker_sprite = scaled_seeker_sprite
            self._hider_sprite = scaled_hider_sprite
            self._hider_sprite = pg.transform.flip(scaled_hider_sprite, True, False)
            self._box_sprite = scaled_box_sprite
            self._obstacle_sprite = obstacle_sprite
        except FileNotFoundError:
            logging.warning("Sprites not found. Rendering without sprites.")
            self._seeker_sprite = None
            self._hider_sprite = None
            self._box_sprite = None
            self._obstacle_sprite = None

    def _init_render(self):
        """Initializes rendering settings."""
        pg.init()
        self.clock = pg.time.Clock()
        self.is_open = True

    def _init_surface(self):
        if self._post_init_surface:
            return
        self.surface = pg.Surface((self._screen_width, self._screen_height))
        self._post_init_surface = True

    def _init_screen(self):
        if self._post_init_screen:
            return
        pg.display.init()
        pg.display.set_caption("")
        self.screen = pg.display.set_mode((self._screen_width, self._screen_height))
        self._post_init_screen = True

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    @render_mode.setter
    def render_mode(self, render_mode: Optional[str] = None):
        if render_mode is None:
            return self._render_mode
        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render mode {render_mode}. "
                f"Available modes: {self.metadata['render_modes']}"
            )
        self._render_mode = render_mode

    def render(
        self,
        state: np.ndarray,
        render_mode: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        self.render_mode = render_mode

        color_matrix = np.full((self._height, self._width, 3), Color.WHITE.value)
        self.apply_color_masks(color_matrix, state)

        for position in self._direct_sight_positions:
            color_matrix[*position.row_major_order] = Color.LIGHT_GRAY.value

        surf = pg.surfarray.make_surface(color_matrix)
        surf = pg.transform.scale(surf, (self._screen_height, self._screen_width))
        surf = pg.transform.flip(surf, True, False)
        surf = pg.transform.rotate(surf, 90)

        surf = self._apply_sprites(surf, state)

        if self.render_mode == "rgb_array":
            self._init_surface()
            self.surface.blit(surf, (0, 0))
            surf_array3d = pg.surfarray.array3d(self.surface)
            return surf_array3d
        elif self.render_mode == "human":
            self._init_screen()
            self.screen.blit(surf, (0, 0))
            pg.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pg.display.flip()
            return None
        else:
            raise ValueError(f"Invalid render mode {self.render_mode}")

    def _apply_sprites(self, surf: pg.Surface, state: np.ndarray):
        if self._seeker_sprite is not None:
            seeker_position = np.argwhere(state == TileType.SEEKER.value)[0]
            sprite = self._transform_sprite(self._seeker_sprite, self.seeker_action)
            surf.blit(
                sprite,
                seeker_position[::-1] * self._scale[0],
            )
        if self._hider_sprite is not None:
            hider_position = np.argwhere(state == TileType.HIDER.value)[0]
            sprite = self._transform_sprite(self._hider_sprite, self.hider_action)
            surf.blit(sprite, hider_position[::-1] * self._scale[0])
        if self._box_sprite is not None:
            box_positions = np.argwhere(state == TileType.BOX.value)
            for box_position in box_positions:
                surf.blit(self._box_sprite, box_position[::-1] * self._scale[0])
        if self._obstacle_sprite is not None:
            obstacle_positions = np.argwhere(state == TileType.OBSTACLE.value)
            for obstacle_position in obstacle_positions:
                surf.blit(
                    self._obstacle_sprite, obstacle_position[::-1] * self._scale[0]
                )
        return surf

    def _transform_sprite(self, sprite: pg.Surface, action: ActionType):
        if action == ActionType.UP:
            return pg.transform.rotate(sprite, -90)
        elif action == ActionType.DOWN:
            return pg.transform.rotate(sprite, 90)
        elif action == ActionType.LEFT:
            return pg.transform.rotate(sprite, 0)
        elif action == ActionType.RIGHT:
            return pg.transform.flip(sprite, True, False)
        else:
            return sprite

    def close(self):
        if hasattr(self, "screen") and self.screen:
            pg.display.quit()
            pg.quit()
            self.is_open = False

    def apply_color_masks(self, color_matrix, full_state):
        """Applies color masks to the tag."""
        color_matrix[full_state == TileType.OBSTACLE.value] = Color.BLACK.value
        color_matrix[full_state == TileType.BOX.value] = Color.YELLOW.value
        color_matrix[full_state == TileType.SEEKER.value] = Color.BLUE.value
        color_matrix[full_state == TileType.HIDER.value] = Color.GREEN.value

    @property
    def direct_sight_positions(self):
        return self._direct_sight_positions

    @direct_sight_positions.setter
    def direct_sight_positions(self, direct_sight_positions: list[Position]):
        assert all(
            isinstance(position, Position) for position in direct_sight_positions
        )
        self._direct_sight_positions = direct_sight_positions
