from typing import Optional

import numpy as np
import pygame as pg

from environments.gymnasium.envs.coop.utils import TileType
from environments.gymnasium.utils import Color


class CoopRenderer:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, height: int, width: int, screen_width: int, screen_height: int):
        self.height = height
        self.width = width
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.render_mode = None
        self.init_render()

    def init_render(self):
        """Initializes rendering settings."""
        pg.init()
        pg.display.init()
        self.surface = pg.Surface((self.screen_width, self.screen_height))
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pg.time.Clock()
        self.is_open = True

    def init_render_mode(self, render_mode: Optional[str] = "human"):
        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render mode {render_mode}. "
                f"Available modes: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

    def render(
        self, state: np.ndarray, render_mode: Optional[str] = None
    ) -> Optional[np.ndarray]:
        render_mode = render_mode or self.render_mode
        self.init_render_mode(render_mode)

        color_matrix = np.full((self.height, self.width, 3), Color.WHITE.value)
        self.apply_color_masks(color_matrix, state)

        surf = pg.surfarray.make_surface(color_matrix)
        surf = pg.transform.scale(surf, (self.screen_height, self.screen_width))
        surf = pg.transform.flip(surf, True, False)
        surf = pg.transform.rotate(surf, 90)

        if self.render_mode == "rgb_array":
            self.surface.blit(surf, (0, 0))
            surf_array3d = pg.surfarray.array3d(self.surface)
            return surf_array3d
        elif self.render_mode == "human":
            self.screen.blit(surf, (0, 0))
            pg.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pg.display.flip()
            return None
        else:
            raise ValueError(f"Invalid render mode {self.render_mode}")

    def close(self):
        if hasattr(self, "screen") and self.screen:
            pg.display.quit()
            pg.quit()
            self.is_open = False

    def apply_color_masks(self, color_matrix, full_state):
        """Applies color masks to the coop."""
        color_matrix[full_state == TileType.OBSTACLE.value] = Color.BLACK.value
        color_matrix[full_state == TileType.START.value] = Color.BLUE.value
        color_matrix[full_state == TileType.END.value] = Color.GREEN.value
