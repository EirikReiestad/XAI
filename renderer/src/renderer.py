import numpy as np
import pygame as pg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from utils import Color
from .utils import q_value_to_color


class Renderer:
    metadata = {"render_fps": 50}

    def __init__(self, height: int, width: int, screen_width: int, screen_height: int):
        self.height = height
        self.width = width
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.render_mode = None
        self._init_render()

    def _init_render(self):
        """Initializes rendering settings."""
        pg.init()
        pg.display.init()
        self.surface = pg.Surface((self.screen_width, self.screen_height))
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pg.time.Clock()
        self.is_open = True

    def render(self, background: np.ndarray, **kwargs):
        self.screen.fill(Color.WHITE.value)
        q_values = kwargs.get("q_values")
        if q_values is not None:
            q_values_surf = self._get_q_values_surf(q_values)
            q_values_surf.set_alpha(128)
            self.screen.blit(q_values_surf, (0, 0))

        saliency_map = kwargs.get("saliency_map")
        if saliency_map is not None:
            print("Rendering saliency_map")
            rgb_array = self._create_heatmap(saliency_map)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))
            surf = pg.surfarray.make_surface(rgb_array)
            saliency_map_surf = pg.surfarray.make_surface(rgb_array)
            saliency_map_surf = pg.transform.scale(
                saliency_map_surf, (self.screen_height, self.screen_width)
            )
            saliency_map_surf = pg.transform.flip(saliency_map_surf, True, False)
            saliency_map_surf = pg.transform.rotate(saliency_map_surf, 90)
            self.screen.blit(saliency_map_surf, (0, 0))

        surf = pg.surfarray.make_surface(background)
        surf.set_colorkey(Color.WHITE.value)
        self.screen.blit(surf, (0, 0))

        pg.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pg.display.flip()

    def _create_heatmap(self, values: np.ndarray) -> np.ndarray:
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        colormap = cm.get_cmap("hot")
        colored_map = colormap(norm(values))
        rgb_array = (colored_map[..., :3] * 255).astype(np.uint8)
        return rgb_array

    def _get_q_values_surf(self, q_values: np.ndarray) -> pg.Surface:
        color_matrix = np.full((self.height, self.width, 3), Color.WHITE.value)
        self._apply_q_values_color_masks(color_matrix, q_values)

        surf = pg.surfarray.make_surface(color_matrix)
        surf = pg.transform.scale(surf, (self.screen_height, self.screen_width))
        surf = pg.transform.flip(surf, True, False)
        surf = pg.transform.rotate(surf, 90)

        return surf

    def close(self):
        if hasattr(self, "screen") and self.screen:
            pg.display.quit()
            pg.quit()
            self.is_open = False

    def _apply_q_values_color_masks(
        self, color_matrix: np.ndarray, q_values: np.ndarray
    ):
        """Applies color masks to the maze."""
        min_q: float = float(np.min(q_values))
        max_q: float = float(np.max(q_values))

        for x in range(self.height):
            for y in range(self.width):
                color_matrix[x, y] = q_value_to_color(q_values[x, y], min_q, max_q)
