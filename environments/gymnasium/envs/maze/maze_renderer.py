from typing import Optional, Any

import numpy as np
import pygame as pg

from environments.gymnasium.envs.maze.utils import TileType
from environments.gymnasium.utils import Color


class MazeRenderer:
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

    def init_render_q_values(self, q_values: np.ndarray | None):
        if q_values is None:
            return
        if q_values.shape[0] != self.height or q_values.shape[1] != self.width:
            raise ValueError(
                f"Q-values must have shape ({self.height}, {self.width}), got {q_values.shape}"
            )

    def render(
        self, state: np.ndarray, info: Optional[dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        render_mode = None
        q_values = None
        if info is not None:
            render_mode = info.get("render_mode")
            q_values = info.get("q_values")

        if q_values is not None:
            if q_values.shape[0] != self.height or q_values.shape[1] != self.width:
                raise ValueError(
                    f"Q-values must have shape ({self.height}, {self.width}), got {q_values.shape}"
                )

        self.init_render_mode(render_mode)

        color_matrix = np.full((self.height, self.width, 3), Color.WHITE.value)
        self.apply_color_masks(color_matrix, state, q_values)

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

    def _blend_colors(
        self, color1: tuple[int, int, int], color2: tuple[int, int, int], ratio: float
    ) -> tuple[int, int, int]:
        color_blend = tuple(
            int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2)
        )
        if len(color_blend) != 3:
            raise ValueError("Color blend must be a 3-tuple.")
        return color_blend

    def _q_value_to_color(
        self, q_value: float | None, min_q: float, max_q: float
    ) -> tuple[int, int, int]:
        """Map a Q-value to a color based on its magnitude."""
        if q_value is None:
            return Color.WHITE.value  # Default color for None values
        normalized = (q_value - min_q) / (max_q - min_q)  # Normalize Q-value
        if normalized < 0.5:
            return self._blend_colors(
                Color.WHITE.value, Color.PURPLE.value, normalized * 2
            )  # Low to medium Q-values
        else:
            return self._blend_colors(
                Color.PURPLE.value, Color.RED.value, (normalized - 0.5) * 2
            )  # Medium to high Q-values

    def apply_color_masks(
        self, color_matrix, full_state, q_values: np.ndarray | None = None
    ):
        """Applies color masks to the maze."""
        color_matrix[full_state == TileType.OBSTACLE.value] = Color.BLACK.value
        color_matrix[full_state == TileType.START.value] = Color.BLUE.value
        color_matrix[full_state == TileType.END.value] = Color.GREEN.value

        if q_values is None:
            return

        sum_q_values = np.zeros((self.height, self.width), dtype=float)

        for y in range(self.height):
            for x in range(self.width):
                up, down, left, right = q_values[x, y]  # Same order as in Direction
                if x > 0:
                    sum_q_values[y, x - 1] += left
                if x < self.height - 1:
                    sum_q_values[y, x + 1] += right
                if y > 0:
                    sum_q_values[y - 1, x] += down
                if y < self.width - 1:
                    sum_q_values[y + 1, x] += up

        normalized_q_values = sum_q_values / np.max(sum_q_values)
        min_q: float = float(np.min(normalized_q_values))
        max_q: float = float(np.max(normalized_q_values))

        for x in range(self.height):
            for y in range(self.width):
                if full_state[x, y] != TileType.EMPTY.value:
                    continue
                color_matrix[x, y] = self._q_value_to_color(
                    normalized_q_values[x, y], min_q, max_q
                )
