import unittest
import numpy as np
import pygame as pg
from environments.gymnasium.envs.tag.tag_renderer import TagRenderer
from environments.gymnasium.envs.tag.utils import TileType


class TestTagRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = TagRenderer(10, 10, 300, 300)
        self.state = np.zeros((10, 10), dtype=np.uint8)
        self.state[0, 0] = TileType.OBSTACLE.value
        self.state[1, 1] = TileType.BOX.value
        self.state[2, 2] = TileType.SEEKER.value
        self.state[3, 3] = TileType.HIDER.value

    def test_init_render(self):
        self.assertTrue(self.renderer.is_open)

    def test_init_render_mode(self):
        self.renderer.init_render_mode("rgb_array")
        self.assertEqual(self.renderer.render_mode, "rgb_array")
        with self.assertRaises(ValueError):
            self.renderer.init_render_mode("invalid_mode")

    def test_render_rgb_array(self):
        self.renderer.init_render_mode("rgb_array")
        result = self.renderer.render(self.state)
        assert result is not None
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (300, 300, 3))

    def test_render_human(self):
        self.renderer.init_render_mode("human")
        result = self.renderer.render(self.state)
        self.assertIsNone(result)

    @unittest.skip("Not implemented")
    def test_close(self):
        self.renderer.close()
        self.assertFalse(self.renderer.is_open)
        # Check if pygame quit properly
        self.assertFalse(pg.get_init())


if __name__ == "__main__":
    unittest.main()
