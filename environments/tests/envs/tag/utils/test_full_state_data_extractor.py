import unittest
import numpy as np
from environments.gymnasium.envs.tag.utils import AgentType
from environments.gymnasium.envs.tag.utils.tile_type import TileType
from environments.gymnasium.envs.tag.utils.agent_tile_type import AGENT_TILE_TYPE
from environments.gymnasium.utils import Position
from environments.gymnasium.envs.tag.utils import FullStateDataExtractor


class TestFullStateDataExtractor(unittest.TestCase):
    def setUp(self):
        self.state = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.agent_type = AgentType.SEEKER
        AGENT_TILE_TYPE[self.agent_type] = 1
        self.tile_type = TileType.EMPTY  # .value = 0

    def test_get_agent_position(self):
        AGENT_TILE_TYPE[self.agent_type] = 1
        position = FullStateDataExtractor.get_agent_position(
            self.state, self.agent_type
        )
        self.assertEqual(position, Position(x=1, y=0))

    def test_get_agent_position_raises_error_if_multiple_agents(self):
        self.state[1, 1] = AGENT_TILE_TYPE[self.agent_type]
        with self.assertRaises(ValueError):
            FullStateDataExtractor.get_agent_position(self.state, self.agent_type)

    def test_agent_exist(self):
        AGENT_TILE_TYPE[self.agent_type] = 1
        exists = FullStateDataExtractor.agent_exist(self.state, self.agent_type)
        self.assertTrue(exists)

    def test_agent_exist_returns_false_if_agent_does_not_exist(self):
        AGENT_TILE_TYPE[self.agent_type] = 9
        exists = FullStateDataExtractor.agent_exist(self.state, self.agent_type)
        self.assertFalse(exists)

    @unittest.skip("Not implemented")
    def test_get_positions(self):
        self.state[1, 2] = self.tile_type.value
        positions = FullStateDataExtractor.get_positions(self.state, self.tile_type)
        self.assertEqual(positions, [Position(x=2, y=1)])

    def test_is_empty_tile(self):
        empty_position = Position(x=0, y=0)
        self.assertTrue(
            FullStateDataExtractor.is_empty_tile(self.state, empty_position)
        )


if __name__ == "__main__":
    unittest.main()
