import unittest
import numpy as np
from environments.gymnasium.envs.tag.utils import (
    FullStateDataModifier,
    AgentType,
    AGENT_TILE_TYPE,
)
from environments.gymnasium.envs.tag.utils.tile_type import TileType
from environments.gymnasium.utils import Position


class TestFullStateDataModifier(unittest.TestCase):
    def setUp(self):
        self.state = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.agent_type = AgentType.SEEKER
        self.other_agent_type = AgentType.HIDER
        AGENT_TILE_TYPE[self.agent_type] = 0
        self.tile_type = TileType.EMPTY  # .value = 0

    @unittest.skip("Not implemented")
    def test_remove_agent(self):
        self.state[1, 1] = AGENT_TILE_TYPE[self.agent_type]
        new_state = FullStateDataModifier.remove_agent(self.state, self.agent_type)
        self.assertEqual(new_state[1, 1], TileType.EMPTY.value)

    def test_remove_agent_raises_error_if_agent_already_removed(self):
        self.state[1, 1] = TileType.EMPTY.value
        with self.assertRaises(ValueError):
            FullStateDataModifier.remove_agent(self.state, self.agent_type)

    @unittest.skip("Not implemented")
    def test_remove_agents(self):
        self.state[1, 1] = AGENT_TILE_TYPE[self.agent_type]
        self.state[0, 0] = AGENT_TILE_TYPE[self.other_agent_type]
        agents = [self.agent_type, self.other_agent_type]
        new_state = FullStateDataModifier.remove_agents(self.state, agents)
        self.assertEqual(new_state[1, 1], TileType.EMPTY.value)
        self.assertEqual(new_state[0, 0], TileType.EMPTY.value)

    def test_occlude(self):
        position = Position(x=1, y=1)
        new_state = FullStateDataModifier.occlude(self.state, position)
        self.assertEqual(new_state[1, 1], TileType.EMPTY.value)

    def test_place_agent(self):
        position = Position(x=1, y=1)
        new_state = FullStateDataModifier.place_agent(
            self.state, position, self.agent_type
        )
        self.assertEqual(new_state[1, 1], AGENT_TILE_TYPE[self.agent_type])


if __name__ == "__main__":
    unittest.main()
