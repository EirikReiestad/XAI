import unittest
import numpy as np
from environments.gymnasium.envs.tag.utils import (
    AgentType,
    TileType,
    Objects,
)
from environments.gymnasium.utils import Position, StateType
from environments.gymnasium.envs.tag.tag_state import TagState


class TestTagState(unittest.TestCase):
    def setUp(self):
        self.screen_width = 300
        self.screen_height = 300
        self.state_type = StateType.FULL
        self.filename = "test_state.txt"
        # Create a dummy file for testing
        map = """12345\n00000\n00000\n00000\n00000"""

        with open(self.filename, "w") as f:
            f.write(map)

        self.tag_state = TagState(
            self.screen_width,
            self.screen_height,
            self.state_type,
            self.filename,
        )

    def tearDown(self):
        import os

        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_init_states(self):
        self.assertIsNotNone(self.tag_state.state.full)
        self.assertIsNotNone(self.tag_state.state.partial)
        self.assertIsNotNone(self.tag_state.state.rgb)

    @unittest.skip("Not implemented")
    def test_reset(self):
        self.tag_state.reset()
        self.assertEqual(
            self.tag_state.state.full.tolist(), self.tag_state.init_full_state.tolist()
        )

    def test_update(self):
        new_full_state = (
            np.ones((self.tag_state.height, self.tag_state.width), dtype=np.uint8)
            * TileType.OBSTACLE.value
        )
        seeker_position = Position(5, 5)
        hider_position = Position(6, 6)
        objects = Objects([], [])  # Empty objects
        self.tag_state.update(new_full_state, seeker_position, hider_position, objects)
        self.assertTrue(np.array_equal(self.tag_state.state.full, new_full_state))

    def test_get_agent_position(self):
        position = self.tag_state.get_agent_position(AgentType.SEEKER)
        self.assertIsInstance(position, Position)

    @unittest.skip("Not implemented")
    def test_concatenate_states(self):
        state1 = np.zeros((self.tag_state.height, self.tag_state.width), dtype=np.uint8)
        state1[0] = [i for i in range(self.tag_state.width)]
        state2 = (
            np.ones((self.tag_state.height, self.tag_state.width), dtype=np.uint8)
            * TileType.BOX.value
        )
        state2[-1] = [i for i in range(self.tag_state.width)]
        concatenated_state, is_same = self.tag_state.concatenate_states(
            [state1, state2]
        )
        self.assertEqual(
            concatenated_state.shape, (self.tag_state.height, self.tag_state.width)
        )
        self.assertIsInstance(is_same, bool)

    def test_get_obstacle_positions(self):
        positions = self.tag_state.get_obstacle_positions()
        self.assertIsInstance(positions, list)
        if positions:
            self.assertIsInstance(positions[0], Position)

    def test_get_all_possible_states(self):
        objects = Objects([], [])
        states = self.tag_state.get_all_possible_states(
            AgentType.SEEKER, AgentType.HIDER, objects
        )
        for state in states:
            self.assertEqual(state.shape[1:], self.tag_state.full_state_size)

    @unittest.skip("Not implemented")
    def test_get_occluded_states(self):
        if self.state_type == StateType.FULL:
            occluded_states = self.tag_state.get_occluded_states()
            self.assertEqual(occluded_states.shape[1:], self.tag_state.full_state_size)
        else:
            with self.assertRaises(ValueError):
                self.tag_state.get_occluded_states()


if __name__ == "__main__":
    unittest.main()
