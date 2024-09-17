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
        self.width = 10
        self.height = 10
        self.screen_width = 300
        self.screen_height = 300
        self.state_type = StateType.FULL
        self.filename = "test_state.txt"
        # Create a dummy file for testing
        with open(self.filename, "w") as f:
            f.write("0" * self.width + "\n" * (self.height - 1) + "0" * self.width)
        self.tag_state = TagState(
            self.width,
            self.height,
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

    def test_reset(self):
        self.tag_state.reset()
        self.assertEqual(
            self.tag_state.state.full.tolist(), self.tag_state.init_full_state.tolist()
        )

    def test_update(self):
        new_full_state = (
            np.ones((self.height, self.width), dtype=np.uint8) * TileType.OBSTACLE.value
        )
        seeker_position = Position(5, 5)
        hider_position = Position(6, 6)
        objects = Objects([], [])  # Empty objects
        self.tag_state.update(new_full_state, seeker_position, hider_position, objects)
        self.assertTrue(np.array_equal(self.tag_state.state.full, new_full_state))

    def test_get_agent_position(self):
        position = self.tag_state.get_agent_position(AgentType.SEEKER)
        self.assertIsInstance(position, Position)

    def test_concatenate_states(self):
        state1 = np.zeros((self.height, self.width), dtype=np.uint8)
        state2 = np.ones((self.height, self.width), dtype=np.uint8) * TileType.BOX.value
        concatenated_state, is_same = self.tag_state.concatenate_states(
            [state1, state2]
        )
        self.assertEqual(concatenated_state.shape, (self.height, self.width))
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
        self.assertEqual(states.shape[1:], self.tag_state.partial_state_size)

    def test_get_occluded_states(self):
        if self.state_type == StateType.FULL:
            occluded_states = self.tag_state.get_occluded_states()
            self.assertEqual(occluded_states.shape[1:], self.tag_state.full_state_size)
        else:
            with self.assertRaises(ValueError):
                self.tag_state.get_occluded_states()


if __name__ == "__main__":
    unittest.main()
