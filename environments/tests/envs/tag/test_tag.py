import unittest
import numpy as np
from environments.gymnasium.envs.tag import TagEnv
from environments.gymnasium.envs.tag.utils import AgentType, ActionType


class TestTagEnv(unittest.TestCase):
    def setUp(self):
        self.env = TagEnv(render_mode="rgb_array")
        self.env.reset()

    def test_step_valid_action(self):
        action = ActionType.UP.value  # Replace with an actual valid action
        state, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(reward, (float, int))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("full_state", info)
        # self.assertIn("data", info)

    def test_step_invalid_action(self):
        with self.assertRaises(ValueError):
            self.env.step(-1)  # Invalid action

    def test_reset(self):
        state, info = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertIn("state_type", info)

    def test_render(self):
        self.env.render()  # Ensure no errors are raised
        # Add additional checks if needed based on rendering

    def test_close(self):
        self.env.close()  # Ensure no errors are raised

    def test_set_active_agent(self):
        self.env.set_active_agent(AgentType.SEEKER)
        self.assertEqual(self.env.agents.active_agent, AgentType.SEEKER)

    def test_get_all_possible_states(self):
        states = self.env.get_all_possible_states()
        self.assertIsInstance(states, np.ndarray)

    def test_get_occluded_states(self):
        if self.env.state_type == "full":
            occluded_states = self.env.get_occluded_states()
            self.assertIsInstance(occluded_states, np.ndarray)


if __name__ == "__main__":
    unittest.main()
