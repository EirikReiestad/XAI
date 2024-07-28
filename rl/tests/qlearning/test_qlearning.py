import unittest
import numpy as np
from rl.src.qlearning.qlearning import QLearning


class TestQLearning(unittest.TestCase):

    def setUp(self):
        self.qlearning = QLearning(action_space=4)
        self.qlearning.epsilon = 0.0

    def test_initialization(self):
        self.assertEqual(self.qlearning.action_space, 4)
        self.assertEqual(len(self.qlearning.q_table), 0)
        self.assertEqual(self.qlearning.alpha, 0.1)
        self.assertEqual(self.qlearning.gamma, 0.6)
        self.assertEqual(self.qlearning.epsilon, 0.0)

    def test_choose_action_random(self):
        state = 0
        action = self.qlearning.choose_action(state)
        self.assertIn(action, range(self.qlearning.action_space))

    def test_choose_action_greedy(self):
        state = 1
        self.qlearning.q_table[state] = np.array([0.1, 0.2, 0.3, 0.4])
        action = self.qlearning.choose_action(state)
        # The action with the highest Q-value
        self.assertEqual(action, 3)

    def test_update(self):
        state = 2
        action = 1
        reward = 1.0
        next_state = 3

        self.qlearning.q_table[state] = np.array([0.0, 0.0, 0.0, 0.0])
        self.qlearning.q_table[next_state] = np.array([0.5, 0.5, 0.5, 0.5])

        self.qlearning.update(state, action, reward, next_state)
        # alpha * (reward + gamma * max(next_state_q) - current_q)
        expected_q_value = 0.1 * (1.0 + 0.6 * 0.5)
        self.assertAlmostEqual(
            self.qlearning.q_table[state][action], expected_q_value)

    def test_update_with_uninitialized_state(self):
        state = 4
        action = 2
        reward = 2.0
        next_state = 5

        self.qlearning.update(state, action, reward, next_state)
        # alpha * (reward + gamma * max(next_state_q) - current_q)
        expected_q_value = 0.1 * (2.0 + 0.6 * 0.0)
        self.assertAlmostEqual(
            self.qlearning.q_table[state][action], expected_q_value)
        self.assertTrue(np.array_equal(
            self.qlearning.q_table[next_state], np.zeros(self.qlearning.action_space)))


if __name__ == '__main__':
    unittest.main()
