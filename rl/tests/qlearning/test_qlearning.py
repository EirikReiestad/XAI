import unittest
import numpy as np
from unittest.mock import patch, mock_open
from rl.src.qlearning.qlearning import QLearning


class TestQLearning(unittest.TestCase):
    def setUp(self):
        self.q_learning = QLearning(action_space=4, alpha=0.1, gamma=0.6, epsilon=0.1)

    def test_initialization(self):
        self.assertEqual(self.q_learning.action_space, 4)
        self.assertEqual(self.q_learning.alpha, 0.1)
        self.assertEqual(self.q_learning.gamma, 0.6)
        self.assertEqual(self.q_learning.epsilon, 0.1)
        self.assertEqual(len(self.q_learning.q_table), 0)

    @patch("random.uniform")
    def test_choose_action_explore(self, mock_uniform):
        mock_uniform.return_value = 0.05  # Less than epsilon
        with patch("random.randint", return_value=2):
            action = self.q_learning.choose_action(0)
        self.assertEqual(action, 2)

    @patch("random.uniform")
    def test_choose_action_exploit(self, mock_uniform):
        mock_uniform.return_value = 0.2  # Greater than epsilon
        self.q_learning.q_table[0] = np.array([1, 2, 3, 4])
        action = self.q_learning.choose_action(0)
        self.assertEqual(action, 3)

    def test_update(self):
        self.q_learning.update(0, 1, 1.0, 1)
        self.assertIn(0, self.q_learning.q_table)
        self.assertIn(1, self.q_learning.q_table)
        self.assertEqual(len(self.q_learning.q_table[0]), 4)
        self.assertEqual(len(self.q_learning.q_table[1]), 4)
        self.assertGreater(self.q_learning.q_table[0][1], 0)

    @patch("numpy.save")
    def test_save(self, mock_save):
        self.q_learning.q_table = {0: np.array([1, 2, 3, 4])}
        self.q_learning.save("test_path")
        mock_save.assert_called_once()

    @patch("numpy.load")
    def test_load(self, mock_load):
        mock_load.return_value = {0: np.array([1, 2, 3, 4])}
        self.q_learning.load("test_path")
        self.assertEqual(len(self.q_learning.q_table), 1)
        np.testing.assert_array_equal(
            self.q_learning.q_table[0], np.array([1, 2, 3, 4])
        )


if __name__ == "__main__":
    unittest.main()
