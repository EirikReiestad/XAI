import unittest
import numpy as np
from rl.src.dqn.common.q_values_map import get_q_values_map


class TestGetQValuesMap(unittest.TestCase):
    def setUp(self):
        self.states = np.zeros((3, 3))
        self.q_values = np.random.rand(3, 3, 4)

    def test_valid_shape(self):
        result = get_q_values_map(self.states, self.q_values)
        self.assertEqual(result.shape, (3, 3))

    def test_invalid_shape(self):
        states_invalid = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            get_q_values_map(states_invalid, self.q_values)

    def test_max_q_values(self):
        result = get_q_values_map(self.states, self.q_values, max_q_values=True)
        expected = np.max(self.q_values, axis=2)
        np.testing.assert_almost_equal(
            result, (expected - np.min(expected)) / np.ptp(expected)
        )

    def test_cumulated_q_values(self):
        result = get_q_values_map(self.states, self.q_values)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))


if __name__ == "__main__":
    unittest.main()
