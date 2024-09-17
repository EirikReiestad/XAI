import unittest
import numpy as np
from gymnasium import spaces
from rl.src.common import getter, checker


class TestRaiseIfNotSameShape(unittest.TestCase):
    def setUp(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 4))
        self.mock_data = getter.get_observation_mock_data(self.observation_space)

    def test_raise_if_not_same_shape(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(3, 4)
        try:
            checker.raise_if_not_same_shape(a, b)
        except ValueError:
            self.fail("raise_if_not_same_shape() raised ValueError unexpectedly!")

    def test_raise_if_not_same_shape_different_shape(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(2, 4)
        with self.assertRaises(ValueError):
            checker.raise_if_not_same_shape(a, b)

    def test_raise_if_not_all_same_shape(self):
        a_vec = [np.random.rand(3, 4), np.random.rand(3, 4)]
        b = np.random.rand(3, 4)
        try:
            checker.raise_if_not_all_same_shape(a_vec, b)
        except ValueError:
            self.fail("raise_if_not_all_same_shape() raised ValueError unexpectedly!")

    def test_raise_if_not_all_same_shape_different_shapes(self):
        a_vec = [np.random.rand(3, 4), np.random.rand(2, 4)]
        b = np.random.rand(3, 4)
        with self.assertRaises(ValueError):
            checker.raise_if_not_all_same_shape(a_vec, b)

    def test_raise_if_not_same_shape_as_observation(self):
        a = np.random.rand(3, 4)
        try:
            checker.raise_if_not_same_shape_as_observation(a, self.observation_space)
        except ValueError:
            self.fail(
                "raise_if_not_same_shape_as_observation() raised ValueError unexpectedly!"
            )

    def test_raise_if_not_all_same_shape_as_observation(self):
        a_vec = [np.random.rand(3, 4), np.random.rand(3, 4)]
        try:
            checker.raise_if_not_all_same_shape_as_observation(
                a_vec, self.observation_space
            )
        except ValueError:
            self.fail(
                "raise_if_not_all_same_shape_as_observation() raised ValueError unexpectedly!"
            )
