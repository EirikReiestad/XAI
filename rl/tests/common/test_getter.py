import unittest
import numpy as np
import torch
from gymnasium import spaces
from rl.src.common import getter


class TestUtils(unittest.TestCase):
    def test_get_numpy_from_numpy(self):
        a = np.array([1, 2, 3])
        result = getter.get_numpy(a)
        np.testing.assert_array_equal(result, a)

    def test_get_numpy_from_tensor(self):
        a = torch.tensor([1, 2, 3])
        result = getter.get_numpy(a)
        np.testing.assert_array_equal(result, a.numpy())

    def test_get_numpy_from_tuple(self):
        a = (1, 2, 3)
        result = getter.get_numpy(a)
        np.testing.assert_array_equal(result, np.array(a))

    def test_get_numpy_invalid_type(self):
        with self.assertRaises(ValueError):
            getter.get_numpy("invalid")

    def test_get_observation_mock_data(self):
        observation_space = spaces.Box(low=0, high=1, shape=(2, 3))
        result = getter.get_observation_mock_data(observation_space)
        np.testing.assert_array_equal(result, np.zeros((2, 3)))

    def test_get_same_type(self):
        a = np.array([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = getter.get_same_type(a, b)
        self.assertTrue(torch.is_tensor(result))

    def test_get_torch_from_numpy(self):
        a = np.array([1, 2, 3])
        result = getter.get_torch_from_numpy(a)
        self.assertTrue(torch.is_tensor(result))

    def test_get_torch_from_tensor(self):
        a = torch.tensor([1, 2, 3])
        result = getter.get_torch_from_numpy(a)
        self.assertTrue(torch.is_tensor(result))


if __name__ == "__main__":
    unittest.main()
