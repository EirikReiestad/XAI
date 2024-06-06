import unittest
import torch
import torch.nn as nn
from src.dqn.dqn import DQN


class TestDQN(unittest.TestCase):
    def setUp(self):
        self.dqn = DQN(input_dim=4, output_dim=2, hidden_dims=[],
                       lr=0.01, gamma=0.9, epsilon=0.1, batch_size=1)
        self.state = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        self.next_state = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        self.action = 0
        self.reward = 1.0
        self.path = "test_model.pth"

    def test_choose_action(self):
        action = self.dqn.choose_action(self.state)
        self.assertIn(action, [0, 1])

    def test_update(self):
        initial_params = list(self.dqn.model.parameters())[0].clone()
        self.dqn.update(
            self.state, self.action, self.reward, self.next_state)
        updated_params = list(self.dqn.model.parameters())[0].clone()
        self.assertFalse(torch.equal(initial_params, updated_params))

    def test_save_and_load(self):
        self.dqn.save(self.path)
        new_dqn = DQN(input_dim=4, output_dim=2, hidden_dims=[],
                      lr=0.01, gamma=0.9, epsilon=0.1)
        new_dqn.load(self.path)
        for param, new_param in zip(self.dqn.model.parameters(), new_dqn.model.parameters()):
            self.assertTrue(torch.equal(param, new_param))

    def test_predict(self):
        prediction = self.dqn.predict(self.state)
        self.assertEqual(prediction.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()
