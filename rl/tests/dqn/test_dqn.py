import unittest
import torch
import torch.nn as nn
from src.nn.nn import NeuralNetwork


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        input_size = 4
        output_size = 2
        self.model = SimpleModel(input_size, output_size)
        self.neural_network = NeuralNetwork(
            self.model, lr=0.01, gamma=0.9, epsilon=0.1)
        self.state = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        self.next_state = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        self.action = 0
        self.reward = 1.0
        self.path = "test_model.pth"

    def test_choose_action(self):
        action = self.neural_network.choose_action(self.state)
        self.assertIn(action, [0, 1])

    def test_update(self):
        initial_params = list(self.model.parameters())[0].clone()
        self.neural_network.update(
            self.state, self.action, self.reward, self.next_state)
        updated_params = list(self.model.parameters())[0].clone()
        self.assertFalse(torch.equal(initial_params, updated_params))

    def test_save_and_load(self):
        self.neural_network.save(self.path)
        new_model = SimpleModel(4, 2)
        new_neural_network = NeuralNetwork(new_model)
        new_neural_network.load(self.path)
        for param, new_param in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(param, new_param))

    def test_predict(self):
        prediction = self.neural_network.predict(self.state)
        self.assertEqual(prediction.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()
