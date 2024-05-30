import torch
import torch.nn as nn
import os
import logging


class NeuralNetwork:
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: [int], lr: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1):
        assert isinstance(
            input_dim, int), "input_dim must be an instance of int"
        assert isinstance(
            hidden_dims, list), "hidden_dims must be an instance of list"
        assert all(
            isinstance(hidden_dim, int) for hidden_dim in hidden_dims), "hidden_dims must be a list of integers"
        assert isinstance(
            output_dim, int), "output_dim must be an instance of int"
        assert lr > 0, "Learning rate must be greater than 0"
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        assert 0 <= epsilon <= 1, "Epsilon must be between 0 and 1"

        self.output_dim = output_dim
        self.model = self._init_model(input_dim, output_dim, hidden_dims)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def _init_model(self, input_dim: int, output_dim: int, hidden_dims: [int]) -> nn.Module:
        assert isinstance(
            input_dim, int), "input_dim must be an instance of int"
        assert isinstance(
            hidden_dims, list), "hidden_dims must be an instance of list"
        assert all(
            isinstance(hidden_dim, int) for hidden_dim in hidden_dims), "hidden_dims must be a list of integers"
        assert isinstance(
            output_dim, int), "output_dim must be an instance of int"

        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def choose_action(self, state: torch.Tensor) -> int:
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"
        if torch.rand(1).item() < self.epsilon:
            # assuming the last layer is a linear layer with the number of actions
            result = torch.randint(0, self.output_dim, (1,)).item()

            assert 0 <= result < self.output_dim, "result must be between 0 and the number of actions"
            return result
        else:
            with torch.no_grad():
                output = self.model(state).argmax()

                # TODO:
                # Note, multiple actions can have the same value, so we need to randomly choose one
                # But for now, we will just choose the first one for simplicity

                result = output.item()

                assert 0 <= result < self.output_dim, f"result must be between 0 and the number of actions, not {result}"
                return result
        raise ValueError("The function should have returned by now")

    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor) -> None:
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"
        assert isinstance(
            next_state, torch.Tensor), "next_state must be an instance of torch.Tensor"
        assert isinstance(action, int), "action must be an instance of int"
        assert isinstance(reward, float), "reward must be an instance of float"
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"

        self.train(state, action, reward, next_state)

    def save(self, path: str) -> None:
        assert isinstance(path, str), "path must be an instance of str"
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        assert isinstance(path, str), "path must be an instance of str"
        assert os.path.exists(path), "Model not found"

        self.model.load_state_dict(torch.load(path))
        logging.info(f"Model loaded from {path}")

    def train(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor) -> None:
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"
        assert isinstance(
            next_state, torch.Tensor), "next_state must be an instance of torch.Tensor"
        assert isinstance(action, int), "action must be an instance of int"
        assert isinstance(reward, float), "reward must be an instance of float"
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"

        # Get Q-values for the current state
        # TODO: Check if we need to flatten the state
        state = state.flatten()
        next_state = next_state.flatten()

        q_values = self.model(state)

        # Get max Q-values for the next state
        next_q_values = self.model(next_state).max().detach()

        # Calculate the target Q-values
        target = reward + self.gamma * next_q_values

        # Ensure action is a tensor with correct shape
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int64)
        if action.dim() == 1:
            action = action.unsqueeze(1)

        # Gather the Q-values corresponding to the actions taken
        current_q_value = q_values[action]

        # Calculate the loss
        loss = self.criterion(current_q_value, target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"

        self.model.eval()
        with torch.no_grad():
            return self.model(state)
