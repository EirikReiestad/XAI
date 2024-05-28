import torch


class NeuralNetwork:
    def __init__(self, model: torch.nn.Module, lr: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1):
        assert isinstance(
            model, torch.nn.Module), "model must be an instance of torch.nn.Module"
        # Last layer should be a linear layer with the number of actions
        assert isinstance(
            model.fc, torch.nn.Linear), "Last layer must be a linear layer"
        assert model.fc.out_features > 0, "Number of actions must be greater than 0"
        assert lr > 0, "Learning rate must be greater than 0"
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        assert 0 <= epsilon <= 1, "Epsilon must be between 0 and 1"

        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def choose_action(self, state: torch.Tensor) -> int:
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"
        if torch.rand(1).item() < self.epsilon:
            # assuming the last layer is a linear layer with the number of actions
            return torch.randint(0, self.model.fc.out_features, (1,)).item()
        else:
            with torch.no_grad():
                return self.model(state).argmax().item()

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
        self.model.load_state_dict(torch.load(path))

    def train(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor) -> None:
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"
        assert isinstance(
            next_state, torch.Tensor), "next_state must be an instance of torch.Tensor"
        assert isinstance(action, int), "action must be an instance of int"
        assert isinstance(reward, float), "reward must be an instance of float"
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"

        self.model.train()
        q_values = self.model(state)
        next_q_values = self.model(next_state).max(1)[0].detach()
        target = reward + self.gamma * next_q_values
        loss = self.criterion(q_values.gather(
            1, torch.tensor([[action]])), target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            state, torch.Tensor), "state must be an instance of torch.Tensor"

        self.model.eval()
        with torch.no_grad():
            return self.model(state)
