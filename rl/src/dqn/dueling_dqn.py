from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layers):
        super(DuelingDQN, self).__init__()
        # Common feature layer
        self.feature = nn.Sequential(
            nn.Linear(n_observations, hidden_layers[0]),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], n_actions)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean())
