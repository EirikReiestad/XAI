from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layers):
        super(DuelingDQN, self).__init__()

        # Define the common feature layers
        layers = []
        input_dim = n_observations
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.feature = nn.Sequential(*layers)

        # Define the value stream layers
        value_layers = []
        input_dim = hidden_layers[-1]  # Last hidden layer size
        for hidden_dim in hidden_layers:
            value_layers.append(nn.Linear(input_dim, hidden_dim))
            value_layers.append(nn.ReLU())
            input_dim = hidden_dim
        # Output layer for value stream
        value_layers.append(nn.Linear(hidden_layers[-1], 1))
        self.value_stream = nn.Sequential(*value_layers)

        # Define the advantage stream layers
        advantage_layers = []
        input_dim = hidden_layers[-1]  # Last hidden layer size
        for hidden_dim in hidden_layers:
            advantage_layers.append(nn.Linear(input_dim, hidden_dim))
            advantage_layers.append(nn.ReLU())
            input_dim = hidden_dim
        # Output layer for advantage stream
        advantage_layers.append(nn.Linear(hidden_layers[-1], n_actions))
        self.advantage_stream = nn.Sequential(*advantage_layers)

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
