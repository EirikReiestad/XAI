import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    global device

    def __init__(
        self, n_observations: tuple, n_actions: int, hidden_layers: list[int] = [128]
    ):
        super(DQN, self).__init__()
        self.layers = []
        input_size = int(np.prod(n_observations))
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        self.out_layer = nn.Linear(hidden_layers[-1], n_actions)
        # Do not ask why, but this is necessary instead of appending the out_layer directly
        self.layers.append(self.out_layer)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
