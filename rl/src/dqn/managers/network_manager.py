import torch.nn as nn
import torch

from rl.src.common import ConvLayer
from rl.src.dqn.q_network import QNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NetworkManager:
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        n_actions: int,
        conv_layers: list[ConvLayer] | None,
        hidden_layers: list[int],
        dueling: bool = False,
    ) -> None:
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.conv_layers = conv_layers
        self.hidden_layers = hidden_layers
        self.dueling = dueling

    def initialize(
        self,
    ) -> tuple[nn.Module, nn.Module]:
        policy_net = QNetwork(
            self.observation_shape,
            self.n_actions,
            self.hidden_layers,
            self.conv_layers,
            self.dueling,
        ).to(device)
        target_net = QNetwork(
            self.observation_shape,
            self.n_actions,
            self.hidden_layers,
            self.conv_layers,
            self.dueling,
        ).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        return policy_net, target_net
