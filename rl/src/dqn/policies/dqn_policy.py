from gymnasium import spaces
from .q_network import QNetwork


class DQNPolicy:
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        dueling: bool = False,
    ):
        hidden_layers = [128, 128]
        conv_layers = []

        self._policy_net: QNetwork = QNetwork(
            observation_space,
            action_space,
            hidden_layers,
            conv_layers,
            dueling,
        )
        self._target_net: QNetwork = QNetwork(
            observation_space,
            action_space,
            hidden_layers,
            conv_layers,
            dueling,
        )

    @property
    def policy_net(self) -> QNetwork:
        return self._policy_net

    @property
    def target_net(self) -> QNetwork:
        return self._target_net
