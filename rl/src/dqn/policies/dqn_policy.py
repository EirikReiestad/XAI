from gymnasium import spaces
from .q_network import QNetwork


class DQNPolicy:
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        dueling: bool = False,
    ):
        policy_net_hidden_layers = [128, 128]
        target_net_hidden_layers = [128, 128]
        self._policy_net: QNetwork = QNetwork(
            observation_space, action_space, policy_net_hidden_layers, dueling
        )
        self._target_net: QNetwork = QNetwork(
            observation_space, action_space, target_net_hidden_layers, dueling
        )

    @property
    def policy_net(self) -> QNetwork:
        return self._policy_net

    @property
    def target_net(self) -> QNetwork:
        return self._target_net
