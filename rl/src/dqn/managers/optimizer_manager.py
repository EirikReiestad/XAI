from torch.optim.adamw import AdamW
import torch.nn as nn


class OptimizerManager:
    def __init__(self, policy_net: nn.Module, lr: float):
        self.policy_net = policy_net
        self.lr = lr

    def initialize(self) -> None:
        self.optimizer = AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
