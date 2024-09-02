from dataclasses import dataclass

import torch


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    observation: torch.Tensor
    reward: torch.Tensor
    terminated: bool
    truncated: bool
