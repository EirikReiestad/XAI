from dataclasses import dataclass

import torch

from .transition import Transition


@dataclass
class Batch:
    states: list[torch.Tensor]
    actions: list[torch.Tensor]
    observations: list[torch.Tensor]
    rewards: list[torch.Tensor]
    terminated: list[bool]
    truncated: list[bool]

    def append(self, transition: Transition) -> None:
        """Append a transition to the batch."""
        self.states.append(transition.state)
        self.actions.append(transition.action)
        self.observations.append(transition.observation)
        self.rewards.append(transition.reward)
        self.terminated.append(transition.terminated)
        self.truncated.append(transition.truncated)
