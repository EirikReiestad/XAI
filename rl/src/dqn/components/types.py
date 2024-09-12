from collections import namedtuple
import torch
from dataclasses import dataclass
from typing import NamedTuple

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class Rollout(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    value: torch.Tensor
    log_prob: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor
    next_value: torch.Tensor


@dataclass
class RolloutReturn:
    """Return type for the rollout method."""

    def __init__(
        self,
    ):
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.next_states: list[torch.Tensor] = []
        self.terminals: list[torch.Tensor] = []
        self.truncated: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.advantages: list[torch.Tensor] = []
        self.returns: list[torch.Tensor] = []
        self.next_value: list[torch.Tensor] = []

    def append(self, rollout: Rollout):
        self.states.append(rollout.state)
        self.actions.append(rollout.action)
        self.rewards.append(rollout.reward)
        self.next_states.append(rollout.next_state)
        self.terminals.append(rollout.terminated)
        self.truncated.append(rollout.truncated)
        self.values.append(rollout.value)
        self.log_probs.append(rollout.log_prob)
        self.advantages.append(rollout.advantage)
        self.returns.append(rollout.returns)
        self.next_value.append(rollout.next_value)
