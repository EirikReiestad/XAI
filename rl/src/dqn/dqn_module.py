"""DQN Module.

This module contains the DQN agent that interacts with the environment.
"""

import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW

from rl import settings
from rl.src.common import ConvLayer
from rl.src.dqn.dqn import DQN
from rl.src.dqn.dueling_dqn import DuelingDQN
from rl.src.dqn.replay_memory import ReplayMemory, Transition
from rl.src.hyperparameters.dqn_hyperparameter import DQNHyperparameter

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNModule:
    """DQN Module for managing the agent, including training and evaluation."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        n_actions: int,
        hidden_layers: list[int] = [128],
        conv_layers: list[ConvLayer] | None = None,
        path: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the DQN agent.

        Args:
            observation_shape (tuple[int, int, int]): Shape of the input observation.
            n_actions (int): Number of actions in the environment.
            hidden_layers (list[int]): Sizes of hidden layers.
            conv_layers (list[ConvLayer] | None): Convolutional layer configurations.
            path (str | None): Path to save/load the model.
            seed (int | None): Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.path = path
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.hp = DQNHyperparameter(
            lr=settings.LR,
            gamma=settings.GAMMA,
            eps_start=settings.EPS_START,
            eps_end=settings.EPS_END,
            eps_decay=settings.EPS_DECAY,
            batch_size=settings.BATCH_SIZE,
            tau=settings.TAU,
        )

        self.policy_net, self.target_net = self._initialize_networks(
            conv_layers, hidden_layers
        )
        self.optimizer = AdamW(
            self.policy_net.parameters(), lr=self.hp.lr, amsgrad=True
        )
        self.memory = ReplayMemory(settings.REPLAY_MEMORY_SIZE)

        if self.path and os.path.exists(self.path):
            self.load()

        self.update_interval = 1
        self.step_count = 0
        self.steps_done = 0

    def _initialize_networks(
        self, conv_layers: list[ConvLayer] | None, hidden_layers: list[int]
    ) -> tuple[nn.Module, nn.Module]:
        """Initialize the policy and target networks.

        Args:
            conv_layers (list[ConvLayer] | None): Convolutional layer configurations.
            hidden_layers (list[int]): Sizes of hidden layers.

        Returns:
            tuple[nn.Module, nn.Module]: Initialized policy and target networks.
        """
        if settings.DUELING_DQN:
            policy_net = DuelingDQN(
                self.observation_shape, self.n_actions, hidden_layers, conv_layers
            ).to(device)
            target_net = DuelingDQN(
                self.observation_shape, self.n_actions, hidden_layers, conv_layers
            ).to(device)
        else:
            policy_net = DQN(self.observation_shape, self.n_actions, hidden_layers).to(
                device
            )
            target_net = DQN(self.observation_shape, self.n_actions, hidden_layers).to(
                device
            )
        return policy_net, target_net

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select an action based on the current state using an epsilon-greedy policy.

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: The selected action.
        """
        if state.shape != self.observation_shape:
            raise ValueError(
                f"Expected state shape {self.observation_shape}, but got {state.shape}"
            )

        eps_threshold = self.hp.eps_end + (
            self.hp.eps_start - self.hp.eps_end
        ) * math.exp(-self.steps_done / self.hp.eps_decay)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]], device=device, dtype=torch.long
            )

    def _optimize_model(self) -> None:
        """Perform one optimization step on the policy network."""
        if len(self.memory) < self.hp.batch_size:
            return

        self.step_count += 1
        if self.step_count % self.update_interval != 0:
            return

        transitions = self.memory.sample(self.hp.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state], device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat([s for s in batch.state])
        action_batch = torch.cat([a for a in batch.action])
        reward_batch = torch.cat([r for r in batch.reward])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.hp.batch_size, device=device)

        if settings.DOUBLE_DQN:
            with torch.no_grad():
                next_actions = (
                    self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                )
                next_state_values[non_final_mask] = (
                    self.target_net(non_final_next_states)
                    .gather(1, next_actions)
                    .squeeze(1)
                )
        else:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states
                ).max(1)[0]

        expected_state_action_values = next_state_values * self.hp.gamma + reward_batch

        loss = nn.SmoothL1Loss()(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> tuple[bool, torch.Tensor]:
        """Store transition and optimize the model.

        Args:
            state (torch.Tensor): Current state of the environment.
            action (torch.Tensor): Action taken in the current state.
            observation (torch.Tensor): Observation from the environment.
            reward (float): Reward from the environment.
            terminated (bool): Whether the environment is terminated.
            truncated (bool): Whether the environment is truncated.

        Returns:
            tuple[bool, torch.Tensor]: Whether the environment is done, and the next state.
        """
        if state.shape != self.observation_shape:
            raise ValueError(
                f"Expected state shape {self.observation_shape}, but got {state.shape}"
            )

        if observation.shape != self.observation_shape:
            raise ValueError(
                f"Expected observation shape {self.observation_shape}, but got {observation.shape}"
            )

        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)
        done = terminated or truncated

        if done:
            return done, state

        next_state = observation.clone().detach()
        if next_state is not None and next_state.shape != self.observation_shape:
            raise ValueError(
                f"Expected next_state shape {self.observation_shape}, but got {next_state.shape}"
            )

        self.memory.push(state, action, next_state, reward_tensor)
        state = next_state

        self._optimize_model()

        self._soft_update_target_net()

        return done, state

    def _soft_update_target_net(self) -> None:
        """Soft update the target network parameters."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.hp.tau + target_net_state_dict[key] * (1 - self.hp.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self) -> None:
        """Save the policy network to the specified path."""
        if self.path is None:
            raise ValueError("Model path is not specified")
        torch.save(self.policy_net.state_dict(), self.path)

    def load(self) -> None:
        """Load the policy network from the specified path."""
        if self.path is None:
            raise ValueError("Model path is not specified")

        if not os.path.exists(self.path):
            logging.warning(f"Model not found at {self.path}")
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return

        self.policy_net.load_state_dict(torch.load(self.path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()
