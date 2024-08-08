"""DQN Module.

This module contains the DQN agent that interacts with the environment.
"""

import os
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
import random
import math
import logging

from rl.src.dqn.replay_memory import ReplayMemory, Transition
from rl.src.dqn.dueling_dqn import DuelingDQN
from rl.src.hyperparameters.dqn_hyperparameter import DQNHyperparameter

from rl.src import settings

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNModule:
    global device

    def __init__(
        self,
        observation_shape: tuple,
        n_actions: int,
        hidden_layers: list[int] = [64],
        path: str | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.path = path

        self.n_actions = n_actions
        self.observation_shape = observation_shape

        if len(self.observation_shape) != 4:
            logging.warning(
                f"Using image as the input resulting in 4 dimensions (batch size, color channels, width, height), got {len(self.observation_shape)}"
            )
            raise ValueError(
                f"Expected observation_shape to have length 3, but got {len(self.observation_shape)}"
            )

        self.hp = DQNHyperparameter()
        if self.path is not None:
            hyperparameter_path = self.path + ".hyper"
            self.hp.load(hyperparameter_path)

        self.policy_net = DuelingDQN(observation_shape, n_actions, hidden_layers).to(
            device
        )
        self.target_net = DuelingDQN(observation_shape, n_actions, hidden_layers).to(
            device
        )

        self.optimizer = AdamW(
            self.policy_net.parameters(), lr=self.hp.lr, amsgrad=True
        )
        self.memory = ReplayMemory(settings.REPLAY_MEMORY_SIZE)

        if self.path is not None and os.path.exists(self.path):
            self.load()

        self.update_interval = 1
        self.step_count = 0

        self.steps_done = 0

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            state (torch.Tensor): The current state of the environment

        Returns:
            torch.Tensor: The action to take in the environment
        """

        if state.shape != self.observation_shape:
            raise ValueError(
                f"Expected state to have shape {self.observation_shape}, but got {state.shape}"
            )

        sample = random.random()
        eps_threshold = self.hp.eps_end + (
            self.hp.eps_start - self.hp.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.hp.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]], device=device, dtype=torch.long
            )

    def optimize_model(self):
        """
        This function samples a batch from the replay memory and performs a single optimization step
        """
        if len(self.memory) < self.hp.batch_size:
            return

        self.step_count += 1
        if self.step_count % self.update_interval != 0:
            return

        transitions = self.memory.sample(self.hp.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # Concatenate states, actions, and rewards
        state_batch = torch.cat([s for s in batch.state])
        action_batch = torch.cat([a for a in batch.action])
        reward_batch = torch.cat([r for r in batch.reward])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.hp.batch_size, device=device)

        # NOTE: Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states)
                .gather(1, next_actions)
                .squeeze(1)
            )

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.hp.gamma
        ) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.save()

    def train(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> tuple[bool, torch.Tensor]:
        """
        Parameters:
            state (torch.Tensor): The current state of the environment
            action (torch.Tensor): The action taken in the current state
            observation (torch.Tensor): The observation from the environment
            reward (float): The reward from the environment
            terminated (bool): Whether the environment is terminated
            truncated (bool): Whether the environment is truncated

        Returns:
            bool: Whether the environment is done
            state (torch.Tensor): The next state of the environment
        """

        if state.shape != self.observation_shape:
            raise ValueError(
                f"Expected state to have shape {self.observation_shape}, but got {state.shape}"
            )

        if observation.shape != self.observation_shape:
            raise ValueError(
                f"Expected observation to have shape {self.observation_shape}, but got {observation.shape}"
            )

        reward = torch.tensor([reward], device=device, dtype=torch.float).item()
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = observation.clone().detach()

        if next_state is not None and next_state.shape != self.observation_shape:
            raise ValueError(
                f"Expected next_state to have shape (1, {self.observation_shape}), but got {next_state.shape}"
            )

        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)

        if next_state is None:
            return done, state

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.hp.tau + target_net_state_dict[key] * (1 - self.hp.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        return done, state

    def save(self):
        """
        Save the model and memory to the specified path
        Parameters:
            path (str): The path to save the model and memory
        """
        self._save_model()

    def _save_model(self):
        """
        Save the model to the specified path
        Parameters:
            path (str): The path to save the model
        """
        if self.path is None:
            raise ValueError("Model path is not specified")

        torch.save(self.policy_net.state_dict(), self.path)

    def load(self):
        """
        Load the model and memory from the specified path
        Parameters:
            path (str): The path to load the model and memory
        """
        self._load_model()

    def _load_model(self):
        """
        Load the model from the specified path
        Parameters:
            path (str): The path to load the model
        """
        if self.path is None:
            raise ValueError("Model path is not specified")

        if not os.path.exists(self.path):
            logging.warning(f"Model not found at {self.path}")
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.load_state_dict(torch.load(self.path, weights_only=True))
        self.target_net.load_state_dict(torch.load(self.path, weights_only=True))
        self.policy_net.eval()
        self.target_net.eval()
