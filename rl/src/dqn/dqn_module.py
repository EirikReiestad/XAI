"""
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

from .replay_memory import ReplayMemory, Transition
from .dueling_dqn import DuelingDQN

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNModule():
    global device

    def __init__(self, n_observations: int, n_actions: int, hidden_layers: [int] = [64], seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.n_actions = n_actions
        self.n_observations = n_observations

        self.batch_size = 512  # The number of transitions sampled from the replay buffer
        self.gamma = 0.999  # The discount factor
        self.eps_start = 0.9  # The starting value of epsilon
        self.eps_end = 0.05  # The final value of epsilon
        # The rate of exponential decay of epsilon, higher means a slower decay
        self.eps_decay = 10000
        self.tau = 0.005  # The update rate of the target network
        self.lr = 1e-4  # The learning rate of the ``AdamW`` optimizer

        self.policy_net = DuelingDQN(n_observations, n_actions,
                                     hidden_layers).to(device)
        self.target_net = DuelingDQN(n_observations, n_actions,
                                     hidden_layers).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

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

        if state.shape != (1, self.n_observations):
            raise ValueError(
                f"Expected state to have shape (1, {self.n_observations}), but got {state.shape}")

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        """
        This function samples a batch from the replay memory and performs a single optimization step
        """
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)

        # NOTE: Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(non_final_next_states).max(1)[
                1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).gather(1, next_actions).squeeze(1)

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor, reward: float, terminated: bool, truncated: bool) -> (bool, torch.Tensor):
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

        if state.shape != (1, self.n_observations):
            raise ValueError(
                f"Expected state to have shape (1, {self.n_observations}), but got {state.shape}")

        if observation.shape != (1, self.n_observations):
            raise ValueError(
                f"Expected observation to have shape (1, {self.n_observations}), but got {observation.shape}")

        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = observation.clone().detach()

        if next_state is not None and next_state.shape != (1, self.n_observations):
            raise ValueError(
                f"Expected next_state to have shape (1, {self.n_observations}), but got {next_state.shape}")

        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        return done, state
