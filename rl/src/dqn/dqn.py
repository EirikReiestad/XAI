import torch
import torch.nn as nn
import os
import logging
import random
from collections import deque


class DQN:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: [int],
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = .995,
                 epsilon_min: float = .01,
                 replay_buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_frequency: int = 10):

        # Validate initialization parameters
        if not isinstance(input_dim, int):
            raise TypeError("input_dim must be an instance of int")
        if not isinstance(hidden_dims, list) or not all(isinstance(hidden_dim, int) for hidden_dim in hidden_dims):
            raise TypeError("hidden_dims must be a list of integers")
        if not isinstance(output_dim, int):
            raise TypeError("output_dim must be an instance of int")
        if lr <= 0:
            raise ValueError("Learning rate must be greater than 0")
        if not (0 <= gamma <= 1):
            raise ValueError("Gamma must be between 0 and 1")
        if not (0 <= epsilon <= 1):
            raise ValueError("Epsilon must be between 0 and 1")
        if not (0 <= epsilon_decay <= 1):
            raise ValueError("Epsilon decay must be between 0 and 1")
        if not (0 <= epsilon_min <= 1):
            raise ValueError("Epsilon min must be between 0 and 1")
        if not isinstance(replay_buffer_size, int) or replay_buffer_size <= 0:
            raise ValueError("Replay buffer size must be a positive integer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")

        self.output_dim = output_dim
        self.model = self._init_model(input_dim, output_dim, hidden_dims)
        self.target_model = self._init_model(
            input_dim, output_dim, hidden_dims)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set target model to evaluation mode
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        # Set frequency for updating target model
        self.target_update_frequency = target_update_frequency
        self.train_step = 0  # Initialize training step counter

    def _init_model(self, input_dim: int, output_dim: int, hidden_dims: [int]) -> nn.Module:
        """Initialize the neural network architecture"""
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Add dropout layer for regularization
            layers.append(nn.Dropout(0.5))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))  # Output layer
        return nn.Sequential(*layers)

    def _update_epsilon(self):
        """Decay epsilon to reduce exploration over time"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def choose_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy"""
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be an instance of torch.Tensor")

        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.output_dim - 1)
        with torch.no_grad():
            # Exploit: action with the highest Q-value
            output = self.model(state).argmax().item()
            return output

    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        """
        Store transition in replay buffer and perform training if buffer is sufficiently populated.

        Parameters
            state: torch.Tensor
            action: int
            reward: float
            next_state: torch.Tensor
            done: bool
                flag to indicate if the episode is done
        """
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor")
        if not isinstance(action, int):
            raise TypeError("action must be an int")
        if not isinstance(reward, (int, float)):
            raise TypeError("reward must be a number")
        if not isinstance(next_state, torch.Tensor):
            raise TypeError("next_state must be a torch.Tensor")
        if not isinstance(done, (int, bool)):
            raise TypeError("done must be an int or bool")

        self.replay_buffer.append(
            (state, action, reward, next_state, done))
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples for training

        # Sample a batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([s.clone().detach() for s in states])
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack([ns.clone().detach() for ns in next_states])
        dones = torch.tensor(dones, dtype=torch.bool)

        self.train(states, actions, rewards,
                   next_states, dones)  # Train the model
        self._update_epsilon()  # Decay epsilon

    def save(self, path: str) -> None:
        if not isinstance(path, str):
            raise TypeError("path must be an instance of str")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model state from a file"""
        if not isinstance(path, str):
            raise TypeError("path must be an instance of str")
        if not os.path.exists(path):
            raise FileNotFoundError("Model not found")

        self.model.load_state_dict(torch.load(path))
        logging.info(f"Model loaded from {path}")

    def train(self, states: [torch.Tensor], actions: [int], rewards: [float], next_states: [torch.Tensor], dones: [bool]) -> None:
        """Perform a training step"""
        if not all(isinstance(s, torch.Tensor) for s in states):
            raise TypeError("All states must be torch.Tensor")
        if not all(isinstance(ns, torch.Tensor) for ns in next_states):
            raise TypeError("All next_states must be torch.Tensor")
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if not isinstance(rewards, torch.Tensor):
            raise TypeError("rewards must be a torch.Tensor")
        if not isinstance(dones, torch.Tensor):
            raise TypeError("dones must be a torch.Tensor")

        # Get the Q-values for the current state and action
        q_values = self.model(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Compute the target Q-values using the target model
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * ~dones

        # Compute the loss between the current Q-values and the target Q-values
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backpropagate loss
        self.optimizer.step()  # Update model parameters

        self.train_step += 1
        if self.train_step % self.target_update_frequency == 0:
            # Periodically update the target model to match the main model
            self.target_model.load_state_dict(
                self.model.state_dict())  # Update target network

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Predict Q-values for a given state"""
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be an instance of torch.Tensor")

        self.model.eval()
        with torch.no_grad():
            return self.model(state)
