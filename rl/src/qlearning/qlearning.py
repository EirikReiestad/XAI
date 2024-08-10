import random
from typing import Dict

import numpy as np


class QLearning:
    """Q-Learning algorithm implementation."""

    def __init__(
        self,
        action_space: int,
        alpha: float = 0.1,
        gamma: float = 0.6,
        epsilon: float = 0.1,
    ):
        """
        Initialize Q-Learning with the given parameters.

        Parameters:
            action_space (int): Number of possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
        """
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[int, np.ndarray] = {}

    def choose_action(self, state: int) -> int:
        """
        Choose an action based on the current state.

        Parameters:
            state (int): The current state.

        Returns:
            int: The chosen action.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
            return random.randint(0, self.action_space - 1)

        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update the Q-table based on the state-action-reward-next_state tuple.

        Parameters:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space)

        q_update = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            q_update - self.q_table[state][action]
        )

    def save(self, path: str) -> None:
        """
        Save the Q-table to a file.

        Parameters:
            path (str): File path to save the Q-table.
        """
        np.save(path, np.array(list(self.q_table.items())))

    def load(self, path: str) -> None:
        """
        Load the Q-table from a file.

        Parameters:
            path (str): File path to load the Q-table from.
        """
        self.q_table = np.load(path, allow_pickle=True).item()
