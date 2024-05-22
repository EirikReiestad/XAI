import numpy as np
import random


class QLearning:
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    def __init__(self, action_space: int):
        self.action_space = action_space
        self.q_table = {}

    def choose_action(self, state: int) -> int:

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))
        else:
            if self.q_table.get(state) is None:
                self.q_table[state] = np.zeros(self.action_space)
                return random.choice(range(self.action_space))
            return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        if self.q_table.get(state) is None:
            self.q_table[state] = np.zeros(self.action_space)

        if self.q_table.get(next_state) is None:
            self.q_table[next_state] = np.zeros(self.action_space)

        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        self.q_table = np.load(path, allow_pickle=True).item()
