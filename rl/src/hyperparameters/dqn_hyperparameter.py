import os
import logging
from dataclasses import dataclass


@dataclass
class DQNHyperparameter:
    lr = 0.1
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    batch_size = 128
    tau = 0.001

    def save(self, path):
        with open(path, 'w') as f:
            f.write(str(self))

    def load(self, path):
        if path is None:
            return
        if not os.path.exists(path):
            logging.warning(f"Hyperparameters not found at {path}")
            self.save(path)

        with open(path, 'r') as f:
            data = f.read()
            lr, gamma, eps_start, eps_end, eps_decay, batch_size, tau = data.split(
                ',')

    def __str__(self):
        return f"{self.lr},{self.gamma},{self.eps_start},{self.eps_end},{self.eps_decay},{self.batch_size},{self.tau}"
