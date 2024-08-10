import os
import logging
from dataclasses import dataclass


@dataclass
class DQNHyperparameter:
    lr = 0.01
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    batch_size = 128
    tau = 0.001

    def __init__(
        self,
        lr: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        batch_size: float,
        tau: float,
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.tau = tau

    def __str__(self):
        return f"{self.lr},{self.gamma},{self.eps_start},{self.eps_end},{self.eps_decay},{self.batch_size},{self.tau}"
