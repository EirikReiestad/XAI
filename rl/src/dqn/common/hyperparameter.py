from dataclasses import dataclass
import wandb


@dataclass
class DQNHyperparameter:
    """Hyperparameters for the DQN algorithm."""

    def __init__(
        self,
        lr: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        batch_size: int,
        tau: float,
        hidden_layers: list[int],
        conv_layers: list[int],
        memory_size: int,
    ) -> None:
        self.lr: float = lr
        self.gamma: float = gamma
        self.eps_start: float = eps_start
        self.eps_end: float = eps_end
        self.eps_decay: float = eps_decay
        self.batch_size: int = batch_size
        self.tau: float = tau
        self.hidden_layers = hidden_layers
        self.conv_layers = conv_layers
        self.memory_size = memory_size

    def init_sweep(self) -> None:
        return
        self.lr = wandb.config.learning_rate
        self.gamma = wandb.config.gamma
        self.eps_start = wandb.config.eps_start
        self.eps_end = wandb.config.eps_end
        self.eps_decay = wandb.config.eps_decay
        self.batch_size = wandb.config.batch_size
        self.tau = wandb.config.tau
        self.hidden_layers = wandb.config.hidden_layers
        self.conv_layers = wandb.config.conv_layers

    def __str__(self) -> str:
        return (
            f"lr={self.lr}, gamma={self.gamma}, eps_start={self.eps_start}, "
            f"eps_end={self.eps_end}, eps_decay={self.eps_decay}, "
            f"batch_size={self.batch_size}, tau={self.tau},"
            f"hidden_layers={self.hidden_layers}, conv_layers={self.conv_layers}"
        )
