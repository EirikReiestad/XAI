from dataclasses import dataclass


@dataclass
class DQNHyperparameter:
    """Hyperparameters for the DQN algorithm."""

    lr: float
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float
    batch_size: int
    tau: float

    def __str__(self) -> str:
        return (
            f"lr={self.lr}, gamma={self.gamma}, eps_start={self.eps_start}, "
            f"eps_end={self.eps_end}, eps_decay={self.eps_decay}, "
            f"batch_size={self.batch_size}, tau={self.tau}"
        )

    @classmethod
    def default(cls) -> "DQNHyperparameter":
        """Return default hyperparameters."""
        return cls(
            lr=1e-4,
            gamma=0.99,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=1000,
            batch_size=128,
            tau=0.005,
        )
