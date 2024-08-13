from dataclasses import dataclass

import numpy as np

from environments.gymnasium.utils.enums import StateType


@dataclass
class State:
    full: np.ndarray
    partial: np.ndarray
    rgb: np.ndarray
    active: StateType

    def __post_init__(self):
        if not isinstance(self.active, StateType):
            raise ValueError("Invalid state type.")

        # Ensure the required state is not None based on the active state type.
        if self.active == StateType.FULL and self.full is None:
            raise ValueError("Full state is required.")
        elif self.active == StateType.PARTIAL and self.partial is None:
            raise ValueError("Partial state is required.")
        elif self.active == StateType.RGB and self.rgb is None:
            raise ValueError("RGB state is required.")

    @property
    def active_state(self) -> np.ndarray:
        """Returns the active state based on the `active` type."""
        match self.active:
            case StateType.FULL:
                return self.full
            case StateType.PARTIAL:
                return self.partial
            case StateType.RGB:
                return self.rgb
            case _:
                raise ValueError("Invalid state type.")
