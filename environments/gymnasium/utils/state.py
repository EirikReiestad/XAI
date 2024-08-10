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

        match self.active:
            case StateType.FULL:
                if self.full is None:
                    raise ValueError("Full state is required.")
            case StateType.PARTIAL:
                if self.partial is None:
                    raise ValueError("Partial state is required.")
            case StateType.RGB:
                if self.rgb is None:
                    raise ValueError("RGB state is required.")
            case _:
                raise ValueError("Invalid state type.")

    def __getattr__(self, name):
        if name == "active_state":
            match self.active:
                case StateType.FULL:
                    return self.full
                case StateType.PARTIAL:
                    return self.partial
                case StateType.RGB:
                    return self.rgb
                case _:
                    raise ValueError("Invalid state type.")
        raise AttributeError(f"'States' object has no attribute '{name}'")
