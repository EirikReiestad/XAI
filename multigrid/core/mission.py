from typing import Callable, Iterable, Sequence

import numpy as np
from gymnasium import spaces


class Mission(np.ndarray):
    def __new__(cls, string: str, index: Iterable[int] | None = None):
        mission = np.array(0 if index is None else index)
        mission = mission.view(cls)
        mission.string = string
        return mission.view(cls)

    def __array_finalize__(self, mission):
        if mission is None:
            return
        self.string = getattr(mission, "string", None)

    def __str__(self):
        return self.string

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.string}")'

    def __eq__(self, value: object) -> bool:
        return self.string == str(value)

    def __hash__(self) -> int:
        return hash(self.string)


class MissionSpace(spaces.MultiDiscrete):
    def __init__(
        self,
        mission_func: Callable[..., str],
        ordered_placeholder: Sequence[Sequence[str]] = [],
        seed: int | np.random.Generator | None = None,
    ):
        self._mission_func = mission_func
        self._arg_groups = ordered_placeholder
        nvec = tuple(len(group) for group in self._arg_groups)
        super().__init__(nvec=nvec if nvec else (1,))

    def __repr__(self) -> str:
        """
        Get a string representation of this space.
        """
        if self._arg_groups:
            return f"MissionSpace({self._mission_func.__name__}, {self._arg_groups})"
        return f"MissionSpace('{self.mission_func()}')"

    def get(self, idx: Iterable[int]) -> Mission:
        """
        Get the mission string corresponding to the given index.

        Parameters
        ----------
        idx : Iterable[int]
            Index of desired argument in each argument group
        """
        if self._arg_groups:
            args = (self._arg_groups[axis][index] for axis, index in enumerate(idx))
            return Mission(string=self.mission_func(*args), index=idx)
        return Mission(string=self.mission_func())

    def sample(self) -> Mission:
        """
        Sample a random mission string.
        """
        idx = super().sample()
        return self.get(idx)

    def contains(self, x: Any) -> bool:
        """
        Check if an item is a valid member of this mission space.

        Parameters
        ----------
        x : Any
            Item to check
        """
        for idx in np.ndindex(tuple(self.nvec)):
            if self.get(idx) == x:
                return True
        return False

    @staticmethod
    def from_string(string: str) -> MissionSpace:
        """
        Create a mission space containing a single mission string.

        Parameters
        ----------
        string : str
            Mission string
        """
        return MissionSpace(mission_func=lambda: string)
