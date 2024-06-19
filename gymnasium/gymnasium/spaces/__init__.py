"""
Tihs modules implements various spaces.

Spaces describe mathematical sets and are used in Gym to specify valid actions and observations.
Every Gym environment must have the attributes ``action_space`` and ``observation_space``.
If, for instance, three possible actions (0,1,2) can be performed in your environment and observations
are vectors in the two-dimensional unit cube, the environment code may contain the following two lines::

    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(0, 1, shape=(2,))

All spaces inherit from the :class:`Space` superclass.
"""

from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete

__all__ = [
    # base space
    # fundamental spaces
    "Box",
    "Discrete",
    # composite spaces
    # utility functions
]
