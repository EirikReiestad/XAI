import numpy as np
from gymnasium import spaces
from .world_object import WorldObject
from .constants import Direction
from .mission import Mission, MissionSpace
from ..utils.misc import PropertyAlias


class Agent:
    def __init__(
        self,
        index: int,
        mission_space: MissionSpace = MissionSpace.from_string("maximize reward"),
        view_size: int = 7,
    ) -> None:
        self._index = index
        self._state: AgentState = AgentState()
        self._mission: Mission = None

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(view_size, view_size, WorldObject.dim),
                    dtype=np.uint8,
                ),
                "direction": spaces.Discrete(len(Direction)),
                "mission": mission_space,
            }
        )

        self.action_space = spaces.Discrete(len(Action))

    # AgentState Properties
    color = PropertyAlias("state", "color", doc="Alias for :attr:`AgentState.color`.")
    dir = PropertyAlias("state", "dir", doc="Alias for :attr:`AgentState.dir`.")
    pos = PropertyAlias("state", "pos", doc="Alias for :attr:`AgentState.pos`.")
    terminated = PropertyAlias(
        "state", "terminated", doc="Alias for :attr:`AgentState.terminated`."
    )
    carrying = PropertyAlias(
        "state", "carrying", doc="Alias for :attr:`AgentState.carrying`."
    )


class AgentState:
    def ___init__(self, agents: int | None = None) -> None:
        self._agents = agents
