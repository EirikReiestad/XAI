import enum


@enum.unique
class AgentType(enum.Enum):
    """Enum for the type of agent."""

    SEEKER = 0
    HIDER = 1
