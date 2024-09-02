from environments.gymnasium.envs.tag.utils.tile_type import TileType
from environments.gymnasium.envs.tag.utils.agents import AgentType

AGENT_TILE_TYPE = {
    AgentType.SEEKER: TileType.SEEKER.value,
    AgentType.HIDER: TileType.HIDER.value,
}
