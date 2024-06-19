"""
Root `__init__` of the gymnasium module setting `__all__` of gymnasium modules.
"""

from gymnasium.core import (
    Env,
)
from gymnasium.spaces.space import Space

# necessary for `envs.__init__` which registers all gymnasium environments and loads plugins
from gymnasium import envs
from gymnasium import spaces

__all__ = [
    # core classes
    "Env",
    "Spaces",
    # registration
    # module folders
    "envs",
    "spaces",
]
