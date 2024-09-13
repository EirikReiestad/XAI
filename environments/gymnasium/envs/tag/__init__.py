from .tag import TagEnv

from . import rewards

assert isinstance(rewards.TAGGED_REWARD, tuple)
assert len(rewards.TAGGED_REWARD) == 2
assert isinstance(rewards.NOT_TAGGED_REWARD, tuple)
assert len(rewards.NOT_TAGGED_REWARD) == 2

assert isinstance(rewards.MOVE_REWARD, tuple)
assert isinstance(rewards.TERMINATED_REWARD, int | float)
assert isinstance(rewards.TRUNCATED_REWARD, tuple)

__all__ = ["TagEnv"]
