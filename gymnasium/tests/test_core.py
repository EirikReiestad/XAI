"""Checks that the core Gymnasium API is implemented as expected."""

from __future__ import annotations

from typing import Any

import pytest

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box
from gymnasium.utils.seeding import np_random


class ExampleEnv(Env):
    """Example testing environment."""

    def __init__(self):
        """Constructor for example environment."""
        self.observation_space = Box(0, 1)
        self.action_space = Box(0, 1)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the environment."""
        return 0, 0, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[ObsType, dict]:
        """Resets the environment."""
        super().reset(seed=seed, options=options)
        return 0, {}


@pytest.fixture
def example_env():
    return ExampleEnv()


def test_example_env(example_env):
    """Tests a gymnasium environment."""

    assert example_env.metadata == {"render_modes": []}
    assert example_env.render_mode is None
    assert example_env.spec is None
    # pyright: ignore [reportPrivateUsage]
    assert example_env._np_random is None


class TestRandomSeeding:
    @staticmethod
    def test_nonempty_seed_retrieved_when_not_set(example_env):
        assert example_env.np_random_seed is not None
        assert isinstance(example_env.np_random_seed, int)

    @staticmethod
    def test_seed_set_at_reset_and_retrieved(example_env):
        seed = 42
        example_env.reset(seed=seed)
        assert example_env.np_random_seed == seed
        # resetting with seed=None means seed remains the same
        example_env.reset(seed=None)
        assert example_env.np_random_seed == seed

    @staticmethod
    def test_seed_cannot_be_set_directly(example_env):
        with pytest.raises(AttributeError):
            example_env.np_random_seed = 42

    @staticmethod
    def test_negative_seed_retrieved_when_seed_unknown(example_env):
        rng, _ = np_random()
        example_env.np_random = rng
        # seed is unknown
        assert example_env.np_random_seed == -1
