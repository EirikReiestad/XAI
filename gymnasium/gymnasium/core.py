"""Core API for Environment."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import numpy as np

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class Env(Generic[ObsType, ActType]):
    """
    The main Gymnasium class for implementing Reinforcment Learning Agents environments.

    The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the :meth:`step` and :meth:`reset` functions.
    An environment can be partially or fully observed by single agents. For multi-agent environments, see PettingZoo.

    The main API methods that users of this class need to know are:

    - :meth:`step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions,
      if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
    - :meth:`reset` - Resets the environment to an initial state, required before calling step.
      Returns the first agent observation for an episode and information, i.e. metrics, debug info.
    - :meth:`render` - Renders the environments to help visualise what the agent see, examples modes are "human", "rgb_array", "ansi" for text.
    - :meth:`close` - Closes the environment, important when external software is used, i.e. pygame for rendering, databases

    Environments have additional attributes for users to understand the implementation

    - :attr:`action_space` - The Space object corresponding to valid actions, all valid actions should be contained within the space.
    - :attr:`observation_space` - The Space object corresponding to valid observations, all valid observations should be contained within the space.
    - :attr:`spec` - An environment spec that contains the information used to initialize the environment from :meth:`gymnasium.make`
    - :attr:`metadata` - The metadata of the environment, e.g., `{"render_modes": ["rgb_array", "human"], "render_fps": 30}`. For Jax or Torch, this can be indicated to users with `"jax"=True` or `"torch"=True`.
    - :attr:`np_random` - The random number generator for the environment. This is automatically assigned during
      ``super().reset(seed=seed)`` and when assessing :attr:`np_random`.

    .. seealso:: For modifying or extending environments use the :class:`gymnasium.Wrapper` class

    Note:
        To get reproducible sampling of actions, a seed can be set with ``env.action_space.seed(123)``.

    Note:
        For strict type checking (e.g., mypy or pyright), :class:`Env` is a generic class with two parameterized types: ``ObsType`` and ``ActType``.
        The ``ObsType`` and ``ActType`` are the expected types of the observations and actions used in :meth:`reset` and :meth:`step`.
        The environment's :attr:`observation_space` and :attr:`action_space` should have type ``Space[ObsType]`` and ``Space[ActType]``,
        see a space's implementation to find its parameterized type.
    """

    # Set this in SOME subclasses
    metadata: dict[str, Any] = {"render_modes": []}
    # define render_mode if your environment supports rendering
    render_mode: str | None = None
    spec: EnvSpec | None = None

    # Set these in ALL subclasses
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    # Created
    _np_random: np.random.Generator | None = None
    # will be set to the "invalid" value -1 if the seed of the currently set rng is unknown
    _np_random_seed: int | None = None

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barto Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        : meth: `reset` is called with ``seed = None``, the RNG is not reset.

        Therefore, : meth: `reset` should ( in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of: meth: `reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged: : v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed(optional int): The seed that is used to initialize the environment's PRNG(`np_random`) and
                the read-only attribute `np_random_seed`.
                If the environment does not already have a PRNG and ``seed = None`` (the default option) is passed,
                a seed will be chosen from some source of entropy(e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed = None`` is passed, the PRNG will * not * be reset
                and the env's: attr: `np_random_seed` will * not * be altered.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer * right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options(optional dict): Additional information to specify how the environment is reset(optional,
                depending on the specific environment)

        Returns:
            observation(ObsType): Observation of the initial state. This will be an element of: attr: `observation_space`
                (typically a numpy array) and is analogous to the observation returned by: meth: `step`.
            info(dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by: meth: `step`.
        """
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Compute the render frames as specified by: attr: `render_mode` during the initialization of the environment.

        The environment's: attr: `metadata` render modes(`env.metadata["render_modes"]`) should contain the possible
        ways to implement the render modes. In addition, list versions for most render modes is achieved through
        `gymnasium.make` which automatically applies a wrapper to collect rendered frames.

        Note:
            As the: attr: `render_mode` is known during ``__init__``, the objects used to render the environment state
            should be initialised in ``__init__``.

        By convention, if the: attr: `render_mode` is:

        - None (default): no render is computed.
        - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption.
          This rendering should occur during: meth: `step` and : meth: `render` doesn't need to be called. Returns ``None``.
        - "rgb_array": Return a single frame representing the current state of the environment.
          A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing RGB values for an x-by-y pixel image.
        - "ansi": Return a strings(``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
          for each time step. The text can include newlines and ANSI escape sequences(e.g. for colors).
        - "rgb_array_list" and "ansi_list": List based version of render modes are possible(except Human) through the
          wrapper, : py: class: `gymnasium.wrappers.RenderCollection` that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
          The frames collected are popped after: meth: `render` is called or : meth: `reset`.

        Note:
            Make sure that your class 's: attr: `metadata` ``"render_modes"`` key includes the list of supported modes.

        .. versionchanged: : 0.25.0

            The render function was changed to no longer accept parameters, rather these parameters should be specified
            in the environment initialised, i.e., ``gymnasium.make("CartPole-v1", render_mode="human")``
        """
        raise NotImplementedError

    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise an error.
        """
        pass

    @property
    def np_random_seed(self) -> int:
        """Returns the environment's internal :attr:`_np_random_seed` that if not set will first initialise with a random int as seed.

        If :attr:`np_random_seed` was set directly instead of through :meth:`reset` or :meth:`set_np_random_through_seed`,
        the seed will take the value -1.

        Returns:
            int: the seed of the current `np_random` or -1, if the seed of the rng is unknown
        """
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random_seed

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        """Sets the environment's internal :attr:`_np_random` with the user-provided Generator.

        Since it is generally not possible to extract a seed from an instance of a random number generator,
        this will also set the :attr:`_np_random_seed` to `-1`, which is not valid as input for the creation
        of a numpy rng.
        """
        self._np_random = value
        # Setting a numpy rng with -1 will cause a ValueError
        self._np_random_seed = -1

    def __str__(self):
        """Returns a string of the environment with :attr:`spec` id's if :attr:`spec.

        Returns:
            A string identifying the environment
        """
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any):
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False
