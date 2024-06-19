"""Functions for registering environments within gymnasium using public functions ``make``, ``register`` and ``spec``."""

from __future__ import annotations
import contextlib
import copy
import dataclasses
import difflib
import importlib
import importlib.util
import json
import re
import sys

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

import gymnasium as gym
from gymnasium import Env, error

from typing import Protocol

__all__ = [
    "EnvSpec",
    # functions
]


class EnvCreator(Protocol):
    """Function type expected for an environment"""

    def __call__(self, *args, **kwargs) -> Env:
        ...


@dataclass
class EnvSpec:
    """A specification for creating environments with :meth:`gymnasium.make`.

    * **id**: The string used to create the environment with :meth:`gymnasium.make`
    * **entry_point**: A string for the environment location, ``(import path):(environment name)`` or a function that creates the environment.
    * **reward_threshold**: The reward threshold for completing the environment.
    * **nondeterministic**: If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
    * **max_episode_steps**: The max number of steps that the environment can take before truncation
    * **order_enforce**: If to enforce the order of :meth:`gymnasium.Env.reset` before :meth:`gymnasium.Env.step` and :meth:`gymnasium.Env.render` functions
    * **disable_env_checker**: If to disable the environment checker wrapper in :meth:`gymnasium.make`, by default False (runs the environment checker)
    * **kwargs**: Additional keyword arguments passed to the environment during initialisation
    * **additional_wrappers**: A tuple of additional wrappers applied to the environment (WrapperSpec)
    * **vector_entry_point**: The location of the vectorized environment to create from

    Changelogs:
        v1.0.0 - Autoreset attribute removed
    """

    id: str
    entry_point: EnvCreator | str | None = field(default=None)

    # Environment attributes
    reward_threshold: float | None = field(default=None)
    nondeterministic: bool = field(default=False)

    # Wrappers

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

    # post-init attributes
    namespace: str | None = field(init=False)
    name: str = field(init=False)
    version: str | None = field(init=False)

    def pprint(
        self,
        disable_print: bool = False,
        include_entry_points: bool = False,
        print_all: bool = False,
    ) -> str | None:
        """Pretty prints the environment spec.

        Args:
            disable_print: If to disable print and return the output
            include_entry_points: If to include the entry_points in the output
            print_all: If to print all information, including variables with default values

        Returns:
            If ``disable_print is True`` a string otherwise ``None``
        """
        output = f"id={self.id}"
        if print_all or include_entry_points:
            output += f"\nentry_point={self.entry_point}"

        if print_all or self.reward_threshold is not None:
            output += f"\nreward_threshold={self.reward_threshold}"
        if print_all or self.nondeterministic is not False:
            output += f"\nnondeterministic={self.nondeterministic}"

        if print_all or self.max_episode_steps is not None:
            output += f"\nmax_episode_steps={self.max_episode_steps}"
        if print_all or self.order_enforce is not True:
            output += f"\norder_enforce={self.order_enforce}"
        if print_all or self.disable_env_checker is not False:
            output += f"\ndisable_env_checker={self.disable_env_checker}"

        if print_all or self.additional_wrappers:
            wrapper_output: list[str] = []
            for wrapper_spec in self.additional_wrappers:
                if include_entry_points:
                    wrapper_output.append(
                        f"\n\tname={wrapper_spec.name}, entry_point={wrapper_spec.entry_point}, kwargs={wrapper_spec.kwargs}"
                    )
                else:
                    wrapper_output.append(
                        f"\n\tname={wrapper_spec.name}, kwargs={wrapper_spec.kwargs}"
                    )

            if len(wrapper_output) == 0:
                output += "\nadditional_wrappers=[]"
            else:
                output += f"\nadditional_wrappers=[{','.join(wrapper_output)}\n]"

        if disable_print:
            return output
        else:
            print(output)
