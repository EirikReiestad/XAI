from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from environments.gymnasium.envs.tag.utils import AgentType

from .tag_state import TagState


@dataclass
class Concept:
    name: str
    description: str
    function: Callable[[int], Tuple[List[np.ndarray], List[str]]]
    info: str = ""

    def __str__(self) -> str:
        if self.info == "":
            return f"{self.name}: {self.description}"
        return f"{self.name}: {self.description}\n\t{self.info}"


class TagConcepts:
    def __init__(self, state: TagState, num_actions: int) -> None:
        self._state = state
        self._num_actions = num_actions

        self._concepts = [
            Concept(
                "box-not-exist",
                "The box does not exist in the environment.",
                self._get_box_not_exist,
                "This requires manually removing the box from the environment.",
            ),
            Concept(
                "agents-close",
                "The seeker is next to the hider. The seeker should be able to catch the hider.",
                self._get_agents_close,
            ),
            Concept(
                "agents-far-apart",
                "The seeker and the hider are far apart. The seeker should not be able to catch the hider.",
                self._get_agents_far_apart,
                "This has a ratio variable which can be adjusted in the source code.",
            ),
            Concept(
                "hider-close-to-box",
                "The hider is close to the box. The hider should be able to push the box.",
                self._get_hider_close_to_box,
            ),
            Concept(
                "seeker-close-to-box",
                "The seeker is close to the box. The seeker should be able to push the box.",
                self._get_seeker_close_to_box,
            ),
            Concept(
                "seeker-not-exist",
                "The seeker does not exist in the environment.",
                self._get_seeker_exists,
            ),
            Concept(
                "hider-not-exist",
                "The hider does not exist in the environment.",
                self._get_hider_exists,
            ),
            Concept(
                "has-direct-sight",
                "The seeker has a direct line of sight to the hider.",
                self._get_has_direct_sight,
            ),
            Concept(
                "has-no-direct-sight",
                "The seeker does not have a direct line of sight to the hider.",
                self._get_has_no_direct_sight,
            ),
            Concept(
                "random",
                "An arbitrary concept that does not have a specific meaning.",
                self._get_random,
            ),
        ]

    @property
    def concept_names(self) -> list[str]:
        return [concept.name for concept in self._concepts]

    def get_concept(
        self, concept: str, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        assert (
            concept in self.concept_names
        ), f"Concept {concept} not found in environment."

        concept_instance = next((c for c in self._concepts if c.name == concept), None)

        assert (
            concept_instance is not None
        ), f"Concept {concept} not found in environment."
        return concept_instance.function(samples)

    def _generate_samples(
        self, samples: int, position_func: Callable[[], None]
    ) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = False
        self._state.random_box_position = True
        states, labels = [], []
        for _ in range(samples):
            self._state.reset()
            position_func()
            states.append(self._state.normalized_full_state)
            labels.append(np.random.randint(self._num_actions))
        return states, labels

    def _get_agents_close(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(samples, self._state.place_seeker_next_to_hider)

    def _get_agents_far_apart(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(samples, self._state.place_agents_far_apart)

    def _get_hider_close_to_box(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(
            samples, lambda: self._state.place_agent_next_to_box(AgentType.HIDER)
        )

    def _get_seeker_close_to_box(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(
            samples, lambda: self._state.place_agent_next_to_box(AgentType.SEEKER)
        )

    def _get_box_not_exist(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(samples, self._state.remove_box)

    def _get_seeker_exists(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(
            samples, lambda: self._state.remove_agent(AgentType.SEEKER)
        )

    def _get_hider_exists(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(
            samples, lambda: self._state.remove_agent(AgentType.HIDER)
        )

    def _get_has_direct_sight(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(
            samples, self._state.place_agent_with_direct_sight
        )

    def _get_has_no_direct_sight(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(
            samples, self._state.place_agent_with_direct_sight
        )

    def _get_random(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        return self._generate_samples(samples, lambda: None)

    def __str__(self) -> str:
        return "\n".join(str(concept) for concept in self._concepts)
