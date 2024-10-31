from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from environments.gymnasium.envs.tag.utils import AgentType

from .tag_state import Position, TagState


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
                "box-block",
                "A block that can be pushed around by the agent. It should be blocking thee path between the agents.",
                self._get_box_block,
            ),
            Concept(
                "box-not-block",
                "A block that can be pushed around by the agent. It should not be blocking the path between the agents.",
                self._get_box_not_block,
                "Assuming the entrance is at position (5, 4).",
            ),
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
        if concept not in self.concept_names:
            raise ValueError(f"Concept {concept} not found in environment.")

        concept_instance = next((c for c in self._concepts if c.name == concept), None)

        if concept_instance is not None:
            return concept_instance.function(samples)
        else:
            raise NotImplementedError(f"Concept {concept} is not implemented yet.")

    def _get_agents_close(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = True
        states = []
        labels = []
        for _ in range(samples):
            self._state.reset()
            self._state.place_seeker_next_to_hider()
            states.append(self._state.normalized_full_state)
            labels.append(np.random.randint(self._num_actions))
        return states, labels

    def _get_agents_far_apart(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = True
        states = []
        labels = []
        distance_ratio = self._state.width // 2
        for _ in range(samples):
            while True:
                self._state.reset()
                if self._state.agent_distance < distance_ratio:
                    continue
                states.append(self._state.normalized_full_state)
                labels.append(np.random.randint(self._num_actions))
                break
        return states, labels

    def _get_hider_close_to_box(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = True
        states = []
        labels = []
        for _ in range(samples):
            self._state.reset()
            self._state.place_agent_next_to_box(agent_type=AgentType.HIDER)
            states.append(self._state.normalized_full_state)
            labels.append(np.random.randint(self._num_actions))
        return states, labels

    def _get_seeker_close_to_box(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = True
        states = []
        labels = []
        for _ in range(samples):
            self._state.reset()
            self._state.place_agent_next_to_box(agent_type=AgentType.SEEKER)
            states.append(self._state.normalized_full_state)
            labels.append(np.random.randint(self._num_actions))
        return states, labels

    def _get_box_not_exist(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = False

        states = []
        labels = []
        for _ in range(samples):
            self._state.reset()
            normalized_state = self._state.normalized_full_state
            states.append(normalized_state)
            labels.append(
                np.random.randint(self._num_actions)
            )  # TODO: This should be changed to the correct action
        return states, labels

    def _get_box_block(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = False
        states = []
        labels = []
        for _ in range(samples):
            self._state.reset()
            states.append(self._state.normalized_full_state)
            labels.append(
                np.random.randint(self._num_actions)
            )  # TODO: This should be changed to the correct action
        return states, labels

    def _get_box_not_block(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = True

        block_position = Position(5, 4)

        states = []
        labels = []
        for _ in range(samples):
            self._state.reset()
            normalized_state = self._state.normalized_full_state
            normalized_state[*block_position.row_major_order] = 0.0
            states.append(normalized_state)
            labels.append(np.random.randint(self._num_actions))
        return states, labels

    def _get_fully_random(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        state_shape = self._state.normalized_full_state.shape

        num_obstacles = self._state._num_obstacles
        num_seekers = 1
        num_hiders = 1
        num_boxes = self._state._num_boxes

        empty_state = np.zeros(state_shape)

        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = True
        states = []
        labels = []
        for _ in range(samples):
            new_state = empty_state.copy()
            indices = np.random.choice(
                new_state.size,
                num_seekers + num_hiders + num_boxes + num_obstacles,
                replace=False,
            )
            np.put(new_state, indices[:num_obstacles], 1)
            np.put(new_state, indices[num_obstacles : num_obstacles + num_seekers], 2)
            np.put(
                new_state,
                indices[
                    num_obstacles + num_seekers : num_obstacles
                    + num_seekers
                    + num_hiders
                ],
                3,
            )
            np.put(new_state, indices[num_obstacles + num_seekers + num_hiders :], 4)
            self._state.init_full_state = new_state
            self._state.reset()
            states.append(self._state.normalized_full_state)
            labels.append(np.random.randint(self._num_actions))
        return states, labels

    def _get_random(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        self._state.random_seeker_position = True
        self._state.random_hider_position = True
        self._state.random_box_position = True
        states = []
        labels = []
        for _ in range(samples):
            self._state.reset()
            states.append(self._state.normalized_full_state)
            labels.append(np.random.randint(self._num_actions))
        return states, labels

    def __str__(self) -> str:
        return "\n".join(str(concept) for concept in self._concepts)
