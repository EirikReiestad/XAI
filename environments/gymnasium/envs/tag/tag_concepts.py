from dataclasses import dataclass
import numpy as np
from .tag_state import TagState


@dataclass
class Concept:
    name: str
    description: str


class TagConcepts:
    concepts = [
        Concept(
            "box-block",
            "A block that can be pushed around by the agent. It should be blocking thee path between the agents.",
        ),
        Concept(
            "box-not-exist",
            "The box does not exist in the environment.",
        ),
        Concept(
            "seeker-next-to-hider",
            "The seeker is next to the hider. The seeker should be able to catch the hider.",
        ),
        Concept(
            "random", "An arbitrary concept that does not have a specific meaning."
        ),
    ]

    def __init__(self, state: TagState, num_actions: int) -> None:
        self.state = state
        self.num_actions = num_actions

    @property
    def concept_names(self) -> list[str]:
        return [concept.name for concept in self.concepts]

    def get_concepts_dict(self) -> dict:
        return {concept.name: concept.description for concept in self.concepts}

    def get_concept(
        self, concept: str, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        if concept not in self.concept_names:
            raise ValueError(f"Concept {concept} not found in environment.")
        if concept == "box-block":
            return self._get_box_block_concept(samples)
        elif concept == "box-not-exist":
            return self._get_box_not_exist_concept(samples)
        elif concept == "seeker-next-to-hider":
            return self._get_seeker_next_to_hider_concept(samples)
        elif concept == "random":
            return self._get_random_concept(samples)
        else:
            raise NotImplementedError(f"Concept {concept} is not implemented yet.")

    def _get_seeker_next_to_hider_concept(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        self.state.random_seeker_position = True
        self.state.random_hider_position = True
        self.state.random_box_position = True
        states = []
        labels = []
        for _ in range(samples):
            self.state.reset()
            self.state.place_seeker_next_to_hider()
            states.append(self.state.normalized_full_state)
            labels.append(np.random.randint(self.num_actions))
        return states, labels

    def _get_box_not_exist_concept(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        self.state.random_seeker_position = True
        self.state.random_hider_position = True
        self.state.random_box_position = False
        states = []
        labels = []
        for _ in range(samples):
            self.state.reset()
            self.state.remove_box()
            states.append(self.state.normalized_full_state)
            labels.append(
                np.random.randint(self.num_actions)
            )  # TODO: This should be changed to the correct action
        return states, labels

    def _get_box_block_concept(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        self.state.random_seeker_position = True
        self.state.random_hider_position = True
        self.state.random_box_position = False
        states = []
        labels = []
        for _ in range(samples):
            self.state.reset()
            states.append(self.state.normalized_full_state)
            labels.append(
                np.random.randint(self.num_actions)
            )  # TODO: This should be changed to the correct action
        return states, labels

    def _get_random_concept(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        self.state.random_seeker_position = True
        self.state.random_hider_position = True
        self.state.random_box_position = True
        states = []
        labels = []
        for _ in range(samples):
            self.state.reset()
            states.append(self.state.normalized_full_state)
            labels.append(np.random.randint(self.num_actions))
        return states, labels
