from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class MultiAgentBase(ABC):
    @abstractmethod
    def explain(self) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def plot(
        self,
        shap_values: Any,
        feature_names: list[str] | None = None,
        include: list[str] | None = None,
    ):
        raise NotImplementedError
