from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from methods.src.shap.utils import ShapType


class SingleAgentBase(ABC):
    @abstractmethod
    def explain(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def plot(
        self,
        shap_values: Any,
        plot_type: ShapType,
        feature_names: list[str] | None = None,
        include: list[str] | None = None,
    ):
        raise NotImplementedError
