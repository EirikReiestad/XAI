from abc import ABC, abstractmethod
from typing import Any

import numpy as np

<<<<<<< HEAD
=======
from methods.src.shap.utils import ShapType

>>>>>>> 41574cb (Cfeat shap iamge)

class MultiAgentBase(ABC):
    @abstractmethod
    def explain(self) -> list[np.ndarray]:
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
