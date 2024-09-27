from abc import ABC, abstractmethod
from typing import Any

import numpy as np

<<<<<<< HEAD
<<<<<<< HEAD
=======
from methods.src.shap.utils import ShapType

>>>>>>> 41574cb (Cfeat shap iamge)
=======
>>>>>>> df19d30 (feat: shap changes)

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
