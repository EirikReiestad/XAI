from abc import ABC, abstractmethod
from typing import Any

import shap


class SingleAgentBase(ABC):
    @abstractmethod
    def explain(self) -> shap.GradientExplainer | shap.Explainer:
        raise NotImplementedError

    @abstractmethod
    def plot(
        self,
        shap_values: Any,
        feature_names: list[str] | None = None,
        include: list[str] | None = None,
    ):
        raise NotImplementedError
