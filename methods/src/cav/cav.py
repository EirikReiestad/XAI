import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_handler import DataHandler
from data_handler.src.utils.data import Sample

from typing import Optional

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class CAV:
    def __init__(
        self, model: nn.Module, positive_sample_path: str, negative_sample_path: str
    ):
        self._model = model
        self._model.eval()
        self._load_data(positive_sample_path, negative_sample_path)
        self._register_hooks()

        self._activations = {}

        self.scaler = StandardScaler()

        np.random.seed(None)

    def _register_hooks(self):
        for name, layer in self._model.named_children():
            if not isinstance(layer, nn.Sequential):
                layer.register_forward_hook(self._module_hook)
                continue
            for sub_layer in layer:
                if not isinstance(sub_layer, nn.Linear | nn.Conv2d):
                    continue
                sub_layer.register_forward_hook(self._module_hook)

    def _load_data(self, positive_sample_path: str, negative_sample_path: str):
        positive_data = DataHandler()
        positive_data.load_data_from_path(positive_sample_path)

        self._positive_data, self._test_positive_data = positive_data.split(0.7)

        negative_data = DataHandler()
        negative_data.load_data_from_path(negative_sample_path)

        self._negative_data, self._test_negative_data = negative_data.split(0.7)

    def compute_cavs(
        self, custom_test_data: Optional[list[Sample]] = None
    ) -> tuple[dict, dict, dict]:
        positive_data, positive_labels = self._positive_data.get_data_lists()
        negative_data, negative_labels = self._negative_data.get_data_lists()
        test_data, test_labels = self._test_positive_data.get_data_lists()

        if custom_test_data is not None:
            test_data_handler = DataHandler()
            test_data_handler.load_samples(custom_test_data)
            test_data, test_labels = test_data_handler.get_data_lists()

        positive_activations, positive_output = self._compute_activations(
            positive_data, requires_grad=True
        )
        negative_activations, negative_output = self._compute_activations(
            negative_data, requires_grad=True
        )
        test_activations, test_output = self._compute_activations(
            test_data, requires_grad=True
        )

        cavs = {}
        binary_concept_scores = {}
        tcav_scores = {}

        for i, layer in enumerate(self._activations.keys()):
            regressor = self._compute_regressor(
                positive_activations[layer], negative_activations[layer]
            )
            cav = self._cav(regressor)
            tcav_score = self._tcav_score(test_activations[layer], test_output, cav)
            binary_concept_score = self._binary_concept_score(
                test_activations[layer], regressor
            )

            cavs[layer] = cav
            binary_concept_scores[layer] = binary_concept_score
            tcav_scores[layer] = tcav_score

        return cavs, binary_concept_scores, tcav_scores

    def _preprocess_activations(self, activations: dict) -> np.ndarray:
        numpy_activations = activations["output"].detach().numpy()
        activations = numpy_activations.reshape(numpy_activations.shape[0], -1)
        scaled_act = self.scaler.fit_transform(activations)
        return activations

    def _cav(self, regressor: LogisticRegression):
        return regressor.coef_

    def _compute_regressor(
        self,
        positive_activations: dict,
        negative_activations: dict,
    ) -> LogisticRegression:
        pos_act = self._preprocess_activations(positive_activations)
        neg_act = self._preprocess_activations(negative_activations)

        assert pos_act.shape[1] == neg_act.shape[1]

        positive_labels = np.ones(pos_act.shape[0])
        negative_labels = np.zeros(neg_act.shape[0])

        combined_activations = np.concatenate([pos_act, neg_act])
        combined_labels = np.concatenate([positive_labels, negative_labels])

        idx = np.random.permutation(combined_activations.shape[0])
        combined_activations = combined_activations[idx]
        combined_labels = combined_labels[idx]

        regressor = LogisticRegression(warm_start=True)
        # Randomize weights
        regressor.coef_ = np.random.rand(1, combined_activations.shape[1])
        regressor.intercept_ = np.random.rand(1)
        regressor.fit(combined_activations, combined_labels)

        return regressor

    def _tcav_score(
        self, activations: dict, network_output: torch.Tensor, cav: np.ndarray
    ) -> float:
        torch_activations = activations["output"]
        assert isinstance(
            torch_activations, torch.Tensor
        ), "Activations must be a tensor"
        assert (
            torch_activations.requires_grad
        ), "Activations must have requires_grad=True"
        sensitivity_score = self._sensitivity_score(
            torch_activations, network_output, cav
        )
        return (sensitivity_score > 0).mean()

    def _sensitivity_score(
        self, activations: torch.Tensor, network_output: torch.Tensor, cav: np.ndarray
    ) -> np.ndarray:
        grads = torch.autograd.grad(
            outputs=network_output,
            inputs=activations,
            grad_outputs=torch.ones_like(network_output),
            retain_graph=True,
        )[0]

        grads_flattened = grads.view(grads.size(0), -1).detach().numpy()

        return np.dot(grads_flattened, cav.T)

    def _binary_concept_score(
        self, activations: dict, regressor: LogisticRegression
    ) -> float:
        act = self._preprocess_activations(activations)
        labels = np.ones(act.shape[0])
        score = regressor.score(act, labels)
        return score

    def _compute_activations(
        self, inputs: list[np.ndarray], requires_grad=False
    ) -> tuple[dict, torch.Tensor]:
        self._activations.clear()

        torch_inputs = torch.stack(
            [torch.tensor(input_array, dtype=torch.float32) for input_array in inputs]
        ).requires_grad_(requires_grad)
        output = self._model(torch_inputs)
        return self._activations.copy(), output

    def _module_hook(self, module: nn.Module, input, output):
        self._activations[module] = {
            "input": input[0],
            "output": output,
        }
