from data_handler import DataHandler
import torch
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CAV:
    def __init__(
        self, model: nn.Module, positive_sample_path: str, negative_sample_path: str
    ):
        self._model = model
        self._model.eval()

        self._load_data(positive_sample_path, negative_sample_path)

        self._register_hooks()

        self._activations = {}
        self._cavs = {}
        self._cav_scores = {}

    def _register_hooks(self):
        for name, layer in self._model.named_children():
            if not isinstance(layer, nn.Sequential):
                layer.register_forward_hook(self._module_hook)
                continue
            for sub_layer in layer:
                sub_layer.register_forward_hook(self._module_hook)

    def _load_data(self, positive_sample_path: str, negative_sample_path: str):
        positive_data = DataHandler()
        positive_data.load_data_from_path(positive_sample_path)

        self._positive_data, self._test_positive_data = positive_data.split(0.8)

        self._negative_data = DataHandler()
        self._negative_data.load_data_from_path(negative_sample_path)

    def compute_cavs(self):
        positive_data, positive_labels = self._positive_data.get_data_lists()
        negative_data, negative_labels = self._negative_data.get_data_lists()
        positive_activations = self._compute_activations(positive_data)
        negative_activations = self._compute_activations(negative_data)
        for layer in self._activations.keys():
            self._compute_cav(
                layer, positive_activations[layer], negative_activations[layer]
            )

    def _compute_cav(
        self, layer, positive_activations: dict, negative_activations: dict
    ):
        pos_act = positive_activations["output"].numpy()
        neg_act = negative_activations["output"].numpy()

        pos_act = pos_act.reshape(pos_act.shape[0], -1)
        neg_act = neg_act.reshape(neg_act.shape[0], -1)

        pos_labels = np.ones(pos_act.shape[0])
        neg_labels = np.zeros(neg_act.shape[0])

        combined_activations = np.concatenate([pos_act, neg_act])
        combined_labels = np.concatenate([pos_labels, neg_labels])

        # TODO: Enable when working
        # idx = np.random.permutation(combined_activations.shape[0])
        # combined_activations = combined_activations[idx]
        # combined_labels = combined_labels[idx]

        scaler = StandardScaler()
        combined_activations = scaler.fit_transform(combined_activations)

        regressor = LogisticRegression()
        regressor.fit(combined_activations, combined_labels)

        self._cav = regressor.coef_
        self._cavs[layer] = self._cav
        return self._cav

    def compute_cav_scores(self):
        test_data, test_labels = self._test_positive_data.get_data_lists()
        test_activations = self._compute_activations(test_data)
        for layer in self._activations.keys():
            self._compute_cav_score(layer, test_activations[layer])

    def _compute_cav_score(self, layer, test_activations: dict):
        layer_activations = test_activations["output"].numpy()
        batch_size = layer_activations.shape[0]
        layer_activations_flat = layer_activations.reshape(batch_size, -1)
        cav_score = layer_activations_flat @ self._cavs[layer].T
        cav_score = np.maximum(cav_score - 0.5, 0)
        mean_cav_score = np.mean(cav_score)
        self._cav_scores[layer] = mean_cav_score
        return mean_cav_score

    def _compute_activations(self, inputs: list[np.ndarray], requires_grad=False):
        self._activations.clear()

        torch_inputs = torch.stack(
            [torch.tensor(input_array, dtype=torch.float32) for input_array in inputs]
        ).requires_grad_(requires_grad)
        _ = self._model(torch_inputs)
        return self._activations

    def _module_hook(self, module: nn.Module, input, output):
        self._activations[module] = {
            "input": input[
                0
            ].detach(),  # TODO: Fix this please future Eirik. It should not be indexed.
            "output": output.detach(),
        }

    @property
    def cav_scores(self) -> dict:
        return self._cav_scores
