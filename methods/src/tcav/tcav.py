import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
import numpy as np

from environments.gymnasium.wrappers import TCAVWrapper

from .linear_classifier import LinearClassifier


class TCAV:
    def __init__(self, env: TCAVWrapper, policy: nn.Module) -> None:
        self.env = env
        self.policy = policy
        self.num_layers = len(list(policy.children()))
        self.activations = {}

        for layer in self.policy.children():
            layer.register_forward_hook(self._module_hook)

    def run(self, concept: str):
        positive_dataset = self.env.get_concept_inputs(concept, samples=1000)
        negative_dataset = self.env.get_concept_inputs("random", samples=1000)
        test_dataset = self.env.get_concept_inputs(concept, samples=200)

        positive_activations = self._compute_activations(positive_dataset)
        negative_activations = self._compute_activations(negative_dataset)

        test_dataset.requires_grad_(True)
        test_activations = self._compute_activations(test_dataset)

        for layer in self.policy.children():
            linear_classifier = self.train_linear_classifier(
                positive_activations[layer], negative_activations[layer]
            )
            cav = self._get_cav(linear_classifier)
            sensitivity = self._concept_sensitivity(test_activations[layer], cav)

    def _get_cav(self, classifier: nn.Module):
        return classifier.weight[1].detach().numpy()

    def _concept_sensitivity(self, activations: dict, cav: np.ndarray):
        return
        input = activations["input"]
        output = activations["output"]
        gradient_score = torch.autograd.grad(
            output, input, grad_outputs=cav, create_graph=True
        )[0]
        return input.grad.mean(dim=0).detach().numpy()

    def _module_hook(self, module: nn.Module, input, output):
        self.activations[module] = {
            "output": output.detach(),
            "input": input.detach(),
        }

    def _compute_activations(self, inputs: torch.Tensor):
        self.activations.clear()
        _ = self.policy(inputs)
        return self.activations

    def train_linear_classifier(
        self,
        positive_samples: dict,
        negative_samples: dict,
        lr=1e-3,
        epochs=100,
    ):
        positive_activations = positive_samples[str]["output"]
        negative_activations = negative_samples[str]["output"]

        positive_labels = torch.ones(positive_activations.size(0))
        negative_labels = torch.zeros(negative_activations.size(0))

        activations = torch.cat([positive_activations, negative_activations], dim=0)
        labels = torch.cat([positive_labels, negative_labels], dim=0)

        permuted_indices = torch.randperm(activations.size(0))
        activations = activations[permuted_indices]
        labels = labels[permuted_indices]

        num_classes = 2
        input_size = positive_activations.size(1)
        classifier = LinearClassifier(input_size, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(classifier.parameters(), lr=lr, amsgrad=True)

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = classifier(activations)
            loss = criterion(logits, labels.long())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: {loss.item()}")

        return classifier
