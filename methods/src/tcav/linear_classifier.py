import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
