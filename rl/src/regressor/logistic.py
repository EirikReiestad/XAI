import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc = None  # Initialize fc as None

    def fit(self, x, y, epochs=200, lr=0.001, verbose: bool = False):
        if self.fc is None:  # Only initialize fc once
            input_size = x.shape[1]
            self.fc = nn.Linear(input_size, 1)
            self.optimizer = optim.AdamW(self.parameters(), lr=lr)
            self.criterion = nn.BCELoss()

        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0 and verbose:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

    def score(self, x, y):
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        output = self(x)
        return (output > 0.5).eq(y).sum().item() / y.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
        assert self.fc is not None, "fit() must be called before forward()"
        return torch.sigmoid(self.fc(x))

    @property
    def coef_(self):
        assert self.fc is not None, "fit() must be called before coef_"
        return self.fc.weight.detach().numpy()
