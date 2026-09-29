import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        return self.layer2(self.relu(self.layer1(x)))
