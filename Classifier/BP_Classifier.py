
import torch
import torch.nn as nn

class BPNerualNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x