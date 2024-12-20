import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, dim_feedforward, output_size, dropout=0.4):
        super().__init__()
        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Additional linear layers
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        # Classifier output layer
        self.classifier = nn.Linear(64, output_size)

    def forward(self, x):
        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Additional linear layers with ReLU activation
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        # Pass through classifier
        out = self.classifier(x)
        return out
