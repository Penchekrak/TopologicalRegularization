import torch
from torch import nn


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, num_hidden_layers, activation=nn.ReLU):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden_dim)]
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
            ])
        layers.extend([
            activation(),
            nn.Linear(hidden_dim, output_dim),
        ])
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, input):
        batch = input.shape[0]
        latents = torch.randn(batch, self.latent_dim, device=input.device)
        generations = self.net(latents)
        return generations


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, activation=nn.ReLU, sigmoid=True):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
            ])
        layers.extend([
            activation(),
            nn.Linear(hidden_dim, 1),
        ])
        if sigmoid:
            layers.append(torch.nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        scores = self.net(input)
        return scores
