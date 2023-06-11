import torch
from torch import nn
from opt import opt

class SCGENVAE(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = opt.input_layer
        hidden_dim = opt.hidden_layer
        latent_dim = opt.latent_layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.img_size = input_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

    def forward(self, x):
        h = self.encoder(x)
        mu , logvar = torch.split(h, h.size(-1) // 2, dim = -1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


