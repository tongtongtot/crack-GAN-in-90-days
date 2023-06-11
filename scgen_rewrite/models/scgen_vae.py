import torch
from torch import nn
from options.opt import opt

class SCGENVAE(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = opt.input_layer
        hidden_dim = opt.hidden_layer
        latent_dim = opt.latent_layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mu_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.logvar_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.img_size = input_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

    def get_z(self, x):
        hidden = self.encoder(x)
        mu = self.mu_encoder(hidden)
        logvar = self.logvar_encoder(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.get_z(x)
        return self.decoder(z), mu, logvar