import torch
from torch import nn

class CVAE(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + label_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.img_size = input_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

    def forward(self, x, label):
        # print("x:",x.shape)
        x = x.view(-1, self.img_size)
        # print(x.size())
        c = self.label_emb(label)
        combined_input = torch.cat([x, c], dim = -1)
        h = self.encoder(combined_input)
        # print("h:", h.shape)
        mu , logvar = torch.split(h, h.size(-1) // 2, dim = -1)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, c], dim = -1)
        # print("z:",z.shape)
        # print(z.shape)
        return self.decoder(z), mu, logvar

    def generator(self, z, label):
        c = self.label_emb(label)
        z = torch.cat([z, c], dim = -1)
        return self.decoder(z)