import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.content_encoder = nn.Sequential(
            nn.Linear(opt.input_dim, opt.hidden_dim),
            nn.ReLU(),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.ReLU(),
            nn.Linear(opt.hidden_dim, opt.context_latent_dim * 2)
        )
        self.style_encoder = nn.Sequential(
            nn.Linear(opt.input_dim, opt.hidden_dim),
            nn.ReLU(),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.ReLU(),
            nn.Linear(opt.hidden_dim, opt.style_latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(opt.context_latent_dim + opt.style_latent_dim, opt.hidden_dim),
            nn.ReLU(),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.ReLU(),
            nn.Linear(opt.hidden_dim, opt.input_dim),
            nn.ReLU()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def get_con_z(self, x):
        # print(x.shape)
        h = self.content_encoder(x)
        mu , logvar = torch.split(h, h.size(-1) // 2, dim = -1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def get_sty_z(self, x):
        h = self.style_encoder(x)
        mu , logvar = torch.split(h, h.size(-1) // 2, dim = -1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def get_delta(self, con, sti):
        z_con_con, self.mu_con, self.logvar_con = self.get_con_z(con)
        z_con_sty, _, __ = self.get_sty_z(con)
        z_sti_con, self.mu_sti, self.logvar_sti = self.get_con_z(sti)
        z_sti_sty, _, __ = self.get_sty_z(sti)
        self.delta = z_sti_sty - z_con_sty
        return z_con_con, z_con_sty, z_sti_con, z_sti_sty
    
    def forward(self, con, sti):
        z_con_con, z_con_sty, z_sti_con, z_sti_sty = self.get_delta(con, sti)
        x_con = torch.cat([z_con_con, z_con_sty + self.delta], dim = -1)
        x_sti = torch.cat([z_sti_con, z_sti_sty], dim = -1)
        return {
            'decoder': [self.decoder(x_con), self.decoder(x_sti)], 
            'KLD': [self.mu_con, self.logvar_con, self.mu_sti, self.logvar_sti], 
            'delta': self.delta,
            'z': [z_con_con, z_con_sty, z_sti_con, z_sti_sty]
            }

        # z1, mu1, var1 = self.enc1(x1)
        # z2, mu2, var2 = self.enc2(x2)
        # delta = z2 - z1
        # concat_z = [z1, delta]
        # out = self.dec(concat_z)
        # ...
        # return {'', ....}

        # return self.decoder(x_con), self.decoder(x_sty), self.mu_con, self.logvar_con, self.mu_sti, self.logvar_sti
