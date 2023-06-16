import torch
import numpy as np
import torch.nn as nn
from scipy import stats
from scipy import sparse
from models.vae import VAE
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

class new_model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = VAE(opt)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), opt.lr)
        self.cirterion = nn.MSELoss(reduction = 'mean')
        self.cos = torch.nn.CosineSimilarity()
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # def norm_var(self, var):
    #     return torch.exp(0.5 * var)

    # def get_KLD(self):
    #     mu_con, logvar_con, mu_sti, logvar_sti = self.outs['KLD']
    #     # logvar_con = self.outs['KLD'][1]
    #     # mu_sti = self.outs['KLD'][2]
    #     # logvar_sti = self.outs['KLD'][3]
    #     # KLD1 = -0.5 * torch.sum(1 + logvar_con - mu_con.pow(2) - logvar_con.exp())
    #     # KLD2 = -0.5 * torch.sum(1 + logvar_sti - mu_sti.pow(2) - logvar_sti.exp())

    #     # return KLD1, KLD2
    #     qz_con = Normal(mu_con, self.norm_var(logvar_con))
    #     qz_sti = Normal(mu_sti, self.norm_var(logvar_sti))
    #     return kl(qz_con, Normal(0,1)).sum(dim = 1).mean(), kl(qz_sti, Normal(0,1)).sum(dim = 1).mean()

    def compute_loss(self):
        con_output, sti_output = self.outs['decoder']
        con_l2_loss = self.cirterion(con_output, self.sti)
        sti_l2_loss = self.cirterion(sti_output, self.sti)
        # z_con_con, z_con_sty, z_sti_con, z_sti_sty = self.outs['z']
        # con_loss = self.cirterion(z_con_con, z_sti_con)
        # cos_loss = abs(self.cos(z_con_sty, z_sti_sty)).mean()
        # KLD1, KLD2 = self.get_KLD()
        # KLD1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.loss = con_l2_loss + sti_l2_loss 
        # + cos_loss + con_loss
        # print("cos_loss:", cos_loss)
        # print("self.loss:", self.loss)
        # print("con_loss:", con_loss)
        # print("loss:")
        # print(self.loss)
        # print(con_l2_loss)
        # print(sti_l2_loss)
        # print(KLD1)
        # print(KLD2)
        return self.loss

    def get_loss(self):
        return self.loss

    def set_input(self, sti, con):
        b,c,g = sti.shape
        self.sti = sti.view(b, g).to(self.opt.device)
        b,c,g = con.shape
        self.con = con.view(b,g).to(self.opt.device)

    # def get_con_z(self):
    #     z_con, mu, logvar = self.model.get_con_z(self.con)
    #     z_sti, mu, logvar = self.model.get_con_z(self.sti)
    #     return z_con, z_sti
    # def get_con_z(self):
    #     z_con, mu, logvar = self.model.get_con_z(self.con)
    #     z_sti, mu, logvar = self.model.get_con_z(self.sti)
    #     return z_con, z_sti
        
    def forward(self):
        self.outs = self.model(self.con, self.sti)
        # self.con_output, self.sti_output, self.mu_con, self.logvar_con, self.mu_sti, self.logvar_sti = self.model(self.con, self.sti)

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_parameter(self):
        self.model.train()
        self.forward()
        self.compute_loss()
        self.backward()

    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data).to(self.opt.device)
        else:
            Exception("This is not a numpy")
        return data

    def tensor2numpy(self, data):
        data = data.cpu().detach().numpy()
        return data

    def adata2numpy(self, adata):
        if sparse.issparse(adata.X):
            return adata.X.A
        else:
            return adata.X
    
    def adata2tensor(self, adata):
        return self.numpy2tensor(self.adata2numpy(adata))

    def _avg_vector(self, adata, condition):
        data = self.adata2numpy(adata)
        tensor_data = self.numpy2tensor(data)
        if condition == 'context':
            latent, _, __ = self.model.get_con_z(tensor_data)
            # latent = self.outs['z'][]
        else:
            latent, _, __ = self.model.get_sty_z(tensor_data)
        latent_avg = np.average(self.tensor2numpy(latent), axis=0)
        return latent_avg

    def get_z(self, adata, condition):
        if condition == 'content':
            z, mu, logvar = self.model.get_con_z(self.adata2tensor(adata))
            # print("z:",z)
            return self.tensor2numpy(z)
        else:
            z, mu, logvar = self.model.get_sty_z(self.adata2tensor(adata))
            return self.tensor2numpy(z)
    
    def get_stim_pred(self, ctrl_x, stim_x, pred_data, condition):
        latent_ctrl = self._avg_vector(ctrl_x, condition)
        latent_stim = self._avg_vector(stim_x, condition)
        # delta = self.outs['delta']
        delta = latent_stim - latent_ctrl
        delta = self.numpy2tensor(delta)
        tensor_train_x = self.adata2tensor(pred_data)
        if condition == 'context':
            latent_cd, _, __ = self.model.get_con_z(tensor_train_x)
            # latent_cd = 
        else:
            latent_cd, _, __ = self.model.get_sty_z(tensor_train_x)
        
        # print(delta.shape)
        # print(latent_cd.shape)

        stim_pred = delta + latent_cd
        return stim_pred

    def predict(self, pred_data, ctrl_data, stim_data):
        self.model.eval()
        # self.forward()
        # ctrl_x = adata[adata.obs[condition_key] == ctrl_key, :]
        # stim_x = adata[adata.obs[condition_key] == stim_key, :]
        stim_pred_con = self.get_stim_pred(ctrl_data, stim_data, pred_data, 'context')
        stim_pred_sty = self.get_stim_pred(ctrl_data, stim_data, pred_data, 'style')
        stim_pred = torch.cat([stim_pred_con, stim_pred_sty], axis = -1)
        gen_img = self.model.decoder(stim_pred)
        return self.tensor2numpy(gen_img)

    # def get_rsqure(self):
    #     pred = self.tensor2numpy(self.con_output)
    #     truth = self.tensor2numpy(self.sti)
    #     x_diff = np.asarray(np.mean(pred, axis=0)).ravel()
    #     y_diff = np.asarray(np.mean(truth, axis=0)).ravel()
    #     m, b, r_value, p_value, std_err = stats.linregress(x_diff, y_diff)
    #     x = np.asarray(np.var(pred, axis=0)).ravel()
    #     y = np.asarray(np.var(truth, axis=0)).ravel()
    #     m, b, r_dif_value, p_value, std_err = stats.linregress(x, y)
    #     return r_value ** 2, r_dif_value **2
