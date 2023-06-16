# from options.opt import opt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import scanpy as sc
import torch.utils.data as data
import numpy as np
import random
from scipy import sparse

class customDataloader(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.train = sc.read(self.opt.read_path)
        self.pred_data = self.train[((self.train.obs[opt.cell_type_key] == opt.exclude_celltype) & (self.train.obs["condition"] == opt.ctrl_key))]
        self.ctrl_data = self.train[self.train.obs['condition'] == opt.ctrl_key, :]
        self.stim_data = self.train[self.train.obs['condition'] == opt.stim_key, :]
        self.new_train = self.train[~((self.train.obs[opt.cell_type_key] == opt.exclude_celltype) & (self.train.obs["condition"] == opt.stim_key))]
        self.train = self.balance(self.train)
        self.condition_mask = (self.train.obs[opt.condition_key] == opt.stim_key)
        self.sti = self.train[self.condition_mask].copy()
        self.con = self.train[~self.condition_mask].copy()
        self.sti_len = len(self.sti)
        self.con_len = len(self.con)
        self.max_len = max(len(self.sti), len(self.con))

    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data).cuda()
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
    
    def __getitem__(self, idx):       
        sti_idx = random.randint(0, self.sti_len - 1) if idx >= self.sti_len else idx
        con_idx = random.randint(0, self.con_len - 1) if idx >= self.con_len else idx
        sti = self.sti[sti_idx]
        con = self.con[con_idx]
        return (self.adata2tensor(sti), self.adata2tensor(con))

    def __len__(self):
        return self.max_len

    def get_val_data(self):
        return self.pred_data, self.ctrl_data, self.stim_data, self.train, self.new_train

    def balance(self, adata):
        cell_type = adata.obs[self.opt.cell_type_key]

        class_num = np.unique(cell_type)
        type_num = {}
        max_num = -1
        for i in class_num:
            type_num[i] = cell_type[cell_type == i].shape[0]
            max_num = max(max_num, type_num[i])
        
        index_add = []
        for i in class_num:
            class_index = np.array(cell_type == i)
            index_cls = np.nonzero(class_index)[0]
            index_cls = index_cls[np.random.choice(len(index_cls), max_num)]
            index_add.append(index_cls)

        balanced_data = adata[np.concatenate(index_add)].copy()
        return balanced_data