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
        # print("please give me some out put:", opt)
        self.opt = opt
        self.train = sc.read(self.opt.read_path)
        # sc.pp.log1p(self.train)
        # sc.pp.normalize_total(self.train, target_sum = 1e3)
        # x = log(1+x)
        # print("train,", self.train.shape)
        self.train = self.balance(self.train)
        self.condition_mask = (self.train.obs['condition'] == 'stimulated')
        self.sti = self.train[self.condition_mask].copy()
        self.con = self.train[~self.condition_mask].copy()
        # print(self.sti)
        self.sti_len = len(self.sti)
        self.con_len = len(self.con)
        self.max_len = max(len(self.sti), len(self.con))
        # print("train2,", self.train.shape)
        # self.balance(self.train)
        # Because we have to ramdomly pick some of the photos every epoch, we cannot write it in the dataloader   

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
        # print(len(self.train))
        # print("train_use,", self.train.shape)
        sti_idx = random.randint(0, self.sti_len - 1) if idx >= self.sti_len else idx
        con_idx = random.randint(0, self.con_len - 1) if idx >= self.con_len else idx
        sti = self.sti[sti_idx]
        # print(self.sti[sti_idx])
        con = self.con[con_idx]
        # print("data", data.shape)
        return (self.adata2tensor(sti), self.adata2tensor(con))

    def __len__(self):
        # print("this is the lenth", len(self.train))
        return self.max_len

    def balance(self, adata):
        cell_type = adata.obs[self.opt.type]

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
            # print(len(index_cls))

        balanced_data = adata[np.concatenate(index_add)].copy()
        return balanced_data
        # lables.unique()
