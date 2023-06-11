import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchvision.utils import save_image
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
import scanpy as sc
from models.scgen_model import SCGEN
from dataloader.customDataset import customDataloader
from options.opt_val import opt
import anndata
from collections import OrderedDict
from tqdm import tqdm

print(opt)
# dataset = customDataloader(opt)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = opt.shuffle)
train = sc.read("data/train_pbmc.h5ad")
model = SCGEN(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if device == 'cuda':
    model.to(device)
    gpus = [0,1]
    # model = nn.DataParallel(model.to(device), device_ids = gpus, output_device = gpus[0])

model.load_state_dict(torch.load(opt.model_save_path + str(opt.get_epoch)))

stim_key = "stimulated"
ctrl_key = "control"
cell_type_key = "cell_type"
condition_key="condition"
data_name = 'scgen_rewrite'

def len(data):
    return data.shape[0]

os.makedirs("./data/reconstructed/scGen/", exist_ok=True)

for idx, cell_type in tqdm(enumerate(train.obs[cell_type_key].unique().tolist())):
    cell_type_data = train[train.obs[cell_type_key] == cell_type]
    cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
    net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
    print(cell_type_data.shape)
    print(cell_type_ctrl_data)
    print(net_train_data.shape)
    print("pre operation done.")
    pred, delta = model.predict(net_train_data, cell_type_ctrl_data, condition_key, stim_key, ctrl_key, cell_type, cell_type_key)
    print("model done")
    # print(pred.shape)
    # print(delta)
    print(pred.shape[0])
    pred_adata = anndata.AnnData(pred, obs={condition_key: [f"pred"] * len(pred), cell_type_key: [cell_type] * len(pred)}, var={"var_names": cell_type_data.var_names})
    print("This is data:")
    print(pred[:10,:10])
    ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X, obs={condition_key: [f"contorl"] * len(cell_type_ctrl_data), cell_type_key: [cell_type] * len(cell_type_ctrl_data)}, var={"var_names": cell_type_ctrl_data.var_names})
    real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
    real_stim_adata = anndata.AnnData(real_stim, obs={condition_key: [f"stimulated"] * len(real_stim), cell_type_key: [cell_type] * len(real_stim)}, var={"var_names": cell_type_data.var_names})
    if idx == 0:
        all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        # all_data.write_h5ad(f"./data/reconstructed/scGen/{data_name}.h5ad")
    else:
        all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)
        # all_data.write_h5ad(f"./data/reconstructed/scGen/{data_name}.h5ad")
    print("all_data:", all_data.shape)

all_data.write_h5ad(f"./data/reconstructed/scGen/{data_name}.h5ad")
