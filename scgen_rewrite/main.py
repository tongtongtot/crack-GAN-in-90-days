# from options.opt import opt
from options.option import scgen_rewrite_options
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import scanpy as sc
import sys
import os
import argparse
from dataloader.customDataset import customDataloader
from models.scgen_model import SCGEN
import torch.nn as nn
from tqdm import tqdm

opt = scgen_rewrite_options.get_opt()

os.makedirs("saved_model", exist_ok = True)
os.makedirs("saved_picture", exist_ok = True)
os.makedirs("saved_loss", exist_ok = True)

print("opt", opt)

dataset = customDataloader(opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = opt.shuffle)

model = SCGEN(opt)

gpus = [0, 1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if device == 'cuda':
    model.cuda()
    # model = nn.DataParallel(model.to(device), device_ids = gpus, output_device = gpus[0])
loss_all = []


for epoch in tqdm(range(opt.epochs)):
    
    # print("start training:")

    for idx, (sti, con) in enumerate(dataloader):
        # print(sti)
        # print(con)
        model.train()
        model.set_input(sti, con)
        # model.set_input(sti, sti)
        model.update_parameter()
        # loss = model.get_current_loss()
        # with open(opt.loss_save_path + opt.model_name + '.txt', 'w') as file:
        #     file.write('batch: ' +str(idx) + ' :\n')
        #     file.write(str(loss) + '\n')
        # print("The %d th batch" % idx)
    
    # loss_all.append(loss)
    
    if epoch % opt.save_interval == opt.save_interval - 1:
        torch.save(model.state_dict(), opt.model_save_path + str(epoch + 1))
        print(opt.model_save_path + str(epoch + 1))
        #image_save...
    
    # print("This is the %d th epoch now" %epoch, ", and there are still %d epochs" % (opt.epochs - epoch), "and the loss is:", loss)

with open(opt.loss_save_path + opt.model_name + '.txt', 'w') as file:
    for idx, loss in enumerate(loss_all):
        file.write(str(loss) + '\n')
