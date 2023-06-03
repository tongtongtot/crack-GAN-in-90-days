import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
from collections import OrderedDict
from PIL import Image
from VAE_model import VAE
import numpy as np
import matplotlib.pyplot as plt
from CVAE_model import CVAE
import torchvision.utils

parser = argparse.ArgumentParser()
parser.add_argument("--get_epoch", type = int, default = 100)
parser.add_argument("--get_batch", type = int, default = 1000)
parser.add_argument("--path", type = str, default = 'trained_model/model_CVAE_')
parser.add_argument("--model", type = str, default = 'CVAE')
parser.add_argument("--labels", nargs = '+', default = [2,0,2,3,0,6,0,2])
parser.add_argument("--save_path", type = str, default = 'result/')
parser.add_argument("--latent_dim", type = int, default = 20)
parser.add_argument("--hidden_dim", type = int, default = 400)
parser.add_argument("--label_dim", type = int, default = 10)
parser.add_argument("--input_dim", type = int, default = 28*28)
opt = parser.parse_args()

os.makedirs("result", exist_ok = True)

if opt.model == 'VAE':
    model = VAE()
    state_dict = torch.load(opt.path + str(opt.get_epoch))
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove `module.` prefix
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
    samples = []
    for i in range(64):
        with torch.no_grad():
            z = torch.randn(1, 20)
            sample = model.decode(z)
            samples.append(sample.view(1, 28, 28))
        # Reshape the tensor and convert it to a PIL image
    save_image(samples, "result/result_VAE_%d.png" % opt.get_epoch)

elif opt.model == 'CVAE':
    model = CVAE(opt.input_dim, opt.label_dim, opt.hidden_dim, opt.latent_dim)
    state_dict = torch.load(opt.path + str(opt.get_batch))
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove `module.` prefix
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
    z = torch.randn(8, 20)
    labels = torch.tensor([2,0,2,3,0,6,0,2])
    gen_img = model.generator(z, labels)
    save_image(gen_img.view(8,1,28,28), "result/reslt_CVAE_%d.png" % opt.get_batch , nrow = 8, Normalize = True)

