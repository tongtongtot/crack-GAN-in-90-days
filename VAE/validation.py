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

parser = argparse.ArgumentParser()
parser.add_argument("--get_epoch", type = int, default = 100)
opt = parser.parse_args()

model = VAE()
os.makedirs("result", exist_ok = True)

state_dict = torch.load('trained_model/model_' + str(opt.get_epoch))
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

# samples = torch.cat(samples, dim = 0)

# grid_img = torchvision.utils.make_grid(samples, nrow=8, padding=1, normalize=True)

save_image(samples, "result/result_%d.png" % opt.get_epoch)

# Convert the grid to a numpy array and transpose the axes
# grid_img = grid_img.cpu().numpy().transpose((1, 2, 0))

# print(grid_img.shape)
# exit(0)

# Plot the grid and save it
# plt.imshow(grid_img, cmap='gray')
# plt.axis('off')
# plt.savefig('vae_output.png', bbox_inches='tight')