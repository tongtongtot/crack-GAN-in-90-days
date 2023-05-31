import os
import torch
import numpy as np
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=1,
    shuffle=False,
)

def save_img(img, save_root, count):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    cv2.imwrite(os.path.join(save_root, "%06d"%(count)+ '.png'), img)


with open('label.txt', 'w') as f:
    count = 0
    save_path = 'saved_mnist_img/'
    for idx, (imgs, labels) in enumerate(dataloader):
        img = imgs[0]
        label = labels[0]
        save_img(np.array(img).transpose(1,2,0), save_path, count)

        msg = os.path.join(save_path, "%07d"%(count)+'.png') + ' ' + str(int(label))
        f.write(msg + '\n')
        count = count + 1
    


