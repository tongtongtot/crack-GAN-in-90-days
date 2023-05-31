import argparse
import os
import numpy as np
import math

import cv2

from data.CustomDataset import customDataset

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model.DCN_GD import Generator
from model.DCN_GD import Discriminator

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--ngf", type=int, default=16)
parser.add_argument("--ndf", type=int, default=16)
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt.ngf, opt.latent_dim)
discriminator = Discriminator(opt.ndf)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)

dataset = customDataset(opt)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

real_label = torch.ones(size=(opt.batch_size, 1), requires_grad=False)
fake_label = torch.zeros(size=(opt.batch_size, 1), requires_grad=False)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(opt.n_epochs):
    for i, dicts in enumerate(dataloader):
        imgs = dicts['img']
        labels = dicts['label']

        # if i < 930: continue

        for k in range(imgs.shape[0]):
            img = imgs[k]
            label = labels[k]

        valid = Variable(Tensor(imgs.size(0), 1, 1, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1, 1, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # z = torch.randn(opt.batch_size, opt.latent_dim, 7, 7)
        z = torch.randn(imgs.shape[0], opt.latent_dim, 7, 7)
        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        
        # print(gen_imgs.size())

        fake_img = discriminator(gen_imgs)

        # print("layer:", fake_img.size())
        # print("ohhh:", valid.size())

        g_loss = adversarial_loss(fake_img, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # print("here") 
        # print(real_imgs.shape)    
        # Measure discriminator's ability to classify real from generated samples

        tmp = discriminator(real_imgs)

        # print("layer:", real_imgs.size())
        # print("qwq:", tmp.size())
        # print("ohhh:", valid.size())
        # print("ohhh:", fake.size())
        
        real_loss = adversarial_loss(discriminator(real_imgs), valid)

        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        # batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        save_image(gen_imgs.data[:64], "images/epoch %d.png" % (i + 1) , nrow=8, normalize=True)
    
