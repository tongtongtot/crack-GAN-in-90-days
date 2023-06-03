import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image

from VAE_model import VAE

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--epoch", type = int, default = 50)
opt = parser.parse_args()

os.makedirs("images", exist_ok = True)
os.makedirs("original", exist_ok = True)
os.makedirs("trained_model", exist_ok = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpus = [0,1]
model = VAE().to(device)
model = nn.DataParallel(model.to(device), device_ids = gpus, output_device = gpus[0])
optimizer = torch.optim.Adam(model.parameters(), opt.lr)

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the PIL Image to a tensor and scales it [0, 1]
])
trainData = torchvision.datasets.MNIST('./data/',train = True,transform = transform,download = True)
dataloader = torch.utils.data.DataLoader(dataset = trainData,batch_size = opt.batch_size, shuffle=True)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # print("recon:",recon_x)
    # print("x:", x.view(-1,784))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print("BCE:", BCE)
    # print("KLD:", KLD)
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        # save_image(data, "original/origin%d.png" % (batch_idx + 1) , nrow=8, normalize=True)
        # print("data:", data)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
    
    if epoch % 10 == 9:
        torch.save(model.state_dict(), 'trained_model/model_' + str(epoch + 1))
    # z = torch.randn(1,20)
    # sample = model.decode(z)
    # save_image(sample, "images/epoch%d.png"%epoch , nrow = 4, Normalize = True)

for epoch in range(opt.epoch):  # 50 epochs
    train(epoch)