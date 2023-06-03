import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image


from CVAE_model import CVAE

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type = float, default = 1e-3)
parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--epoch", type = int, default = 50)
parser.add_argument("--save_size", type = int, default = 10)
parser.add_argument("--save_model_path", type = str, default = 'trained_model/model_CVAE_')
parser.add_argument("--latent_dim", type = int, default = 20)
parser.add_argument("--hidden_dim", type = int, default = 400)
parser.add_argument("--label_dim", type = int, default = 10)
parser.add_argument("--input_dim", type = int, default = 28*28)
parser.add_argument("--save_num", type = int, default = 1000)
opt = parser.parse_args()

os.makedirs("images", exist_ok = True)
# os.makedirs("original", exist_ok = True)
os.makedirs("trained_model", exist_ok = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpus = [0,1]
model = CVAE(opt.input_dim, opt.label_dim, opt.hidden_dim, opt.latent_dim).to(device)
model = nn.DataParallel(model.to(device), device_ids = gpus, output_device = gpus[0])
optimizer = torch.optim.Adam(model.parameters(), opt.lr)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
trainData = torchvision.datasets.MNIST('./data/',train = True,transform = transform,download = True)
dataloader = torch.utils.data.DataLoader(dataset = trainData,batch_size = opt.batch_size, shuffle=True)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch, train_idx):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(dataloader):
        if label.size(0) != opt.batch_size: 
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))
            # print("batch_passed:", train_idx)
            return train_idx
        # print(label.shape)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model.forward(data, label)
        # print("ans:", recon_batch.shape)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        train_idx = train_idx + 1
        if train_idx % opt.save_num == 0:
            torch.save(model.state_dict(), opt.save_model_path + str(train_idx))
            save_image(recon_batch.view(opt.batch_size,1,28,28), "images/CVAE_%d.png" % train_idx, nrow = 8, Normalize = True)
    return train_idx
    
batch_passed_count = 0
for epoch in range(opt.epoch):
    batch_passed_count = train(epoch, batch_passed_count)