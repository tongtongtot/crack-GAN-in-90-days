import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def makepaths(path, save_name='a.txt'):
    # path1/path2/ ...
    # goal
    # data/random/a.txt
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(os.path.join(path, save_name), 'w') as f:
        f.write('aaa')

class FashionMNIST(Dataset):
    def __init__(self, path, img_size, transform=None):
        self.transform = transform
        fashion_df = pd.read_csv(path)
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, img_size, img_size)
        self.labels = fashion_df.label.values
        print('Image size:', self.images.shape)
        print('--- Label ---')
        print(fashion_df.label.value_counts())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.images[idx]
        img = Image.fromarray(self.images[idx])
            
        if self.transform:
            img = self.transform(img)
        
        return img, label

class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_size, class_num):
        super().__init__()
        
        self.z_size = z_size
        self.img_size = img_size
        
        self.label_emb = nn.Embedding(class_num, class_num)
        
        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], self.img_size * self.img_size),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        
        # Reshape z
        z = z.view(-1, self.z_size)
        
        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        
        # Concat image & label
        x = torch.cat([z, c], 1)
        
        # Generator out
        out = self.model(x)
        
        return out.view(-1, self.img_size, self.img_size)

class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, img_size, class_num):
        super().__init__()
        
        self.label_emb = nn.Embedding(class_num, class_num)
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(self.img_size * self.img_size + class_num, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        
        # Reshape fake image
        x = x.view(-1, self.img_size * self.img_size)
        
        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        
        # Concat image & label
        x = torch.cat([x, c], 1)
        
        # Discriminator out
        out = self.model(x)
        
        return out.squeeze()

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    
    # Init gradient
    g_optimizer.zero_grad()
    
    # Building z
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    
    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)
    
    # Generating fake images
    fake_images = generator(z, fake_labels)
    
    # Disciminating fake images
    validity = discriminator(fake_images, fake_labels)
    
    # Calculating discrimination loss (fake images)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    
    # Backword propagation
    g_loss.backward()
    
    #  Optimizing generator
    g_optimizer.step()
    
    return g_loss.data

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    
    # Init gradient 
    d_optimizer.zero_grad()

    # Disciminating real images
    real_validity = discriminator(real_images, labels)
    
    # Calculating discrimination loss (real images)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    
    # Building z
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    
    # Building fake labels
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)
    
    # Generating fake images
    fake_images = generator(z, fake_labels)
    
    # Disciminating fake images
    fake_validity = discriminator(fake_images, fake_labels)
    
    # Calculating discrimination loss (fake images)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    
    # Sum two losses
    d_loss = real_loss + fake_loss
    
    # Backword propagation
    d_loss.backward()
    
    # Optimizing discriminator
    d_optimizer.step()
    
    return d_loss.data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# train_data_path = './data/Fashion MNIST/fashion-mnist_train.csv' # Path of data
# valid_data_path = './data/Fashion MNIST/fashion-mnist_test.csv' # Path of data

img_size = 28 # Image size
batch_size = 1024  # Batch size

os.makedirs("images", exist_ok=True)
os.makedirs("trained_model", exist_ok=True)
os.makedirs("original", exist_ok=True)

# Model
z_size = 100
generator_layer_size = [256, 512, 1024]
discriminator_layer_size = [1024, 512, 256]

# Training
epochs = 1000  # Train epochs
learning_rate = 1e-4

class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_num = len(class_list)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
trainData = torchvision.datasets.MNIST('./data/',train = True,transform = transform,download = True)
dataloader = torch.utils.data.DataLoader(dataset = trainData,batch_size = batch_size,shuffle = True)

generator = Generator(generator_layer_size, z_size, img_size, class_num).to(device)
discriminator = Discriminator(discriminator_layer_size, img_size, class_num).to(device)

criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()
    # g_optimizer.cuda()
    # d_optimizer.cuda()

Validation = False
if Validation:
    get_epoch = 991
    generator.load_state_dict(torch.load('trained_model/G_' + str(get_epoch)))
    discriminator.load_state_dict(torch.load('trained_model/D_' + str(get_epoch)))
    z = Variable(torch.randn(8, z_size)).to(device)
    labels = Variable(torch.LongTensor([2,0,2,3,0,5,3,1])).to(device)
    os.makedirs("results", exist_ok=True)
    save_image(generator(z, labels).unsqueeze(1), "results/ans_%d epochs.png" % get_epoch, nrow=8, normalize=True)
    exit(0)

for epoch in range(epochs):
    print('Starting epoch {}...'.format(epoch+1))
    for i, (images, labels) in enumerate(dataloader):
        # save_image(images, "original/origin%d.png" % (i + 1) , nrow=1, normalize=True)
        # Train data
        real_images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        # Set generator train
        generator.train()
        # Train discriminator
        d_loss = discriminator_train_step(len(real_images), discriminator,generator, d_optimizer, criterion, real_images, labels)

        # Train generator
        g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
    # Set generator eval
    generator.eval()
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
    # Building z 
    z = Variable(torch.randn(class_num, z_size)).to(device)
    # Labels 0 ~ 8
    labels = Variable(torch.LongTensor(np.arange(class_num))).to(device)
    # Generating images
    # save_image(generator(z, labels).data, "images/epoch %d.png" % (epoch + 1) , nrow=8, normalize=True)
    # sample_images = generator(z, labels).unsqueeze(1).data.cpu()
    # save_image(generator(z, labels).unsqueeze(1), "epoch%d.png"% (epoch + 1), nrow = 9, normalize = True)
    gen_pic = generator(z,labels).unsqueeze(1)

    # print(gen_pic.shape)

    save_image(gen_pic, "images/epoch %d.png" % (epoch + 1) , nrow=10, normalize=True)
    # Gpath = os.path.join('/trained_','G_', str(epoch_name + 1) + '.')
	# Dpath = os.path.join('/trained_','D_', str(epoch_name + 1))
    if epoch % 10 == 9:
        torch.save(generator.state_dict(), 'trained_model/G_' + str(epoch + 1))
        torch.save(discriminator.state_dict(), 'trained_model/D_' + str(epoch + 1))
    # Show images
    # grid = make_grid(sample_images, nrow=3, normalize=True).permute(1,2,0).numpy()
    # plt.imshow(gird)
    # plt.show()