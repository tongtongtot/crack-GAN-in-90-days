# 另一个赛道的图片生成：VAE&CVAE

### 1.VAE

##### 1.1 introduction

$VAE$做的东西其实与$GAN$很像，做的实际上也是一个从一个分布到另一个分布的映射。

只是现在的分布不是随机生成的高斯分布了，而是能代表原本图片的高斯分布。

$VAE$由两个部分组成，一个是$encoder$, 用于生成图片专属的高斯分布，另一个是$decoder$, 用于将高斯分布还原回图片。之后就会统计生成的图片和还原的图片之间的差值，并用这个差值来训练这个神经网络。

##### 1.2 Encoder

$encoder$ 的做法是将原图片投影到 $hidden$_$space$ 然后再分别投影给 $\mu$ 和 $\sigma$ 来得到一个正态分布。

然后我们在这个正态分布上随机选择一个点 (此处用 $z_-score$ 来表示) 然后将这个点输入到 $decoder$ 中生成图像。

因此，我们就可以得出 $encoder$ 的代码：

```python
def __init__(self):
    super(VAE, self).__init__()
    self.fc1 = nn.Linear(784, 400)  # hidden layer
    self.fc21 = nn.Linear(400, 20)  # mu layer
    self.fc22 = nn.Linear(400, 20)  # logvariance layer

def encode(self, x):
    h1 = F.relu(self.fc1(x)) # pic -> hidden + relu
    return self.fc21(h1), self.fc22(h1)  # hidden -> mu, logvariance

def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)  # standard deviation
    eps = torch.randn_like(std)  # eps
    return mu + eps*std  # z score
  
def forward(self, x):
    mu, logvar = self.encode(x.view(-1, 784))  # get mean and standard deviation
    z = self.reparameterize(mu, logvar)  # get z score
```

##### 1.3 Decoder

$Decoder$ 的做法其实和 $Encoder$ 反过来差不多。

也就是 $z_-score \to hidden_-layer \to generate_-picture$

所以我们也很容易就得出代码：

```python
def decode(self, z):
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3))
```

##### 1.4 VAE.model

然后将上面两个融合一下再加入一些 $import$ 的库就变成了：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mu layer
        self.fc22 = nn.Linear(400, 20)  # logvariance layer
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

##### 1.5 Train

这一步其实与之前的基本相同：

```python
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
```

##### 1.6 save

这里 $save_-img$的方式是：

```python
z = torch.randn(1,20)
sample = model.decode(z)
save_image(sample, "images/epoch%d.png"%epoch , nrow = 4, Normalize = True)
```

$save_-model$的方式是： 

```python
if epoch % 10 == 9:
torch.save(model.state_dict(), 'trained_model/model_' + str(epoch + 1))
```





### 2 CVAE

##### 2.1 Introduction

如我们在$GAN$ 到 $CGAN$ 的想法一致，既然生成了一个图片那能否通过标签来控制它生成特定的图片呢？

当然可以，而且方法和 $GAN \to CGAN$ 几乎一致：将 $labels$ 插入到图片后面作为输入。

这里我们还可以换一种理解方式：在将 $labels$ 插入图片后再输入相当于生成了图片的下半部分来让 $decoder$ 生成上半部分。那为了减少差距，$decoder$ 就会生成和加入标签一致的图片。

##### 2.2 Change

改动的地方也不多，除了 $Train$ 中的 $model(data) \to model(data, labels)$ 外，只需要在 $forward$ 中稍加改动即可。

```python
def forward(self, x, labels):
		x = x.view(-1, 784)
		c = self.label_emb(labels)  # self.label_emb = nn.Embedding(label_dim, label_dim)
		x = torch.cat([x,c], dim = -1)
    mu, logvar = self.encode(x)
    
    z = self.reparameterize(mu, logvar)
    z = torch.cat([z,c], dim = -1)
    return self.decode(z), mu, logvar
```

当然，$self.fc1() $ 和 $self.fc3()$ 中的输入维度也需要更改，增加一个 $label_-dim$ 就好了:

```python
def __init__(self):
    super(VAE, self).__init__()
    self.fc1 = nn.Linear(784 + label_dim , 400)
    self.fc21 = nn.Linear(400, 20) 
    self.fc22 = nn.Linear(400, 20)
    self.fc3 = nn.Linear(20 + label_dim , 400)
    self.fc4 = nn.Linear(400, 784)
```