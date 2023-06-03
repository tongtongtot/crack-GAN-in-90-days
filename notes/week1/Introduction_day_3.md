# 4. 模型训练

### 4.1 损失函数

GAN的训练优化目标其实就是如下函数：

![v2-44f9286d725d88202fc0a795c8ca208f_1440w.webp (1440×129) (zhimg.com)](https://pic4.zhimg.com/80/v2-44f9286d725d88202fc0a795c8ca208f_1440w.webp)

可以看到，这里有两个loss：一个是训练鉴别器时使用的 $D$_$loss$， 另一个是训练生成器时使用的 $G$ _ $loss$。

而这个模型的目标是要最小化 G_loss, 以及最大化 D_loss。

这里我们使用了$Adam$优化策略和$BCE\space loss$。 

于是可以写出：

```python
criterion = nn.BCE.loss()
g_optimizer = torch.optimizer.Adam(generator.parameters(), lr = opt.lr)
d_optimizer = torch.optimizer.Adam(discriminator.parameters(), lr = opt.lr)
```

### 4.2 模型迭代

在模型迭代的过程中，我们会做如下步骤：

1. 我们会读取图像和标签(暂时没用)
2. 然后生成一个随机的噪声$z$ 并放入生成器生成一张假的图片，称为$fake$_$img$
3. 之后将$fake$ _ $image$ 放入鉴别器得出 $fake$ _ $image$ 的评分
4. 将这个评分与 $1$ 比较得到 $G$_$loss$
5. 再将输入的图像和$fake$_$image$ 加上真假标签后放入鉴别器中得到$D$ _ $loss$
6. 循环以上过程 $opt.epoch$ 次

由此，我们可以得到这部分的代码：

```python
for epoch in range(opt.epoch):
		for i,(image,labels) in enumerate(dataloader):
				real_imgs = Variable(imgs)
				valid = Variable(Tensor(imgs.size(0)).fill_(1.0), requires_grad = False).to(device)
				fake = Variable(Tensor(imgs.size(0)).fill_(0.0), requires_grad = False).to(device)
				#valid指 全为1
        #fake指 全为0
        
				g_optimizer.zero_grad()
				z = Variable(torch.randn(imgs.shape, opt.latent_dim)).to(device) #z是一个随机生成的数
				gen_img = generator(z) #生成图片
				gen_score = discriminator(gen_img) #获得分数
				g_loss = criterion(gen_score, valid) #得到g_loss
				g_loss.backward()
				g_optimizer.step()
				
        
				real_img = Variable(imgs.type(Tensor)).to(device)
				d_optimizer.zero_grad()
				real_loss = criterion(discriminator(real_img), valid)  #将真正的图片放入训练
				fake_loss = criterion(discriminator(gen_img.detach()), fake)  #将生成的图片放入训练
				d_loss = (real_loss + fake_loss) / 2  #相加得到d_loss
				d_loss.backward()
				d_optimizer.step()
```

至此，模型已经训练完毕。

# 5.保存图片以及模型

这里我们使用$torchvision.utils$ 库中的 $save$_$image$函数来存储图片，用法如下：

```python
os.makedirs(path, exist_ok=True)
save_img(image, path, nrow = 8, normalize = True)
```

我们使用$torch.save$来保存模型即其中的参数，实际上需要保存的其实就是 $generator$ 和 $discriminator$ 这两个东西，用法如下： 

```python
torch.save(generator.state_dict(), path)
torch.save(discriminator.state_dict(), path)
```

然后使用的时候就只需要$load$一下就行了

```python
generator.load_state_dict(torch.load(path))
discriminator.load_state_dict(torch.load(path))
```

之后就像之前一样使用$generator$和$discriminator$就可以了