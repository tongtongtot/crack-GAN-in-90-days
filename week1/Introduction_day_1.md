# 1. 什么是GAN

GAN, 全称是Generative Adversarial Networks, 是一种对抗生成网络，用于生成图片：比如AI换脸，AI绘画风格转换。该模型由两个部分组成，分别是 生成器 generator 和 鉴别器 discriminator。 其中，生成器的作用就是生成图片，而鉴别器的作用就是鉴别该图片究竟是输入的图片还是生成器生成的图片 (若是输入的图片则返回1，否则返回0)。

# 2. GAN的构建(以MNIST数据集为例)

### 2.1. 数据预处理

为了更加贴近实际使用，首先使用 $gen$_$label.py$ 将下载的二进制文件转换为图片

首先使用 $pytorch$ 内置的函数获取 $MNIST$ 数据集：

```python
dataloader = torch.utils.data.DataLoader( #需要import Dataloader和datasets
    datasets.MNIST(
        "../../data/mnist", 
        train=False, 
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=1, 
    shuffle=False, 
)
```

此时数据已经下载到 $/data/mnist$ 目录下，并且已经存储在了$dataloader$中 (格式为 $(图片,标签)$)，下一步需要做的就是将图片从二进制文件转化为$.png$等可以可视化的方式，因此我们构造了以下函数 (需要$import\space cv2$) ：

```python
def save_img(img, save_root, count):
    if not os.path.exists(save_root) #如果没有出现的话
  	    os.makedirs(save_root) #就创建一个
    cv2.imwrite(os.path.join(save_root, "%06d"%(count)+ '.png'), img)
    #由于该数据集中有大约60w张图片，锁此处使用 %06d 对齐文件名
```

之后我们就可以调用 $save$_$img$ 函数来把图片写入该目录

另外为了方便之后读取，我们在 $/data/mnist$ 目录下增加了一个 $.txt$ 文件用于索引，格式为 $图片地址\space 标签$ 

实现的方法很简单，只需要遍历一遍 $dataloader$ 就好了

```python
with open('label.txt', 'w') as f:
    count = 0
    save_path = 'saved_mnist_img/'
    for idx, (imgs, labels) in enumerate(dataloader):
        img = imgs[0]
        label = labels[0]
        save_img(np.array(img).transpose(1,2,0), save_path, count)
        #由于np.array(img)函数期望的图片格式为[Height,Weight,Channel]
        #而pytorch所期望的格式是[Channel,Height,Weight],所以需要转置一下
        msg = os.path.join(save_path, "%07d"%(count)+'.png') + ' ' + str(int(label))
        f.write(msg + '\n')
        count = count + 1
```

