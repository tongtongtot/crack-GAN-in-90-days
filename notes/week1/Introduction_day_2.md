### 2.2 将数据输入模型

书接上回，我们将二进制文件转换为了 $.png$ 图片以及 $label.txt$, 现在我们要把这些图片输入到模型中。 

因此我们构造了一个 $class\space customDataset$ 来导入数据（其实就是放到 $label[]$ 和 $img[]$ 里)。

首先是初始化，在初始化的部分会导入图片的路径和标签

```python
def __init__(self, opt):
    self.opt = opt
    self.path = opt.path #方便导入路径
    txt_file = open(os.path.join(self.path, "label.txt"), "r") #读取label.txt
    lines = txt_file.readlines()
    self.label = []
    self.img_path = [] #初始化label和img_path数组
    self.transform = transform.ToTensor()
    for line_idx, line in enumerate(lines):
        line = line.strip().split(' ')  #移除换行，根据空格分开路径和标签
        self.img_path.append(line[0])
        self.label.append(line[1])  #导入图片路径和标签
```

其次是将图片通过 $cv2$ 导入

```python
def __getitem__(self, idx):
    img = cv2.imread(self.img_path[idx])  #读入图片
    img = img.transpose(2,0,1) #同理，此处需要转置
    return {"img": img, "label": self.label[idx]} 
```

最后是后面的函数需要的 __$len$__ 函数

```python
def __len__(self):
    return len(self.img_path)
```

# 3. 构建模型

### 3.1. 生成器 generator

此处使用的是 $Multi-Layer Perception$ 的全连接层来链接不同的层。

由于图片是 $28\times 28$ 的格式，所以最后应该是一个$1\times784$ 的层。

由此可以得出整个模型的形式 : $latent\space dim \to 1024 \to 784$

而对于每一层，形式是：$nn.linear() \to normalize \to 激活函数$ (此处为$Leaky \space ReLU$)

于是可以写出初始化的代码：

```python
def __init__(self):
    super().__init__() #调用超类
    
    def block(in_num, out_num, normalize = True):  #in_idx: 上一层的大小 out_idx 下一层的大小
        layers = [nn.linear(in_num, out_num)]
        if normalize:
            layers.append(BatchNorm1d(out_num, 0.8))
        layers.append(nn.LeakyReLU(0.2, True))
        #0.2和0.8都是经验主义,可以自行修改
        #inplace的作用是可以减小内存
        return layers
    
    self.model = nn.Sequential(
        *block(latent_dim, 128, normalize: False),
        *block(128,256),
        *block(256,512),
        *block(512,1024),
        nn.Linear(1024,int(nn.prod(img_shape))), 
      	nn.Tanh()
      	#img_shape = (opt.channels, opt.img_size, opt.img_size)
    )
```

之后就是 $forward$ 函数：

```python
def forward(self, z):
    img = self.model(z)
    img = img.view(img.size(0), *img_shape)
    #将 img 转换为 batch_size * img_shape 的
    return img
```

### 3.2. 鉴别器 discriminator

鉴别器和生成器结构非常相似，只是反过来而已。

```python
class discriminator(nn.Module):
    def __init__(self):
        super().__init__() #调用超类 
        
    def block(in_num, out_num):  #in_idx: 上一层的大小 out_idx 下一层的大小
        layers = [nn.Linear(in_num, out_num)]
        layers.append(nn.LeakyReLU(0.2, True))
        return layers
 
    self.model = nn.Sequential(
        *block(int(nn.prod(img_shape)), 512),
      	*block(512,256),
      	nn.Linear(256,1),
        nn.Sigmoid()
    )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

