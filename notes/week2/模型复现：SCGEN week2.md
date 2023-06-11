# 模型复现：SCGEN

### 1 scgen 是如何运作的？

[Scgen](https://github.com/theislab/scgen) 是一个用于预测细胞扰动的神经网络，结构如下：![sketch](/Users/tongtongtot/Desktop/crack-GAN-in-90-days/notes/week2/sketch.png)

更详细地说：分为以下几个步骤：

1. 数据预处理：

    包括将数据的筛选，数据的提取，数量归一化等操作。

2. 导入模型

    $scgen$ 使用了 $VAE$ 的方法作为核心，并增加了一些向量操作优化了模型效果。

3. 比较与训练

    $scgen$ 使用了 $L1Loss$ 和 $KL$散度 来作为它的 $loss$, 默认训练 1000 个 $epochs$， $batch_-size = 32$, $dropout_-rate = 0.2,\space learning_-rate= 0.001$ 

### 2 scgen 的模型是怎样的？

观摩别人模型的重点是**模型的输入和输出**

在 $scgen$ 模型中, 输入和输出的数据为$AnnData$, 一种类似矩阵的数据，如下所示：

```jupyter
AnnData object with n_obs × n_vars = 13766 × 6998
    obs: 'condition', 'n_counts', 'n_genes', 'mt_frac', 'cell_type', '_scvi_batch', '_scvi_labels'
    var: 'gene_symbol', 'n_cells'
    uns: 'cell_type_colors', 'condition_colors', 'neighbors', '_scvi_uuid', '_scvi_manager_uuid'
    obsm: 'X_pca', 'X_tsne', 'X_umap'
    obsp: 'distances', 'connectivities'
```

所以第一步就是要处理这个数据：

首先是从文件中读取数据：

```python
import scanpy as sc
train = sc.read(path)
```

然后需要从中去除一种细胞的数据来进行训练和预测：

```python
cell_type_ctrl_data = train[((train.obs['cell_type'] == cell_type) & (train.obs['condition'] == 'control'))]
net_train_data = train[~((train.obs['cell_type'] == cell_type) & (train.obs['condition'] == 'stimulated'))]
```

然后就可以输入到模型之中：

```python
pred, delta = network.predict(
  						adata=net_train_data, 
  						conditions={"ctrl": ctrl_key, "stim": stim_key}, 
  						cell_type_key=cell_type_key,
  						condition_key=condition_key, 
  						celltype_to_predict=cell_type)
```

之后就直接输入 $VAE$ 就可以了, $Adata$ 可以直接输入。

$VAE$ 会返回一个 $loss$, 该 $loss = \alpha \times L1Loss + \beta \times BCELoss$.

$loss$ 仍然是加和的形式，所以 $loss$ 也会很大。

最后就通过 $adam$ 来优化这个 $loss$.

### 3 prediction

首先输入一个 $AnnData$ 数据，然后对数据做一次 $balance$ 操作。

***注：由于原数据中每一种 cell_type 的数量不一致，为了增加模型效果(?)， 所以增加了一个balance 操作来讲每一种 cell_type 的数据的数量归一化。具体操作方式为：随机找 maxlen (cell_type数量最多的那一个的长度) 个数据 (可能重复)。 之后还有一次这样的操作，但是 是因为标签的是 condition和stimulated数量不同***

之后就要把这个数据拆成两份，一份是 $condtion$, 另一份是 $stimulated$. $condtion$ 是正常情况下的细胞，而 $stimulated$ 是有扰动之后的细胞的图像。

之后把$stimulated$ 和 $condition$ 投影到隐空间并获取它们之间的差，称为$\delta$。

之后就把 $to_-latent(condition)\space + \space \delta$ 作为 $Z$ 输入 $VAE$ 得到结果。

结果也是一个 $AnnData$ 数据

# 如何复现别人的模型？

方式非常简单，用的是 $ablation\space study$ 的思路，也就是先实现一个最基本的步骤，然后一步一步往中间加内容。

### 写在前面

由于是第一次复现模型，在这里写一个模板方便之后构建模型：

首先是 $option.py$：

在这个文件中，我们会输入一些在模型中会用的参数：

```python
import os
import argparse
import torch.nn as nn

class options():
		def get_opt():
				self.parser = argparse.ArgumentParser()
				self.parser.add_argument("")
        self.init()
				opt = self.parser.parse_args()
				return opt
    
    def __init__():
      	# 将所有以下定义的函数都调用一遍
    
    '''
    def check_gpu():
				get_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parser("--device", type = torch.device(), default = get_device, help = "Which device to use")
```

之后是$model.py$:

```python
from torch import nn
import numpy as np
import scanpy as sc
from real_model import MODEL

class model_name(nn.Module):
		def __init__(self, opt):
				super().__init__()
				self.opt = opt
				self.model = MODEL(opt)
				self.cirterion = loss_function()
				self.optimizer = optimizer(self.model.parameters(), opt.lr)
		
		def set_input(self, input):
				self.input = input
		
		def comput_loss(self):
				self.loss = self.cirterion(self.output, self.groundtruth)
		
    def get_loss(self):
      	return self.loss
    
		def forward(self, input):
				self.model.train()
				self.outputs = self.model(input)
				return self.outputs
		
		def backward(self):
				self.optimzer.zero_grad()
				self.loss.backward()
				self.optimizer.step()
		
		def update_parameter(self):
				self.forward(self.input)
				self.compute_loss()
				self.backward()
```

在之后是$real_-model.py$ ,也就是模型的部分，这里按照模型不同可能有差异，这里就先不写出了。

然后是$dataloader$：

```python
import torch
import torch.uitls.data as data

class customDataset(data.Dataset):
		def __init__(self, opt):
				super().__init__()
				self.opt = opt
				self.train = load_data(opt.train_data_path)
				self.val = load_data(opt.val_data_path)
		
		def __getitem__(self, idx):
				if self.opt.train == True:
						return self.train[idx]
				else:
						return self.val[idx]
		
		def __len__(self):
				if self.opt.train == True:
						return len(self.train)
				else:
						return len(self.val)
```

最后是$training$ 和 $validation$ 的部分：

```python
from options import options
from model import model_name
import torch.uitls.data

Opt = options()
opt = Opt.get_opt()

dataset = customDataloader(opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = opt.shuffle)

model = model_name(opt)

if opt.device == 'cuda':
		model.to(opt.device)
if opt.parallel == 'True':
		model = nn.Parallel(model.to(opt.device), device_ids = opt.device_id, output_device = opt.output_device)

for epoch in range(opt.epochs):
		
    with tqdm(total=opt.epoch) as t:
        
        for idx, input in enumerate(dataloader):
            model.train()
            model.set_input(input)
            model.update_parameter()
            
        if epoch % opt.save_interval == opt.save_interval - 1:
          	torch.save(model.state_dict(), opt.model_save_path + str(epoch + 1))
        
        t.set_description("进度条左边文字")
        t.set_postfix(loss = model.get_loss())
        sleep(0.1)
        t.update(1)	
```



### 1 training

这里的$training$相当简单，就是直接将数据放入$VAE$ 中训练：

因此我们就 