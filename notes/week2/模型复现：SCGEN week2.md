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



接下来要做的事情就很简单了，只需要顺着上面的步骤往里面添加东西即可：

### 1 options

```python
import sys
import os
import argparse
import torch

class scgen_rewrite_options():      
    def get_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, default="baseline_addbalance", help="The name of the current model.")
        parser.add_argument("--read_path", type = str, default = './data/train_pbmc.h5ad', help = "The path of the data.")
        parser.add_argument("--loss_save_path", type=str,default= 'saved_loss/saved_loss_changeloss_', help = "The path to save the loss.")
        parser.add_argument("--model_save_path", type = str, default = 'saved_model/saved_model_changeloss_', help = "The path of saved model")
        parser.add_argument("--picture_save_path", type = str, default = 'saved_picture/saved_picture_', help = "The path of saved picture")
        parser.add_argument("--backup_url", type = str, default = 'https://drive.google.com/uc?id=1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk')
        parser.add_argument("--batch_size", type = int, default = 6400, help = "This is the batch size.")
        parser.add_argument("--lr", type = float, default = 1e-3, help = "This is the learning rate.")
        parser.add_argument("--hidden_layer", type = int, default = 800, help = "This is the size of the hidden layer.")
        parser.add_argument("--latent_layer", type = int, default = 100, help = "This is the size of the latent layer.")
        parser.add_argument("--input_layer", type = int, default = 6998, help = "This is the size of the input layer.")
        parser.add_argument("--type", type = str, default = 'cell_type', help = "This is the type of labels that we want to balance")
        # parser.add_argument("--n_layers", type = int, default = 2, help = "This is the number of layers.")
        parser.add_argument("--epochs", type = int, default = 10, help = "This is the number of epochs.")
        parser.add_argument("--drop_out", type = float, default = 0.2, help = "This is the drop out rate.")
        parser.add_argument("--save_interval", type = int, default = 100, help = "Save model every how many epochs.")
        parser.add_argument("--shuffle", type = bool, default = True, help = "Whether to shuffle the input data or not.")
        parser.add_argument("--exclude", type= bool, default = True, help="Whether to exclude some of the cell type or not.")
        parser.add_argument("--exclude_celltype", type=str, default="CD4T", help="The type of the cell that is going to be excluded.")
        parser.add_argument("--exclude_condition", type=str, default="stimulated", help="The condition of the cell that is going to be excluded.")
        parser.add_argument("--training", type=bool, default=True, help="Whether training or not.")
        opt = parser.parse_args()
        return opt

```

### 2 Dataloader

```python
# from options.opt import opt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import scanpy as sc
import torch.utils.data as data
import numpy as np
import random
from scipy import sparse

class customDataloader(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.train = sc.read(self.opt.read_path)
        self.train = self.balance(self.train)
        self.condition_mask = (self.train.obs['condition'] == 'stimulated')
        self.sti = self.train[self.condition_mask].copy()
        self.con = self.train[~self.condition_mask].copy()
        self.sti_len = len(self.sti)
        self.con_len = len(self.con)
        self.max_len = max(len(self.sti), len(self.con))

    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data).cuda()
        else:
            Exception("This is not a numpy")
        return data

    def tensor2numpy(self, data):
        data = data.cpu().detach().numpy()
        return data

    def adata2numpy(self, adata):
        if sparse.issparse(adata.X):
            return adata.X.A
        else:
            return adata.X
    
    def adata2tensor(self, adata):
        return self.numpy2tensor(self.adata2numpy(adata))
    
    def __getitem__(self, idx):       
        sti_idx = random.randint(0, self.sti_len - 1) if idx >= self.sti_len else idx
        con_idx = random.randint(0, self.con_len - 1) if idx >= self.con_len else idx
        sti = self.sti[sti_idx]
        con = self.con[con_idx]
        return (self.adata2tensor(sti), self.adata2tensor(con))

    def __len__(self):
        return self.max_len

    def balance(self, adata):
        cell_type = adata.obs[self.opt.type]

        class_num = np.unique(cell_type)
        type_num = {}
        max_num = -1
        for i in class_num:
            type_num[i] = cell_type[cell_type == i].shape[0]
            max_num = max(max_num, type_num[i])
        
        index_add = []
        for i in class_num:
            class_index = np.array(cell_type == i)
            index_cls = np.nonzero(class_index)[0]
            index_cls = index_cls[np.random.choice(len(index_cls), max_num)]
            index_add.append(index_cls)

        balanced_data = adata[np.concatenate(index_add)].copy()
        return balanced_data

```

### 3 model

```python
import torch
from torch import nn
from models.scgen_vae import SCGENVAE
import numpy as np
from matplotlib import pyplot
from scipy import stats
import pandas as pd
import os
import torch.nn.functional as F
from adjustText import adjust_text
from scipy import sparse
import scanpy as sc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
gpus = [0, 1]


class SCGEN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cirterion = nn.MSELoss()
        self.model = SCGENVAE().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.lr)
        self.loss_stat = {}

    def set_input(self, sti, con):
        b,c,g = sti.shape
        self.sti = sti.view(b, g).to(device)
        b,c,g = con.shape
        self.con = con.view(b,g).to(device)

    def forward(self, _input):
        self.model.train()
        self.outputs, self.mu, self.logvar = self.model(_input)
        return self.outputs, self.mu, self.logvar

    def compute_loss(self):
        L2loss = self.cirterion(self.outputs, self.sti)
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        self.loss = L2loss + KLD * 0.005
        print("loss:", self.loss)
        print("l1loss:", L2loss)
        print("KLD", KLD)
        self.loss_stat = {
            "loss": self.loss.item(),
            "L1loss": L2loss.item(),
            "KLD": KLD.item()
        }
    
    def get_rsqure(self):
        pred = self.tensor2numpy(self.outputs)
        truth = self.tensor2numpy(self.sti)
        x_diff = np.asarray(np.mean(pred, axis=0)).ravel()
        y_diff = np.asarray(np.mean(truth, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x_diff, y_diff)
        return r_value ** 2

    def get_current_loss(self):
        return self.loss_stat

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def update_parameter(self):
        print("get_rsqure")
        self.forward(self.con)
        self.compute_loss()
        self.backward()
        print(self.get_rsqure())
        self.forward(self.sti)
        self.compute_loss()
        self.backward()
        print(self.get_rsqure())    

    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray): 
            data = torch.from_numpy(data).to(device)
        else:
            Exception("This is not a numpy")
        return data

    def tensor2numpy(self, data):
        data = data.cpu().detach().numpy()
        return data

    def adata2numpy(self, adata):
        if sparse.issparse(adata.X):
            return adata.X.A
        else:
            return adata.X
    
    def adata2tensor(self, adata):
        return self.numpy2tensor(self.adata2numpy(adata))

    def _avg_vector(self, adata):
        data = self.adata2numpy(adata)
        tensor_data = self.numpy2tensor(data)
        print(data.shape)
        latent, _, __ = self.model.get_z(tensor_data)
        latent_avg = np.average(self.tensor2numpy(latent), axis=0)
        return latent_avg

    def predict(self, adata, pred_data, condition_key, stim_key, ctrl_key, cell_type, cell_type_key):
        ctrl_x = adata[adata.obs[condition_key] == ctrl_key, :]
        stim_x = adata[adata.obs[condition_key] == stim_key, :]
        latent_ctrl = self._avg_vector(ctrl_x)
        latent_stim = self._avg_vector(stim_x)
        delta = latent_stim - latent_ctrl
        delta = self.numpy2tensor(delta)
        tensor_train_x = self.adata2tensor(pred_data)
        latent_cd, _, __ = self.model.get_z(tensor_train_x)
        stim_pred = delta + latent_cd
        print(stim_pred.shape)
        gen_img = self.model.decoder(stim_pred)
        return self.tensor2numpy(gen_img), self.tensor2numpy(delta)

    def reg_mean_plot(
        self,
        adata,
        axis_keys,
        labels,
        path_to_save="./reg_mean.pdf",
        save=True,
        gene_list=None,
        show=False,
        top_100_genes=None,
        verbose=False,
        legend=True,
        title=None,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
    ):
        """
        Plots mean matching figure for a set of specific genes.

        Parameters
        ----------
        adata: `~anndata.AnnData`
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            corresponding to batch and cell type metadata, respectively.
        axis_keys: dict
            Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
             `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
        labels: dict
            Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
        path_to_save: basestring
            path to save the plot.
        save: boolean
            Specify if the plot should be saved or not.
        gene_list: list
            list of gene names to be plotted.
        show: bool
            if `True`: will show to the plot after saving it.
        Examples
        --------
        >>> import anndata
        >>> import scgen
        >>> import scanpy as sc
        >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
        >>> scgen.SCGEN.setup_anndata(train)
        >>> network = scgen.SCGEN(train)
        >>> network.train()
        >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
        >>> pred, delta = network.predict(
        >>>     adata=train,
        >>>     adata_to_predict=unperturbed_data,
        >>>     ctrl_key="control",
        >>>     stim_key="stimulated"np.asarray(np.mean(ctrl_diff.X, axis=0)).ravel()
        >>>     show=False
        >>> )
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        condition_key = 'condition'

        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = np.asarray(np.mean(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.mean(stim_diff.X, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("top_100 DEGs mean: ", r_value_diff**2)
        x = np.asarray(np.mean(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.mean(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes mean: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
                force_points=(0.0, 0.0),
            )
        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=fontsize)
        else:
            pyplot.title(title, fontsize=fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )
        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2
    
    def reg_var_plot(
        self,
        adata,
        axis_keys,
        labels,
        path_to_save="./reg_var.pdf",
        save=True,
        gene_list=None,
        top_100_genes=None,
        show=False,
        legend=True,
        title=None,
        verbose=False,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
    ):
        """
        Plots variance matching figure for a set of specific genes.

        Parameters
        ----------
        adata: `~anndata.AnnData`
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
            corresponding to batch and cell type metadata, respectively.
        axis_keys: dict
            Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
             `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
        labels: dict
            Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
        path_to_save: basestring
            path to save the plot.
        save: boolean
            Specify if the plot should be saved or not.
        gene_list: list
            list of gene names to be plotted.
        show: bool
            if `True`: will show to the plot after saving it.

        Examples
        --------
        >>> import anndata
        >>> import scgen
        >>> import scanpy as sc
        >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
        >>> scgen.SCGEN.setup_anndata(train)
        >>> network = scgen.SCGEN(train)
        >>> network.train()
        >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
        >>> pred, delta = network.predict(
        >>>     adata=train,
        >>>     adata_to_predict=unperturbed_data,
        >>>     ctrl_key="control",
        >>>     stim_key="stimulated"
        >>>)
        >>> pred_adata = anndata.AnnData(
        >>>     pred,
        >>>     obs={"condition": ["pred"] * len(pred)},
        >>>     var={"var_names": train.var_names},
        >>>)
        >>> CD4T = train[train.obs["cell_type"] == "CD4T"]
        >>> all_adata = CD4T.concatenate(pred_adata)
        >>> network.reg_var_plot(
        >>>     all_adata,
        >>>     axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
        >>>     gene_list=["ISG15", "CD3D"],
        >>>     path_to_save="tests/reg_var4.pdf",
        >>>     show=False
        >>>)
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        condition_key = 'condition'

        sc.tl.rank_genes_groups(
            adata, groupby=condition_key, n_genes=100, method="wilcoxon"
        )
        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            print("stim&ctrl")
            print(stim_diff.shape)
            print(ctrl_diff.shape)
            x_diff = np.asarray(np.var(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.var(stim_diff.X, axis=0)).ravel()
            print("x&y")
            print(x_diff)
            print(y_diff)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("Top 100 DEGs var: ", r_value_diff**2)
        if "y1" in axis_keys.keys():
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        x = np.asarray(np.var(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.var(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes var: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        # _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
        # pyplot.plot(x, m * x + b, "-", color="green")
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if "y1" in axis_keys.keys():
            y1 = np.asarray(np.var(real_stim.X, axis=0)).ravel()
            _ = pyplot.scatter(
                x,
                y1,
                marker="*",
                c="grey",
                alpha=0.5,
                label=f"{axis_keys['x']}-{axis_keys['y1']}",
            )
        if gene_list is not None:
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                pyplot.text(x_bar, y_bar, i, fontsize=11, color="black")
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
                # 
                if "y1" in axis_keys.keys():
                    y1_bar = y1[j]
                    pyplot.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=12)
        else:
            pyplot.title(title, fontsize=12)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2

```

### 4 real model

```python
import torch
from torch import nn
from options.opt import opt

class SCGENVAE(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = opt.input_layer
        hidden_dim = opt.hidden_layer
        latent_dim = opt.latent_layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mu_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.logvar_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.img_size = input_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

    def get_z(self, x):
        hidden = self.encoder(x)
        mu = self.mu_encoder(hidden)
        logvar = self.logvar_encoder(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.get_z(x)
        return self.decoder(z), mu, logvar
```

### 5 train

```python
# from options.opt import opt
from options.option import scgen_rewrite_options
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import scanpy as sc
import sys
import os
import argparse
from dataloader.customDataset import customDataloader
from models.scgen_model import SCGEN
import torch.nn as nn
from tqdm import tqdm

opt = scgen_rewrite_options.get_opt()

os.makedirs("saved_model", exist_ok = True)
os.makedirs("saved_picture", exist_ok = True)
os.makedirs("saved_loss", exist_ok = True)

print("opt", opt)

dataset = customDataloader(opt)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = opt.shuffle)

model = SCGEN(opt)

gpus = [0, 1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if device == 'cuda':
    model.cuda()
    # model = nn.DataParallel(model.to(device), device_ids = gpus, output_device = gpus[0])
loss_all = []


for epoch in tqdm(range(opt.epochs)):

    for idx, (sti, con) in enumerate(dataloader):
        model.train()
        model.set_input(sti, con)
        model.update_parameter()
    
    if epoch % opt.save_interval == opt.save_interval - 1:
        torch.save(model.state_dict(), opt.model_save_path + str(epoch + 1))
        print(opt.model_save_path + str(epoch + 1))
    

with open(opt.loss_save_path + opt.model_name + '.txt', 'w') as file:
    for idx, loss in enumerate(loss_all):
        file.write(str(loss) + '\n')

```

### 6 validation

```python
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchvision.utils import save_image
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
import scanpy as sc
from models.scgen_model import SCGEN
from dataloader.customDataset import customDataloader
from options.opt_val import opt
import anndata
from collections import OrderedDict
from tqdm import tqdm

print(opt)
train = sc.read("data/train_pbmc.h5ad")
model = SCGEN(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if device == 'cuda':
    model.to(device)
    gpus = [0,1]
    # model = nn.DataParallel(model.to(device), device_ids = gpus, output_device = gpus[0])

model.load_state_dict(torch.load(opt.model_save_path + str(opt.get_epoch)))

stim_key = "stimulated"
ctrl_key = "control"
cell_type_key = "cell_type"
condition_key="condition"
data_name = 'scgen_rewrite'

def len(data):
    return data.shape[0]

os.makedirs("./data/reconstructed/scGen/", exist_ok=True)

for idx, cell_type in tqdm(enumerate(train.obs[cell_type_key].unique().tolist())):
    cell_type_data = train[train.obs[cell_type_key] == cell_type]
    cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
    net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
    print(cell_type_data.shape)
    print(cell_type_ctrl_data)
    print(net_train_data.shape)
    print("pre operation done.")
    pred, delta = model.predict(net_train_data, cell_type_ctrl_data, condition_key, stim_key, ctrl_key, cell_type, cell_type_key)
    print("model done")
    # print(pred.shape)
    # print(delta)
    print(pred.shape[0])
    pred_adata = anndata.AnnData(pred, obs={condition_key: [f"pred"] * len(pred), cell_type_key: [cell_type] * len(pred)}, var={"var_names": cell_type_data.var_names})
    print("This is data:")
    print(pred[:10,:10])
    ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X, obs={condition_key: [f"contorl"] * len(cell_type_ctrl_data), cell_type_key: [cell_type] * len(cell_type_ctrl_data)}, var={"var_names": cell_type_ctrl_data.var_names})
    real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
    real_stim_adata = anndata.AnnData(real_stim, obs={condition_key: [f"stimulated"] * len(real_stim), cell_type_key: [cell_type] * len(real_stim)}, var={"var_names": cell_type_data.var_names})
    if idx == 0:
        all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        # all_data.write_h5ad(f"./data/reconstructed/scGen/{data_name}.h5ad")
    else:
        all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)
        # all_data.write_h5ad(f"./data/reconstructed/scGen/{data_name}.h5ad")
    print("all_data:", all_data.shape)

all_data.write_h5ad(f"./data/reconstructed/scGen/{data_name}.h5ad")

```