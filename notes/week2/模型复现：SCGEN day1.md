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