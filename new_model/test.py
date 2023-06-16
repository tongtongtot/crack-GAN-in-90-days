import scanpy as sc

train = sc.read("pbmc.h5ad")
cell_type = 'CD4T'
tmp = train[train.obs['condition'] == 'pred']
print(tmp.obs['condition'])