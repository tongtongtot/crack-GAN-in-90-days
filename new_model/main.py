import os
import torch
import anndata
import numpy as np
import scanpy as sc
from tqdm import tqdm
from utils.utils import Utils
import matplotlib.pyplot as plt
from options.option import options
# from reportlab.pdfgen import canvas
from models.new_model import new_model
from dataloader.customDataset import customDataloader

def train_model(opt):

    model = new_model(opt)
    model.to(opt.device)

    dataset = customDataloader(opt)
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers)

    scores = 0
    best_model = 0

    for epoch in tqdm(range(opt.epochs)):
        print("Start training")
        for idx, (sti, con) in enumerate(dataloader):
            model.train()
            model.set_input(sti, con)
            model.update_parameter()
        
        if epoch % opt.save_interval == 0:
            print(epoch)
            model.save(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')
            tmp_scores = validation(opt, epoch + 1)
            if tmp_scores > scores:
                scores = tmp_scores
                best_model = epoch
                os.rename(src = opt.result_save_path + '/now_epoch.pdf', dst = opt.result_save_path + f'/best_epoch.pdf')
                os.rename(src = opt.model_save_path + "/now_epoch.h5ad", dst = opt.model_save_path + "/best_epoch.h5ad")
                model.save(opt.model_save_path + '/'  + opt.exclude_celltype + '_best_epoch.pt')

            print("result so far:")
            print(best_model)
            print(scores)
    
    return (best_model, scores)

def plot(data, num, opt, axs, diff_genes):
    print("data:", data.shape)
    draw = Utils(opt, diff_genes)
    conditions = {"real_stim": opt.stim_key, "pred_stim": opt.pred_key}
    return draw.make_plots(adata = data, conditions = conditions, model_name=opt.model_name, figure="b", x_coeff=0.45, y_coeff=0.8, num = num, opt = opt, axs = axs)

def validation(opt, num):

    print("start validation.")

    model = new_model(opt)
    model.to(opt.device)

    dataset = customDataloader(opt)
    pred_data, ctrl_data, stim_data, train, new_train= dataset.get_val_data()

    model.load(opt.model_save_path + '/'  + opt.exclude_celltype + '_now_epoch.pt')

    # gen_img = []
    fig, axs = plt.subplots(4, 2, figsize = (30, 30))
    # fig.subplots_adjust(right = 3, top = 3, hspace=0.5)
    # fig.subplots_adjust(hspace=0.5)

    predicts = model.predict(pred_data, ctrl_data, stim_data)

    cell_type = opt.exclude_celltype
    cell_type_data = train[train.obs[opt.cell_type_key] == cell_type]

    pred = anndata.AnnData(predicts, obs={opt.condition_key: [opt.pred_key] * len(predicts), opt.cell_type_key: [cell_type] * len(predicts)}, var={"var_names": cell_type_data.var_names})

    content_latent_X = model.get_z(new_train, 'content')

    latent_adata = sc.AnnData(X=content_latent_X, obs=new_train.obs.copy())

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=['condition'], wspace=0.4, frameon=False, show = False, ax = axs[0,0])
    sc.pl.umap(latent_adata, color=[opt.cell_type_key], wspace=0.4, frameon=False, show = False, ax = axs[0,1])
    # gen_img.append(latent_context_img)

    style_latent_X = model.get_z(new_train, 'style')
    latent_adata = sc.AnnData(X=style_latent_X, obs=new_train.obs.copy())

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=['condition'], wspace=0.4, frameon=False, show = False, ax = axs[1,0])
    sc.pl.umap(latent_adata, color=[opt.cell_type_key], wspace=0.4, frameon=False, show = False, ax = axs[1,1])
    # gen_img.append(latent_style_img)

    ctrl_adata = train[((train.obs[opt.cell_type_key] == cell_type) & (train.obs['condition'] == opt.ctrl_key))]
    stim_adata = train[((train.obs[opt.cell_type_key] == cell_type) & (train.obs['condition'] == opt.stim_key))]

    eval_adata = ctrl_adata.concatenate(stim_adata, pred)

    eval_adata.write_h5ad(opt.model_save_path + "/now_epoch.h5ad")

    # sc.tl.pca(eval_adata)
    # stim_data_img = sc.pl.pca(eval_adata, color="condition", frameon=False, show = False)
    # gen_img.append(stim_data_img)

    exclude_cell = train[train.obs[opt.cell_type_key] == cell_type]
    
    sc.tl.rank_genes_groups(exclude_cell, groupby="condition", method="wilcoxon")
    diff_genes = exclude_cell.uns["rank_genes_groups"]["names"][opt.stim_key]
    print(diff_genes)

    sc.pl.violin(eval_adata, keys= diff_genes[0], groupby="condition", show = False, ax = axs[3,0])
    # gen_img.append(violin_img)

    scores = plot(eval_adata, num, opt, axs, diff_genes)

    # gen_img.append(stim_img)

    # save_Image(gen_img, opt)

    fig.savefig(opt.result_save_path + '/now_epoch.pdf')

    return scores


if __name__ == '__main__':
    Opt = options()
    opt = Opt.init()
    
    torch.multiprocessing.set_start_method('spawn')

    if opt.validation == True:
        validation(opt, opt.best_model)
    else:
        best_model, scores = train_model(opt)
        print("the best model is:", best_model)
        print("and, the socore of that model is:", scores)
    # f = open(opt.save_log + '/' + opt.model_name + '.txt','w')
    # for x in accuracy:
    #     f.write(x + '\n')
    # f.close()