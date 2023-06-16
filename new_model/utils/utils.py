import os 
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy import stats

class Utils():
    def __init__(self,opt, diff_genes):
        self.opt = opt
        self.diff_genes = diff_genes
        matplotlib.rc('ytick', labelsize=18)
        matplotlib.rc('xtick', labelsize=18)
        sc.set_figure_params(dpi_save=300)
        sc.settings.figdir = opt.picture_save_path
    
    def adata2numpy(self, adata):
        if sparse.issparse(adata.X):
            return adata.X.A
        else:
            return adata.X

    def reg_plot(
        self,
        axs,
        adata,
        axis_keys,
        labels,
        gene_list=None,
        top_100_genes=None,
        show=False,
        legend=True,
        title=None,
        verbose=False,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        type='mean',
        **kwargs,
    ):

        sns.set()
        sns.set(color_codes=True)

        condition_key = self.opt.condition_key

        sc.tl.rank_genes_groups(
            adata, groupby=condition_key, n_genes=100, method="wilcoxon"
        )
        diff_genes = top_100_genes
        stim = self.adata2numpy(adata[adata.obs[condition_key] == axis_keys["y"]])
        ctrl = self.adata2numpy(adata[adata.obs[condition_key] == axis_keys["x"]])
        
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = self.adata2numpy(adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]])
            ctrl_diff = self.adata2numpy(adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]])

            if type == 'variance':

                x_diff = np.asarray(np.var(ctrl_diff, axis=0)).ravel()
                y_diff = np.asarray(np.var(stim_diff, axis=0)).ravel()
            else: 
                x_diff = np.asarray(np.mean(ctrl_diff, axis=0)).ravel()
                y_diff = np.asarray(np.mean(stim_diff, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("Top 100 DEGs var: ", r_value_diff**2)
        
        if "y1" in axis_keys.keys():
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        
        if type == 'variance':
            x = np.asarray(np.var(ctrl, axis=0)).ravel()
            y = np.asarray(np.var(stim, axis=0)).ravel()
        else:
            x = np.asarray(np.mean(ctrl, axis=0)).ravel()
            y = np.asarray(np.mean(stim, axis=0)).ravel()
        
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        
        if verbose:
            print("All genes var: ", r_value**2)
        
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, ax = axs)
        ax.tick_params(labelsize=fontsize)
        
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        
        if "y1" in axis_keys.keys():
            if type == 'variance':
                y1 = np.asarray(np.var(self.adata2numpy(real_stim), axis=0)).ravel()
            else:
                y1 = np.asarray(np.mean(self.adata2numpy(real_stim), axis=0)).ravel()
            _ = plt.scatter(
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
                plt.text(x_bar, y_bar, i, fontsize=11, color="black")
                plt.plot(x_bar, y_bar, "o", color="red", markersize=5)
                if "y1" in axis_keys.keys():
                    y1_bar = y1[j]
                    plt.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        
        if legend:
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
        if title is None:
            plt.title("", fontsize=12)
        
        else:
            plt.title(title, fontsize=12)
        
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.4f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.4f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        # if save:
        #     plt.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            plt.show()
        plt.close()

        return r_value**2

    def make_plots(self, adata, conditions, model_name, figure, opt, axs, x_coeff=0.3, y_coeff=0.1, num = 1):

        if model_name == "RealCD4T":
            mean_labels = {"x": "ctrl mean", "y": "stim mean"}
            var_labels = {"x": "ctrl var", "y": "stim var"}
        else:
            mean_labels = {"x": "pred mean", "y": "stim mean"}
            var_labels = {"x": "pred var", "y": "stim var"}

        scores = self.reg_plot(     
                                    axs = axs[2,0],
                                    adata=adata, 
                                    axis_keys={"x": conditions["pred_stim"], "y": conditions["real_stim"]},
                                    gene_list=self.diff_genes[:5],
                                    top_100_genes=self.diff_genes,
                                    path_to_save=os.path.join(self.opt.picture_save_path, f"Fig{num}_{figure}_{model_name}_reg_mean.pdf"),
                                    legend=False,
                                    title="",
                                    labels=mean_labels,
                                    fontsize=26,
                                    x_coeff=x_coeff,
                                    y_coeff=y_coeff,
                                    show=True,
                                    type = 'mean',
        )
        
        scores = scores * 2 + self.reg_plot(
                                    axs = axs[2,1],
                                    adata=adata, 
                                    axis_keys={"x": conditions["pred_stim"], "y": conditions["real_stim"]},
                                    gene_list=self.diff_genes[:5],
                                    top_100_genes=self.diff_genes,
                                    path_to_save=os.path.join(self.opt.picture_save_path, f"Fig{num}_{figure}_{model_name}_reg_var.pdf"),
                                    legend=False,
                                    labels=var_labels,
                                    title="",
                                    fontsize=26,
                                    x_coeff=x_coeff,
                                    y_coeff=y_coeff,
                                    save=True,
                                    type = 'variance',
                                    show=True
        )

        if model_name == "scGen":
            adata = adata[adata.obs["condition"].isin(["CD4T_ctrl", "CD4T_pred_stim", "CD4T_real_stim"])]
            adata.obs["condition"].replace("CD4T_ctrl", "ctrl", inplace=True)
            adata.obs["condition"].replace("CD4T_real_stim", "real_stim", inplace=True)
            adata.obs["condition"].replace("CD4T_pred_stim", "pred_stim", inplace=True)
        
        sc.pp.neighbors(adata, n_neighbors=20)
        sc.tl.umap(adata, min_dist=1.1)
        plt.style.use('default')
        if model_name == "scGen":
            sc.pl.umap(adata, color=["condition"],
                    legend_loc=False,
                    frameon=False,
                    title="",
                    palette=matplotlib.rcParams["axes.prop_cycle"],
                    show=False,
                    ax = axs[3,1],
                    )
        else:
            if model_name == "RealCD4T":
                sc.pl.umap(adata, color=["condition"],
                    legend_loc=False,
                    frameon=False,
                    title="",
                    palette=['#1f77b4', '#2ca02c'],
                    show=False,
                    ax = axs[3,1],
                    )
            else:
                
                sc.pl.umap(adata, color=["condition"],
                        legend_loc=False,
                        frameon=False,
                        title="",
                        palette=matplotlib.rcParams["axes.prop_cycle"],
                        show=False,
                        ax = axs[3,1],
                        )
        
        return scores / 3.0 * 100.0