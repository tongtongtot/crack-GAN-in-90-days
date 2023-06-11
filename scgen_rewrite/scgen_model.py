import torch
from torch import nn
from opt import opt
from scgen_vae import SCGENVAE
import numpy as np
from matplotlib import pyplot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
gpus = [0, 1]

class SCGEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cirterion = nn.L1Loss()
        self.model = SCGENVAE().to(device)
        self.model = nn.DataParallel(self.model.to(device), device_ids = gpus, output_device = gpus[0])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), opt.lr)
        self.loss_stat = {}

    def set_input(self, sti, con):
        b,c,g = sti.shape
        self.sti = sti.view(b, g).to(device)
        b,c,g = con.shape
        self.con = con.view(b,g).to(device)
        # print(sti.shape)
        # print(con.shape)

    def forward(self):
        # latent_con = self._avg_vector(con)
        # latent_sti = self._avg_vector(sti)
        # delta = latent_sti - latent_con

        # self.outputs = self.model(con, sti)
        self.model.train()
        self.outputs, self.mu, self.logvar = self.model(self.con)
        # self.outputs, self.mu, self.logvar = self.model(self.con, self.sti)
    
    def predict(self):
        self.model.eval()
        outputs, mu, logvar = self.model(self.con)
        # outputs, mu, logvar = self.model(self.con, self.sti)
        return outputs, mu, logvar

    def compute_loss(self):
        # print("outputs_size", self.outputs.shape)
        # print("sti_size", self.sti.shape)
        self.loss = self.cirterion(self.outputs, self.sti)
        self.loss = self.loss - 0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        self.loss_stat = {"loss": self.loss}
    
    def get_current_loss(self):
        return self.loss_stat

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def update_parameter(self):
        self.forward()
        self.compute_loss()
        self.backward()

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
        >>>     stim_key="stimulated"
        >>>)
        >>> pred_adata = anndata.AnnData(
        >>>     pred,
        >>>     obs={"condition": ["pred"] * len(pred)},
        >>>     var={"var_names": train.var_names},
        >>>)
        >>> CD4T = train[train.obs["cell_type"] == "CD4T"]
        >>> all_adata = CD4T.concatenate(pred_adata)
        >>> network.reg_mean_plot(
        >>>     all_adata,
        >>>     axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
        >>>     gene_list=["ISG15", "CD3D"],
        >>>     path_to_save="tests/reg_mean.pdf",
        >>>     show=False
        >>> )
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key

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

        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key

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
            x_diff = np.asarray(np.var(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.var(stim_diff.X, axis=0)).ravel()
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


