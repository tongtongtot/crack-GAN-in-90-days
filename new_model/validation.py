import torch
import scanpy as sc
from models.new_model import new_model
from options.option import options
import anndata
from tqdm import tqdm
from utils.utils import Utils

Opt = options()
opt = Opt.init()

train = sc.read(opt.read_path)
model = new_model(opt)

if opt.device == 'cuda':
    model.to(opt.device)

model.load_state_dict(torch.load(opt.model_save_path + '/' + str(opt.get_epoch)))

def len(data):
    return data.shape[0]

def plot(data):
    draw = Utils(opt)
    get_now_data = data
    conditions = {"real_stim": opt.stim_key, "pred_stim": opt.pred_key}
    draw.make_plots(adata = get_now_data, conditions = conditions, model_name=opt.model_name, figure="b", x_coeff=0.45, y_coeff=0.8)

for idx, cell_type in tqdm(enumerate(train.obs[opt.cell_type_key].unique().tolist())):
    cell_type_data = train[train.obs[opt.cell_type_key] == cell_type]
    cell_type_ctrl_data = train[((train.obs[opt.cell_type_key] == cell_type) & (train.obs[opt.condition_key] == opt.ctrl_key))]
    net_train_data = train[~((train.obs[opt.cell_type_key] == cell_type) & (train.obs[opt.condition_key] == opt.stim_key))]
    pred = model.predict(net_train_data, cell_type_ctrl_data, opt.condition_key, opt.stim_key, opt.ctrl_key)
    pred_adata = anndata.AnnData(pred, obs={opt.condition_key: [opt.pred_key] * len(pred), opt.cell_type_key: [cell_type] * len(pred)}, var={"var_names": cell_type_data.var_names})
    # ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X, obs={opt.condition_key: [opt.ctrl_key] * len(cell_type_ctrl_data), opt.cell_type_key: [cell_type] * len(cell_type_ctrl_data)}, var={"var_names": cell_type_ctrl_data.var_names})
    real_stim = cell_type_data[cell_type_data.obs[opt.condition_key] == opt.stim_key].X
    real_stim_adata = anndata.AnnData(real_stim, obs={opt.condition_key: [opt.stim_key] * len(real_stim), opt.cell_type_key: [cell_type] * len(real_stim)}, var={"var_names": cell_type_data.var_names})
    if idx == 0:
        all_data = pred_adata.concatenate(real_stim_adata)
    else:
        all_data = all_data.concatenate(pred_adata, real_stim_adata)

all_data.write_h5ad(opt.result_save_path + '/' + opt.model_name + ".h5ad")

plot(all_data)