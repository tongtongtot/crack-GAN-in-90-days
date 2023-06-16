import os
import torch
import argparse

class options():      
    def init(self):
        self.opt = self.get_opt()
        self.make_dic()
        self.check_device()
        return self.opt
    
    def make_dic(self):
        opt = self.opt
        os.makedirs(opt.save_path, exist_ok=True)
        self.get_save_path()
        os.makedirs(opt.model_save_path, exist_ok=True)
        # os.makedirs(opt.picture_save_path, exist_ok=True)
        os.makedirs(opt.result_save_path, exist_ok=True)
        # os.makedirs(opt.save_log, exist_ok=True)

    def check_device(self):
        if torch.cuda.is_available():
            self.opt.device = 'cuda:' + self.opt.gpu

    def get_save_path(self):
        opt = self.opt
        opt.loss_save_path = './' + opt.save_path + '/loss_' + opt.model_name
        opt.model_save_path = './' + opt.save_path + '/model_' + opt.model_name
        opt.picture_save_path = './' + opt.save_path + '/pic_' + opt.model_name
        opt.result_save_path = './' +opt.save_path + '/res_' + opt.model_name
        opt.save_log = './' + opt.save_path + '/logs'
        self.opt = opt

    def get_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, default="baseline", help="The name of the current model.")
        parser.add_argument("--training", type=bool, default=True, help="Whether training or not.")

        parser.add_argument("--read_path", type = str, default = './data/train_pbmc.h5ad', help = "The path of the data.")
        parser.add_argument("--backup_url", type = str, default = 'https://drive.google.com/uc?id=1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk')
        parser.add_argument("--save_path", type=str, default='saved', help="The folder that stores the saved things.")
        parser.add_argument("--loss_save_path", type=str,default= './saved/saved_loss/saved_loss_bassline_', help = "The path to save the loss.")
        parser.add_argument("--model_save_path", type = str, default = './saved/saved_model/saved_model_changeloss_', help = "The path of saved model")
        parser.add_argument("--picture_save_path", type = str, default = './saved/saved_picture/saved_picture_', help = "The path of saved picture")
        parser.add_argument("--result_save_path", type = str, default = './saved/result/result_', help = "The path of saved result")
        parser.add_argument("--save_log", type=str, default='saved/saved_log/log_', help='The path to save the log.')
        
        parser.add_argument("--gpu", type=str, default='0', help= "Which gpus to use.")
        parser.add_argument("--batch_size", type = int, default = 64, help = "This is the batch size.")
        parser.add_argument("--num_workers", type = int, default = 4, help= "How many cpus will try to load the data into the model.")
        parser.add_argument("--lr", type = float, default = 1e-3, help = "This is the learning rate.")
        parser.add_argument("--style_latent_dim", type = int, default = 2, help = "This is the size of the style hidden layer.")
        parser.add_argument("--context_latent_dim", type = int, default = 100, help = "This is the size of the context hidden layer")
        parser.add_argument("--hidden_dim", type = int, default = 800, help = "This is the size of the latent layer.")
        parser.add_argument("--input_dim", type = int, default = 6998, help = "This is the size of the input layer.")
        parser.add_argument("--drop_out", type = float, default = 0.2, help = "This is the drop out rate.")
        parser.add_argument("--alpha", type = float, default = 0.005, help = "This is the parameter before KLD loss")
        parser.add_argument("--beta", type = float, default = 0.005, help = "This is the parameter before KLD loss")
        parser.add_argument("--epochs", type = int, default = 200, help = "This is the number of epochs.")
        
        parser.add_argument("--save_interval", type = int, default = 5, help = "Save model every how many epochs.")
        parser.add_argument("--shuffle", type = bool, default = True, help = "Whether to shuffle the input data or not.")

        parser.add_argument("--exclude", type= bool, default = True, help="Whether to exclude some of the cell type or not.")
        parser.add_argument("--exclude_celltype", type=str, default="CD4T", help="The type of the cell that is going to be excluded.")
        parser.add_argument("--exclude_condition", type=str, default="stimulated", help="The condition of the cell that is going to be excluded.")
        
        parser.add_argument("--get_epoch", type=int, default=10, help="Which model to load.")
        parser.add_argument("--stim_key", type=str, default="stimulated", help="This is the stimulation key.")
        parser.add_argument("--pred_key", type=str, default="pred", help="This is the prediction key.")
        parser.add_argument("--ctrl_key", type=str, default="control", help="This is the control key.")
        parser.add_argument("--condition_key", type=str, default="condition", help="This is the condition key.")
        parser.add_argument("--cell_type_key", type = str, default="cell_type", help="This is the cell type key.")

        parser.add_argument("--validation", type=bool, default=False, help="Whether it is validation or not.")
        parser.add_argument("--best_model", type=int, default=1, help="Which model to load")
        # parser.add_argument("--img_size", type=[], default=[300,300], help="The size of each picture in the result.")

        parser.add_argument("--device", type = str, default = 'cpu', help = "Which device to use.")
        self.opt = parser.parse_args()
        return self.opt
