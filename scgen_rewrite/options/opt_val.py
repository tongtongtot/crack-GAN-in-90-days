import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--read_path", type = str, default = './tests/data/train_pbmc.h5ad', help = "The path of the data.")
parser.add_argument("--model_save_path", type = str, default = 'saved_model/saved_model_changeloss_', help = "The path of saved model")
parser.add_argument("--get_epoch", type=int,default=100, help = "Which model will load.")
parser.add_argument("--picture_save_path", type = str, default = 'saved_picture/saved_picture_', help = "The path of saved picture")
parser.add_argument("--backup_url", type = str, default = 'https://drive.google.com/uc?id=1r87vhoLLq6PXAYdmyyd89zG90eJOFYLk')
parser.add_argument("--batch_size", type = int, default = 32, help = "This is the batch size.")
parser.add_argument("--lr", type = float, default = 1e-3, help = "This is the learning rate.")
parser.add_argument("--hidden_layer", type = int, default = 800, help = "This is the size of the hidden layer.")
parser.add_argument("--latent_layer", type = int, default = 100, help = "This is the size of the latent layer.")
parser.add_argument("--input_layer", type = int, default = 6998, help = "This is the size of the input layer.")
parser.add_argument("--type", type = str, default = 'cell_type', help = "This is the type of labels that we want to balance")
parser.add_argument("--drop_out", type = float, default = 0.2, help = "This is the drop out rate.")
parser.add_argument("--save_interval", type = int, default = 1, help = "Save model every how many epochs.")
parser.add_argument("--shuffle", type = bool, default = False, help = "Whether to shuffle the input data or not.")
parser.add_argument("--exclude", type= bool, default = True, help="Whether to exclude some of the cell type or not.")
parser.add_argument("--exclude_celltype", type=str, default="CD4T", help="The type of the cell that is going to be excluded.")
parser.add_argument("--exclude_condition", type=str, default="stimulated", help="The condition of the cell that is going to be excluded.")
opt = parser.parse_args()
# print("This is opt output", opt)