import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms

class customDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.path = opt.path
        txt_file = open(os.path.join(self.path, "label.txt"), "r")
        lines = txt_file.readlines()
        self.img_path = []
        self.label = []
        self.transform = transforms.ToTensor()
        # line = lines.strip().split(' ')
        for line_idx, line in enumerate(lines):
            line = line.strip().split(' ')
            
            # if(int(line[1]) != 1): 
            #     continue
            
            self.img_path.append(line[0])
            self.label.append(line[1])
        
        # print(len(self.label))
        # # import pdb
        # # pdb.set_trace()
        # exit(0)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        # H W C -> C H W 
        img = img.transpose((2,0,1))
        return {"img": img, "label": self.label[idx]}

    def __len__(self):
        return len(self.img_path)
