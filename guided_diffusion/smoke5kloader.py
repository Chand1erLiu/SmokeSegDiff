import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class SMOKE5KDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode ='train', plane = False):


        # df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
        # self.name_list = df.iloc[:,0].tolist()
        # self.label_list = df.iloc[:,1].tolist()
        self.name_list = os.listdir(os.path.join(data_path, mode, 'img'))
        self.data_path = data_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        # name = self.name_list[index]+'.jpg'
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'img', name)
        
        mask_name = name.split('.')[0] + '.png'
        msk_path = os.path.join(self.data_path, self.mode, 'gt', mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        if self.mode == 'train':
            return (img, mask, name) # name: xxx.jpg 
        else:
            return (img, mask, name)
            # return (img, mask)