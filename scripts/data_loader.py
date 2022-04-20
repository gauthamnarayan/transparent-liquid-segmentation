import os

import cv2
import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader

import utils

class TransparentPouringSegmentationDataset(Dataset):

    def __init__(self, dataDir, img_size=(150, 300), transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(os.path.join(self.dataDir, "trainA"))
        self.datasetSize = len(imageFileNames)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.datasetSize
        
    def __getitem__(self,idx):
        rgbImage = cv2.imread(os.path.join(self.dataDir, "fakeB", "rgb_" + str(idx) + ".png"))
        rgbImage = cv2.resize(rgbImage, self.img_size)
        rgbImage = np.rollaxis(rgbImage, 2, 0)

        maskImage = cv2.imread(os.path.join(self.dataDir, "trainA_liquid_masks", "liquid_mask_" + str(idx) + ".jpg"), cv2.IMREAD_GRAYSCALE)
        maskImage[maskImage>0] = 1
        maskImage = cv2.resize(maskImage, self.img_size)

        if(rgbImage is  None):
            print("rgb none at ", self.rgbFileIDs[idx])
        if(maskImage is None):
            print("mask none at ", self.rgbFileIDs[idx])

        return rgbImage, maskImage 
