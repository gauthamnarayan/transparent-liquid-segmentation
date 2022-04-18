import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import random
import math
import copy
import utils
import albumentations as A


class TransparentPouringSegmentationDataset(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "transparent" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[2] for x in self.rgbFileIDs]
        random.shuffle(self.rgbFileIDs)
        self.datasetSize = len(self.rgbFileIDs)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self, idx):
        rgbImage = cv2.imread(
            self.dataDir + "/rgb_transparent_" + str(self.rgbFileIDs[idx]) + ".jpg")
        rgbImage = cv2.resize(rgbImage, (100, 200))
        rgbImage = np.rollaxis(rgbImage, 2, 0)
        maskImage = cv2.imread(self.dataDir + "/liquidMask_" +
                               str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        maskImage[maskImage > 1] = 0
        depthImage = np.zeros_like(maskImage)

        if(rgbImage is None):
            print("rgb none at ", self.rgbFileIDs[idx])
        if(depthImage is None):
            print("depth none at ", self.rgbFileIDs[idx])
        if(maskImage is None):
            print("mask none at ", self.rgbFileIDs[idx])

        return rgbImage, depthImage, maskImage


class TransparentPouringSegmentationDataset_verticalCrop(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "transparent" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[2] for x in self.rgbFileIDs]
        random.shuffle(self.rgbFileIDs)
        self.datasetSize = len(self.rgbFileIDs)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self, idx):
        rgbImage = cv2.imread(
            self.dataDir + "/rgb_transparent_" + str(self.rgbFileIDs[idx]) + ".jpg")
        rgbImage = cv2.resize(rgbImage, (100, 200))
        rgbImage = rgbImage[25:175, 25:75]
        maskImage = cv2.imread(self.dataDir + "/liquidMask_" +
                               str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        maskImage[maskImage > 1] = 0

        # Removing borders
        maskImage[:, :25] = 0  # Left boundary removed
        maskImage[:, 75:] = 0  # Right boundary removed
        maskImage[:25, :] = 0  # Top boundary removed
        maskImage[175:, :] = 0  # Bottom boundary removed
        maskImage[maskImage > 0] = 255

        rgbImage = np.rollaxis(rgbImage, 2, 0)

        if(rgbImage is None):
            print("rgb none at ", self.rgbFileIDs[idx])
        if(depthImage is None):
            print("depth none at ", self.rgbFileIDs[idx])
        if(maskImage is None):
            print("mask none at ", self.rgbFileIDs[idx])

        return rgbImage, maskImage


class TransparentPouringSegmentationDataset_zeropadded(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "transparent" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[2] for x in self.rgbFileIDs]
        random.shuffle(self.rgbFileIDs)
        self.datasetSize = len(self.rgbFileIDs)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self, idx):
        rgbImage = cv2.imread(
            self.dataDir + "/rgb_transparent_" + str(self.rgbFileIDs[idx]) + ".jpg")
        rgbImage = cv2.resize(rgbImage, (100, 200))
        maskImage = cv2.imread(self.dataDir + "/liquidMask_" +
                               str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        maskImage[maskImage > 1] = 0
        depthImage = np.zeros([480, 480])

        rgbImage_zeropadded = np.zeros([480, 480, 3]).astype(np.uint8)
        maskImage_zeropadded = np.zeros([480, 480]).astype(np.uint8)

        centers = np.random.randint(0, 300, size=[2])
        corners = centers + \
            np.array([[0, 0], [rgbImage.shape[0], rgbImage.shape[1]]])
        corners = np.clip(corners, a_min=0, a_max=480)
        imgShape = corners[1] - corners[0]

        rgbImage_zeropadded[corners[0, 0]: corners[1, 0], corners[0, 1]: corners[1, 1], :] = rgbImage[0: imgShape[0], 0: imgShape[1], :]
        maskImage_zeropadded[corners[0, 0]: corners[1, 0], corners[0, 1]: corners[1, 1]] = maskImage[0: imgShape[0], 0: imgShape[1]]

        rgbImage_zeropadded = np.rollaxis(rgbImage_zeropadded, 2, 0)

        if(rgbImage is None):
            print("rgb none at ", self.rgbFileIDs[idx])
        if(depthImage is None):
            print("depth none at ", self.rgbFileIDs[idx])
        if(maskImage is None):
            print("mask none at ", self.rgbFileIDs[idx])

        # return rgbImage, depthImage, maskImage
        return rgbImage_zeropadded, depthImage, maskImage_zeropadded


def getMaskedImage(image, mask, threshold):
    tempImage = image.copy()
    tempImage[mask < threshold] = 0.0
    return tempImage


def main():

    dataDir = "rope_data_segmented/train/"
    ropeSegmentationDataset = RopeSegmentationDataset(dataDir)

    print(ropeSegmentationDataset.__len__())

    image, imageMask = ropeSegmentationDataset.__getitem__(10)
    cv2.imshow("image", image)
    cv2.imshow("imageMask", imageMask*255)
    cv2.waitKey()


if __name__ == '__main__':
    main()
