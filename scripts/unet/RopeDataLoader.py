import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import random
#import matplotlib.pyplot as plt
import math

#def rect(ax, poke, c):
#    x, y, t, l, good = poke
#    dx = -200 * l * math.cos(t)
#    dy = -200 * l * math.sin(t)
#    ax.arrow(x, y, dx, dy, head_width=5, head_length=5)
#
#def plot_sample(img_before, img_after, action):
#    #plt.figure()
#    f, (ax1, ax2) = plt.subplots(1, 2)
#    ax1.imshow(img_before.copy())
#    rect(ax1, action, "blue")
#    ax2.imshow(img_after.copy())

class FrankaSegmentationDataset_48x48(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "rgb" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[1] for x in self.rgbFileIDs]
        random.shuffle(self.rgbFileIDs)
        self.datasetSize = len(self.rgbFileIDs)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self,idx):
        rgbImage = cv2.imread(self.dataDir + "/rgb_" + str(self.rgbFileIDs[idx]) + ".jpg")
        depthImage = cv2.imread(self.dataDir + "/depth_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        maskImage = cv2.imread(self.dataDir + "/mask_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)

        rgbImage = cv2.resize(rgbImage, (48,48))
        rgbImage = np.rollaxis(rgbImage, 2, 0)
        depthImage = cv2.resize(depthImage, (48,48))
        maskImage = cv2.resize(maskImage, (48,48))

        return rgbImage, depthImage, maskImage 

class FrankaSegmentationDataset_withDistractor(Dataset):

    def __init__(self, dataDir, transform=None, distractor_type='puck', imsize=480):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "rgb" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[1] for x in self.rgbFileIDs]
        # random.shuffle(self.rgbFileIDs)
        self.datasetSize = len(self.rgbFileIDs)
        self.distractor_type = distractor_type
        self.imsize = imsize

    def __len__(self):
        return self.datasetSize

    def __getitem__(self,idx):
        print(self.rgbFileIDs[idx]) 
        # x = 103
        x = 9
        rgbImage = cv2.imread(self.dataDir + "/rgb_" + str(x) + ".jpg")
        depthImage = cv2.imread(self.dataDir + "/depth_" + str(x) + ".jpg", cv2.IMREAD_GRAYSCALE)
        maskImage = cv2.imread(self.dataDir + "/puckMask_" + str(x) + ".jpg", cv2.IMREAD_GRAYSCALE)
        # rgbImage = cv2.imread(self.dataDir + "/rgb_" + str(self.rgbFileIDs[idx]) + ".jpg")
        # depthImage = cv2.imread(self.dataDir + "/depth_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        # maskImage = cv2.imread(self.dataDir + "/puckMask_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)

        if(self.distractor_type == 'bluePuck'):
            bluePuck = cv2.imread("./blue_puck_images/bluePuck_" + str(np.random.randint(1,6)) + ".jpg")
            # orangePuck = cv2.imread("./orange_puck.jpg")
            xb = np.random.randint(0,380)
            yb = np.random.randint(0,380)
            xr = np.random.randint(0,380)
            yr = np.random.randint(0,380)
            for i in range(bluePuck.shape[0]):
                for j in range(bluePuck.shape[1]):
                    if(bluePuck[i,j].all() != 0):
                        rgbImage[yb+i, xb+j] = bluePuck[i,j]
                        maskImage[yb+i, xb+j] = 0

                    # if(orangePuck[i,j].all() != 0):
                        # rgbImage[yr+i, xr+j] = orangePuck[i,j]
                        # maskImage[yr+i, xr+j] = 0

        if(self.distractor_type == 'door'):
            doorImage = cv2.imread("./door_image.jpg")
            doorImage = doorImage[:, :165]

            randX = np.random.randint(0,65)
            randY = np.random.randint(0,380)
            randomCropDoor = doorImage[randY:randY+100, randX:randX+100]

            xd = np.random.randint(0,380)
            yd = np.random.randint(0,380)
            rgbImage[yd:yd+100, xd:xd+100] = randomCropDoor
            maskImage[yd:yd+100, xd:xd+100] = 0

        rgbImage = cv2.resize(rgbImage, (self.imsize, self.imsize))
        depthImage = cv2.resize(depthImage, (self.imsize, self.imsize))
        maskImage = cv2.resize(maskImage, (self.imsize, self.imsize))

        rgbImage = np.rollaxis(rgbImage, 2, 0)
        # cv2.imshow("rgbImage", rgbImage)
        # cv2.imshow("maskImage", maskImage*255)
        # cv2.waitKey()
        return rgbImage, depthImage, maskImage 

class FrankaSegmentationDataset(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "rgb" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[1] for x in self.rgbFileIDs]
        random.shuffle(self.rgbFileIDs)
        self.datasetSize = len(self.rgbFileIDs)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self,idx):
        rgbImage = cv2.imread(self.dataDir + "/rgb_" + str(self.rgbFileIDs[idx]) + ".jpg")
        rgbImage = np.rollaxis(rgbImage, 2, 0)
        depthImage = cv2.imread(self.dataDir + "/depth_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        # maskImage = cv2.imread(self.dataDir + "/mask_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        maskImage = depthImage

        return rgbImage, depthImage, maskImage 

class MujocoSawyerReachSegmentationDataset(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "rgb" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[1] for x in self.rgbFileIDs]
        random.shuffle(self.rgbFileIDs)

        #print(self.rgbFileIDs)
        #print(len(self.rgbFileIDs))

        self.datasetSize = len(self.rgbFileIDs)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self,idx):
        rgbImage = cv2.imread(self.dataDir + "/rgb_" + str(self.rgbFileIDs[idx]) + ".jpg")
        rgbImage = np.rollaxis(rgbImage, 2, 0)
        maskImage = cv2.imread(self.dataDir + "/mask_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)

        #cv2.imshow("rgbImage", rgbImage)
        #cv2.waitKey()


        return rgbImage, maskImage 

class HandSegmentationDataset(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir

        imageFileNames = os.listdir(self.dataDir)
        rgbFileNames = [x for x in imageFileNames if "rgb" in x]
        self.rgbFileIDs = [x.split(".")[0] for x in rgbFileNames]
        self.rgbFileIDs = [x.split("_")[1] for x in self.rgbFileIDs]
        random.shuffle(self.rgbFileIDs)

        #print(self.rgbFileIDs)
        #print(len(self.rgbFileIDs))

        self.datasetSize = len(self.rgbFileIDs)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self,idx):
        rgbImage = cv2.imread(self.dataDir + "/rgb_" + str(self.rgbFileIDs[idx]) + ".jpg")
        rgbImage = cv2.resize(rgbImage, (640,360))
        rgbImage = np.rollaxis(rgbImage, 2, 0)
        maskImage = cv2.imread(self.dataDir + "/mask_" + str(self.rgbFileIDs[idx]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        maskImage = cv2.resize(maskImage, (640,360))
        return rgbImage, maskImage 

class RopeSegmentationDataset(Dataset):

    def __init__(self, dataDir, transform=None):
        self.dataDir = dataDir
        runDirs = os.listdir(self.dataDir)
        self.imagePathList = []
    
        for run in runDirs:
            imageList = os.listdir(self.dataDir + "/" + run)
            imageList.remove("actions.npy")
            imageList = [x for x in imageList if "mask" not in x]
            imageList = [self.dataDir + run + "/" + x  for x in imageList]
            self.imagePathList.extend(imageList)

        self.datasetSize = len(self.imagePathList)
        
    def __len__(self):
        return self.datasetSize

    def __getitem__(self,idx):
        imageName = self.imagePathList[idx]
        imageId = imageName.split(".jpg")[0]
        imageMaskName = imageId + "_mask.jpg" 
        return np.rollaxis(cv2.imread(imageName), 2, 0), cv2.imread(imageMaskName, cv2.IMREAD_GRAYSCALE)


class RopeDataset(Dataset):

    def __init__(self, dataDir, transform=None):
        
        self.dataDir = dataDir
        self.imageActions = []
        with open(self.dataDir + "/train_image_actions.txt") as imageActionFile:
            reader = csv.reader(imageActionFile)
            for row in reader:
                self.imageActions.append(row)

        # To remove csv header row.
        self.imageActions = self.imageActions[1:] 
        self.datasetSize = len(self.imageActions)
        
    def __len__(self):
        return self.datasetSize

    def __getitem__(self,idx):
        imageBefore = cv2.imread(self.dataDir + "/" + self.imageActions[idx][0])
        imageAfter = cv2.imread(self.dataDir + "/" + self.imageActions[idx][1])
        action = np.array(self.imageActions[idx][2:]).astype(np.float)
        return imageBefore, imageAfter, action

def getMaskedImage(image, mask, threshold):
    tempImage = image.copy()
    tempImage[mask<threshold] = 0.0
    return tempImage

def main():

    dataDir = "rope_data_segmented/train/"
    ropeSegmentationDataset = RopeSegmentationDataset(dataDir)

    print(ropeSegmentationDataset.__len__())

    image, imageMask = ropeSegmentationDataset.__getitem__(10)
    cv2.imshow("image", image)
    cv2.imshow("imageMask", imageMask*255)
    cv2.waitKey()


#    ropeDataset = RopeDataset("./rope_data_segmented/train")
#
#    testImgDir = "./rope_data_segmented/train/run03/"
#    testImgs = os.listdir(testImgDir)
#    testImgs.remove("actions.npy")
#
#    backSub = cv2.createBackgroundSubtractorMOG2()
#
#    for i in range(10):
#        img = cv2.imread(testImgDir + testImgs[i])
#        fgMask = backSub.apply(img)
#        #cv2.imshow("fgMask", getMaskedImage(img, fgMask, 200))
#        #cv2.waitKey()
    


if __name__ == '__main__':
    main()



