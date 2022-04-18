import os
import sys
import numpy as np
import cv2
import time
import copy

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import imageio

import unet

def applyMask(image, mask, threshold):
    tempImage = image.copy()
    tempImage[mask<threshold] = 0.0
    return tempImage

def applyBlending(image, mask):
    image = copy.deepcopy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(mask[i,j] > 0):
                image[i,j] = 0.5 * image[i,j]  + [127,0,0] # Half of Blue mask. 
    return image

def drawParticles(img, p, pSize):
    img = copy.deepcopy(img)
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i in range(p.shape[0]):
        x = p[i, 0] * pSize
        y = p[i, 1] * pSize
        # img = 
        img = cv2.circle(img, (x, y), pSize//2, (255, 0, 0), -1) # Draws solid circles
        img = cv2.circle(img, (x, y), pSize//2, (0, 0, 255), 1) # Draws boundary of circles
    return img

def calcParticles(maskImage, pSize, pNum):
    xRange = maskImage.shape[1]
    yRange = maskImage.shape[0]

    distribution = np.array(np.where(maskImage > 0)).T
    distribution = distribution//pSize
    distribution = distribution[:, ::-1]

    if(distribution.shape[0] < pNum):
        replace = True
    else:
        replace = False

    idx = np.random.choice(distribution.shape[0], [pNum], replace=replace)
    particles = distribution[idx]
    return particles

def getZeroPaddedImage(image):
    imageZeropadded = np.zeros([480, 480, 3]).astype(np.uint8)
    centers = np.random.randint(0, 300, size=[2])
    # centers = np.array([240, 240])
    corners = centers + np.array([[0,0],[image.shape[0], image.shape[1]]])
    corners = np.clip(corners, a_min=0, a_max=480)
    imgShape = corners[1] - corners[0]

    imageZeropadded[corners[0, 0]: corners[1, 0], corners[0, 1]: corners[1, 1], :] = image[0: imgShape[0], 0: imgShape[1], :]
    return imageZeropadded


def getRectangleCoords(mask):

    ret,thresh = cv2.threshold(mask,0,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bigCnt = 0
    bigCntIdx = 0
    for k in range(len(contours)):
        if contours[k].shape[0] > bigCnt:
            bigCnt = contours[k].shape[0]
            bigCntIdx = k

    rect = cv2.minAreaRect(contours[bigCntIdx])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x1 = box[:, 1].min()
    x2 = box[:, 1].max()
    y1 = box[:, 0].min()
    y2 = box[:, 0].max()

    cropPad = 10
    imgCorners = np.array([x1-cropPad, x2+cropPad, y1-cropPad, y2+cropPad])
    imgCorners = np.clip(imgCorners, a_min=0, a_max=480)
    return imgCorners


def calcIOU(img1, img2):
    intersectionPixels = 0
    unionPixels = 0
    for k in range(img1.shape[0]):
        for l in range(img1.shape[1]):
            if(img1[k,l].all() > 0 and img2[k,l].all() > 0):
                intersectionPixels += 1
            if(img1[k,l].all() > 0 or img2[k,l].all() > 0):
                unionPixels += 1
    iou = intersectionPixels/unionPixels
    return iou


def testCupGeneralization(testDir, model):
    unetResults = []
    rgbImageList = []
    iouSum = 0
    imgCounter = 0

    filenames = os.listdir(testDir)
    filenames = [x.split(".")[0] for x in filenames if "rgb" in x and "json" not in x]
    filenames = [x.split("_")[1] for x in filenames if "rgb" in x and "json" not in x]

    print(testDir)
    print(filenames)
    for i in filenames:
        rgbImage = cv2.imread(testDir + "/rgb_" + str(i) + ".jpg")
        cupMaskImage = cv2.imread(testDir + "/cupMask_" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        tempRGBImage = copy.deepcopy(rgbImage)

        # Get zero padded image for teesting with cups of different aspect ratio
        tempRGBImage[cupMaskImage==0] = 0
        
        image = np.rollaxis(tempRGBImage, 2, 0)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image).cuda()
        maskPred = model(image.float()).cpu().detach().numpy()
        maskPred = np.squeeze(maskPred)

        maskPred[cupMaskImage == 0] = 0

        threshold = 0.95
        mask = copy.deepcopy(maskPred)
        mask[mask<threshold] = 0
        mask[mask>threshold] = 255
        
        blendedImage = applyBlending(rgbImage, mask)
        unetResults.append(blendedImage)
        rgbImageList.append(rgbImage)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        maskPred = cv2.cvtColor(maskPred, cv2.COLOR_GRAY2BGR)

        # Obtain rgbImage mask
        imgCorners = getRectangleCoords(cupMaskImage)

        # Calculate IoU
        GTMask = cv2.imread(os.path.join(testDir, "rgb_" + str(i) + "_json", "label.png"), cv2.IMREAD_GRAYSCALE)
        iou = calcIOU(mask, GTMask)
        print("IoU: ", iou)
        iouSum += iou
        imgCounter += 1


        croppedRGBImage = rgbImage[imgCorners[0]: imgCorners[1], imgCorners[2]: imgCorners[3]]
        croppedBlendedImage = blendedImage[imgCorners[0]: imgCorners[1], imgCorners[2]: imgCorners[3]]

    unetResults = [x[:, :, ::-1] for x in unetResults]
    rgbImageList = [x[:, :, ::-1] for x in rgbImageList]

def testBackgroundGeneralization(testDir, model, threshold=0.8):
    unetResults = []
    rgbImageList = []
    iouSum = 0
    imgCounter = 0
    filenames = os.listdir(testDir)
    filenames = [x.split(".")[0] for x in filenames if "rgb" in x and "json" not in x]
    filenames = [x.split("_")[1] for x in filenames if "rgb" in x and "json" not in x]

    for i in filenames:
        rgbImage = cv2.imread(testDir + "/rgb_" + str(i) + ".jpg")
        tempRGBImage = copy.deepcopy(rgbImage)

        # Get zero padded image for teesting with cups of different aspect ratio
        tempRGBImage[cupMaskImage==0] = 0
        
        image = np.rollaxis(tempRGBImage, 2, 0)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image).cuda()
        maskPred = model(image.float()).cpu().detach().numpy()
        maskPred = np.squeeze(maskPred)

        mask = copy.deepcopy(maskPred)
        mask[mask<threshold] = 0
        mask[mask>threshold] = 255
        
        blendedImage = applyBlending(rgbImage, mask)
        unetResults.append(blendedImage)
        rgbImageList.append(rgbImage)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        maskPred = cv2.cvtColor(maskPred, cv2.COLOR_GRAY2BGR)

        # Calculate IoU
        GTMask = cv2.imread(os.path.join(testDir, "rgb_" + str(i) + "_json", "label.png"), cv2.IMREAD_GRAYSCALE)
        iou = calcIOU(mask, GTMask)
        print("IoU: ", iou)
        iouSum += iou
        imgCounter += 1

    print("Avg IoU: ", iouSum/imgCounter)
    unetResults = [x[:, :, ::-1] for x in unetResults]
    rgbImageList = [x[:, :, ::-1] for x in rgbImageList]


def main():
    # Cup shape generalization
    # testDir = "/home/gautham/programs/RPAD/RPAD_backup/pouring/temp_data/CUT_UNet_data_transparent_liquid_7_testset_different_shapes_cup_bowls_jars_dimLight_1_2_combined_good_segmentation_cases/split_tall_medium_short/short/"
    # testDir = "/home/gautham/programs/RPAD/RPAD_backup/pouring/temp_data/CUT_UNet_data_transparent_liquid_7_testset_different_shapes_cup_bowls_jars_dimLight_1_2_combined_good_segmentation_cases/split_tall_medium_short/medium/"
    testDir = "/home/gautham/programs/RPAD/RPAD_backup/pouring/temp_data/CUT_UNet_data_transparent_liquid_7_testset_different_shapes_cup_bowls_jars_dimLight_1_2_combined_good_segmentation_cases/split_tall_medium_short/tall/"


    # Diverse Background generalization
    # testDir = "/home/gautham/programs/RPAD/RPAD_backup/pouring/temp_data/kai_data_full_scene/testing_raw/cropped/split_short_med_tall/tall/"
    # testDir = "/home/gautham/programs/RPAD/RPAD_backup/pouring/temp_data/kai_data_full_scene/testing_raw/cropped/split_short_med_tall/medium/"
    # testDir = "/home/gautham/programs/RPAD/RPAD_backup/pouring/temp_data/kai_data_full_scene/testing_raw/cropped/split_short_med_tall/short/"

    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()
    model = model.cuda()
    model.load_state_dict(torch.load("/home/gautham/programs/RPAD/RPAD_backup/unet_trials/saved_models/saved_models_pouring_unet_6_transparent/pytorchmodel_pouring_unet_transparent_CUT_UNet_liquid_mask_2_3_4_combined_zeropadded_epoch_30_20210625_00_49_15"))

    testBackgroundGeneralization(testDir, model)

if __name__ == "__main__":
    main()