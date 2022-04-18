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

import utils

import unet


def applyMask(image, mask, threshold):
    tempImage = image.copy()
    tempImage[mask < threshold] = 0.0
    return tempImage


def applyBlending(image, mask):
    image = copy.deepcopy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(mask[i, j] > 0):
                # Half of Blue mask.
                image[i, j] = 0.5 * image[i, j] + [127, 0, 0]
    return image


def drawParticles(img, p, pSize):
    img = copy.deepcopy(img)
    if(len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i in range(p.shape[0]):
        x = p[i, 0] * pSize
        y = p[i, 1] * pSize
        # img =
        # Draws solid circles
        img = cv2.circle(img, (x, y), pSize//2, (255, 0, 0), -1)
        img = cv2.circle(img, (x, y), pSize//2, (0, 0, 255),
                         1)  # Draws boundary of circles
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
    corners = centers + np.array([[0, 0], [image.shape[0], image.shape[1]]])
    corners = np.clip(corners, a_min=0, a_max=480)
    imgShape = corners[1] - corners[0]

    imageZeropadded[corners[0, 0]: corners[1, 0], corners[0, 1]        : corners[1, 1], :] = image[0: imgShape[0], 0: imgShape[1], :]
    return imageZeropadded


def get_height_from_mask(mask):
    ret, thresh = cv2.threshold(mask, 0, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bigCnt = 0
    bigCntIdx = 0
    for k in range(len(contours)):
        if contours[k].shape[0] > bigCnt:
            bigCnt = contours[k].shape[0]
            bigCntIdx = k

    print("Contour length: ", len(contours))

    rect = cv2.minAreaRect(contours[bigCntIdx])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x1 = box[:, 1].min()
    x2 = box[:, 1].max()
    y1 = box[:, 0].min()
    y2 = box[:, 0].max()
    print(x1, x2, y1, y2)

    height = abs(y1 - y2)
    return height


def getRectangleCoords(mask):

    ret, thresh = cv2.threshold(mask, 0, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


def retain_target_container(cup_mask, padding=10):
    ret, thresh = cv2.threshold(cup_mask, 0, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bigCnt = 0
    bigCntIdx = 0
    for k in range(len(contours)):
        if contours[k].shape[0] > bigCnt:
            bigCnt = contours[k].shape[0]
            bigCntIdx = k

    x, y, w, h = cv2.boundingRect(contours[bigCntIdx])
    x1 = x - padding
    y1 = y - padding
    x2 = x1 + w + padding
    y2 = y1 + h + padding

    temp = np.zeros_like(cup_mask)
    temp[y1:y2, x1:x2] = cup_mask[y1:y2, x1:x2]
    return temp


def calcIOU(img1, img2):
    intersectionPixels = 0
    unionPixels = 0
    for k in range(img1.shape[0]):
        for l in range(img1.shape[1]):
            if(img1[k, l].all() > 0 and img2[k, l].all() > 0):
                intersectionPixels += 1
            if(img1[k, l].all() > 0 or img2[k, l].all() > 0):
                unionPixels += 1
    iou = intersectionPixels/unionPixels
    return iou


def testModelOrig(testDir, modelFileDir, threshold=0.8):
    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()
    device = torch.device("cuda")
    model.load_state_dict(torch.load(modelFileDir, map_location=device))
    model = model.to(device)

    unetResults = []
    rgbImageList = []

    files = os.listdir(testDir)
    files = [x for x in files if "glass" not in x]
    number_of_files = len(files)

    cupMaskImage = cv2.imread(os.path.join(
        testDir, "rgb_" + str(0) + "_glass.png"), cv2.IMREAD_GRAYSCALE)
    cupMaskImage = cv2.resize(cupMaskImage, (854, 480))
    for i in range(50, number_of_files):
        start = time.time()
        rgbImage = cv2.imread(os.path.join(testDir, "rgb_" + str(i) + ".jpg"))
        rgbImage = cv2.resize(rgbImage, (854, 480))
        tempRGBImage = copy.deepcopy(rgbImage)

        # Get zero padded image for teesting with cups of different aspect ratio
        # Not removing background reduces perfomance.
        tempRGBImage[cupMaskImage == 0] = 0
        image = np.rollaxis(tempRGBImage, 2, 0)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image).cuda()
        image = torch.tensor(image)
        maskPred = model(image.float()).cpu().detach().numpy()
        maskPred = np.squeeze(maskPred)

        mask = copy.deepcopy(maskPred)
        mask[mask < threshold] = 0
        mask[mask > threshold] = 255

        kernel = np.ones([5, 5])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Obtain rgbImage mask
        imgCorners = getRectangleCoords(cupMaskImage)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        maskPred = cv2.cvtColor(maskPred, cv2.COLOR_GRAY2BGR)
        stackedImage = np.hstack(
            [rgbImage, tempRGBImage, maskPred*255, mask]).astype(np.uint8)
        rgbImageList.append(stackedImage)

    return rgbImageList


def testModel_verticalCrop(testDir, modelFileDir, threshold=0.8):
    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()
    device = torch.device("cpu")
    model.load_state_dict(torch.load(modelFileDir, map_location=device))

    unetResults = []
    rgbImageList = []

    for i in range(200, number_of_files):
        print(i)
        rgbImage = cv2.imread(os.path.join(testDir, "rgb_" + str(i) + ".jpg")
        tempRGBImage=copy.deepcopy(rgbImage)

        # Get zero padded image for teesting with cups of different aspect ratio
        # Not removing background reduces perfomance.
        tempRGBImage[cupMaskImage == 0]=0

        # Cropping to vertical strip
        tempRGBImage[:, :25]=0  # Left boundary removed
        tempRGBImage[:, 75:]=0  # Right boundary removed
        tempRGBImage[:25, :]=0  # Top boundary removed
        tempRGBImage[175:, :]=0  # Bottom boundary removed

        image=np.rollaxis(tempRGBImage, 2, 0)
        image=np.expand_dims(image, axis=0)
        # image = torch.tensor(image).cuda()
        image=torch.tensor(image)
        maskPred=model(image.float()).cpu().detach().numpy()
        maskPred=np.squeeze(maskPred)

        mask=copy.deepcopy(maskPred)
        mask[mask < threshold]=0
        mask[mask > threshold]=255

        mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        maskPred=cv2.cvtColor(maskPred, cv2.COLOR_GRAY2BGR)

        # Obtain rgbImage mask
        imgCorners=getRectangleCoords(cupMaskImage)
        unetResults.append(maskPred)
        rgbImageList.append(tempRGBImage)

        return unetResults, rgbImageList

def testModel_150x300(testDir, modelFileDir, threshold=0.8):
    model=unet.UNet(n_channels=3, n_classes=1)
    model=model.float()
    device=torch.device("cuda")
    model.load_state_dict(torch.load(modelFileDir, map_location=device))
    model=model.to(device)

    unetResults=[]
    rgbImageList=[]

    files=os.listdir(testDir)
    files=[x for x in files if "glass" not in x]
    files=[x for x in files if "empty_cup" not in x]
    number_of_files=len(files)
    print("Number of files: ", number_of_files)

    crop_pad=5
    frame_id=0
    cup_frame_id=0
    cup_mask=cv2.imread(os.path.join(
        testDir, "empty_cup_0_glass.png"), cv2.IMREAD_GRAYSCALE)

    imgCorners, contour_img=utils.get_cup_corners_contours(cup_mask, crop_pad)
    imgCorners=np.clip(imgCorners, a_min=0, a_max=1280)

    for i in range(23, number_of_files):
        start=time.time()

        rgbImage=cv2.imread(os.path.join(testDir, "rgb_" + str(i) + ".jpg"))
        rgbImage=rgbImage[imgCorners[0]: imgCorners[1],
            imgCorners[2]: imgCorners[3]]
        rgbImage=cv2.resize(rgbImage, (150, 300))
        tempRGBImage=copy.deepcopy(rgbImage)

        # Get zero padded image for teesting with cups of different aspect ratio
        # Not removing background reduces perfomance.
        tempRGBImage[cupMaskImage == 0]=0

        image=np.rollaxis(tempRGBImage, 2, 0)
        image=np.expand_dims(image, axis=0)
        image=torch.tensor(image).cuda()
        image=torch.tensor(image)
        maskPred=model(image.float()).cpu().detach().numpy()
        maskPred=np.squeeze(maskPred)

        maskThresholded=np.zeros_like(maskPred)
        maskThresholded[maskPred >= threshold]=255
        maskThresholded[maskPred < threshold]=0

        maskThresholded=maskThresholded.astype(np.uint8)
        mask=copy.deepcopy(maskPred)
        mask[mask < threshold]=0
        mask[mask > threshold]=255

        print("Time for pred: ", time.time() - start)
        maskOrig=copy.deepcopy(mask)

        kernel=np.ones([10, 10])
        mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        maskThresholded=cv2.morphologyEx(
            maskThresholded, cv2.MORPH_OPEN, kernel)
        mask=maskThresholded

        # Liquid height detection
        mask=mask.astype(np.uint8)
        maskThresholded=maskThresholded.astype(np.uint8)
        # ret,thresh = cv2.threshold(mask,0,255,0)
        ret, thresh=cv2.threshold(maskThresholded, 0, 255, 0)
        contours, hierarchy=cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        maskPred=cv2.cvtColor(maskPred, cv2.COLOR_GRAY2BGR)

        bigCnt=0
        bigCntIdx=0
        for k in range(len(contours)):
            if contours[k].shape[0] > bigCnt:
                bigCnt=contours[k].shape[0]
                bigCntIdx=k
        rect=cv2.minAreaRect(contours[bigCntIdx])
        (x, y, w, h)=cv2.boundingRect(contours[bigCntIdx])
        mask=cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cropped_cup_mask=cup_mask[imgCorners[0]: imgCorners[1], imgCorners[2]: imgCorners[3]]
        cropped_cup_mask=cv2.resize(cropped_cup_mask, (150, 300))
        cupCorners, contour_img=utils.get_cup_corners_contours(
            cropped_cup_mask, cropPad=0)
        mask=cv2.rectangle(
            mask, (cupCorners[2], cupCorners[0]), (cupCorners[3], cupCorners[1]), (255, 0, 0), 2)

        contourImg=cv2.drawContours(mask, contours, bigCntIdx, (0, 255, 0), 3)

        # Liquid height calculation
        liquid_height=y - cupCorners[1]
        cup_height=cupCorners[0] - cupCorners[1]
        height_percentage=round(100 * liquid_height/cup_height, 2)
        textImg=np.zeros_like(mask)
        cv2.putText(textImg, "Height% : " + str(height_percentage), (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        maskOrig=cv2.cvtColor(maskOrig, cv2.COLOR_GRAY2BGR)
        stackedImage=np.hstack(
            [rgbImage, tempRGBImage, maskPred*255, mask]).astype(np.uint8)
        rgbImageList.append(stackedImage)

    unetResults=[x[:, :, ::-1] for x in unetResults]
    rgbImageList=[x[:, :, ::-1] for x in rgbImageList]

    return unetResults, rgbImageList

def run_all_test_videos(testDir, model_path):

    iouSumGlobal=0
    img_counter_global=0

    for i in range(len(testDir)):
        iouSum, img_counter=testModel_150x300_for_icra(testDir[i], model_path)

        iouSumGlobal += iouSum
        img_counter_global += img_counter

    print("global Iou:", iouSumGlobal/img_counter_global)



def main():
    modelFileDir="/home/gautham/programs/RPAD/pouring/unet_trials/saved_models/pytorchmodel_pouring_unet_transparent_CUT_UNet_pouring_videos_colored_background_2_090321_150x300_epoch_10_20210903_16_52_19"  # Ours-full dataset
    testDir="/home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/test_pouring_videos/video_11/"
    modelFileDir="/home/gautham/programs/RPAD/pouring/unet_trials/saved_models/pytorchmodel_pouring_unet_transparent_CUT_UNet_pouring_videos_082721_zeropadded_epoch_5_20210901_10_40_48"

    testModelOrig(testDir, modelFileDir)
    # testModel_verticalCrop(testDir, modelFileDir)
    # run_all_test_videos()

if __name__ == "__main__":
    main()
