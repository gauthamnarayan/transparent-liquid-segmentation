import argparse
import os
import sys
import time
import copy
import argparse

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import imageio

import utils
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
    corners = centers + np.array([[0,0],[image.shape[0], image.shape[1]]])
    corners = np.clip(corners, a_min=0, a_max=480)
    imgShape = corners[1] - corners[0]

    imageZeropadded[corners[0, 0]: corners[1, 0], corners[0, 1]: corners[1, 1], :] = image[0: imgShape[0], 0: imgShape[1], :]
    return imageZeropadded

def get_height_from_mask(mask):
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
    print(x1, x2, y1, y2)

    height = abs(y1 - y2)
    return height

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

def retain_target_container(cup_mask, padding=10):
    ret,thresh = cv2.threshold(cup_mask,0,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bigCnt = 0
    bigCntIdx = 0
    for k in range(len(contours)):
        if contours[k].shape[0] > bigCnt:
            bigCnt = contours[k].shape[0]
            bigCntIdx = k

    x,y,w,h = cv2.boundingRect(contours[bigCntIdx])
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
            if(img1[k,l].all() > 0 and img2[k,l].all() > 0):
                intersectionPixels += 1
            if(img1[k,l].all() > 0 or img2[k,l].all() > 0):
                unionPixels += 1
    iou = intersectionPixels/unionPixels
    return iou

def test_model(test_dir, modelFileDir, args):
    if args.mode == "gpu":
        device = torch.device("cuda")

    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()
    model.load_state_dict(torch.load(modelFileDir, map_location=device))
    model = model.to(device)

    unetResults = []
    rgbImageList = []

    files = os.listdir(test_dir)
    files = [x for x in files if "glass" not in x]
    files = [x for x in files if "empty_cup" not in x]
    files = [x for x in files if "json" not in x]
    number_of_files = len(files)
    print("Number of files: ", number_of_files)

    cup_mask = cv2.imread(os.path.join(test_dir, "empty_cup_0_glass.png"), cv2.IMREAD_GRAYSCALE)
    iou_sum = 0
    img_counter = 0
    for i in range(number_of_files):
        start = time.time()
        print("Frame ID: ", i)
        rgbImage = cv2.imread(os.path.join(test_dir, files[i]))
        print(os.path.join(test_dir, files[i]))
        rgbImage = cv2.resize(rgbImage, (150, 300))
        tempRGBImage = copy.deepcopy(rgbImage)

        image = np.rollaxis(tempRGBImage, 2, 0)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image).cuda()
        image = torch.tensor(image)
        maskPred = model(image.float()).cpu().detach().numpy()
        maskPred = np.squeeze(maskPred)

        maskThresholded = np.zeros_like(maskPred)
        maskThresholded[maskPred > 0.5] = 255
        maskThresholded = maskThresholded.astype(np.uint8)
        # maskThresholded = cv2.cvtColor(maskThresholded, cv2.BGR2)

        GTMask_dirname = files[i].split(".")[0] + "_json"
        print(os.path.join(test_dir, GTMask_dirname, "label.png"))
        GTMask = cv2.imread(os.path.join(test_dir, GTMask_dirname, "label.png"), cv2.IMREAD_GRAYSCALE)
        GTMask[GTMask>0] = 255

        iou = calcIOU(maskThresholded, GTMask)
        print(i)
        print("IoU: ", iou)
        iou_sum +=  iou
        img_counter += 1
        
        stacked_img = np.hstack([rgbImage, 
                                 cv2.cvtColor(maskThresholded, cv2.COLOR_GRAY2BGR), 
                                 cv2.cvtColor(GTMask, cv2.COLOR_GRAY2BGR)])            
        cv2.imwrite(os.path.join("../results", "stacked_results_" + files[i].split(".")[0].split("_")[1] + ".jpg"), stacked_img)

    print("AvgIoU: ", iou_sum/i)
    return iou_sum, img_counter


def run_all_test_videos(args):
    video_names = os.listdir(os.path.abspath(args.test_videos_path))
    
    if not os.path.exists(os.path.abspath(args.results_dir)):
        os.makedirs(os.path.abspath(args.results_dir))
    
    iou_sum_global = 0
    img_counter_global = 0
    for name in video_names:
        print("Processing: ", name)
        iou_sum, img_counter = test_model(os.path.join(args.test_videos_path, name), args.model_path, args)

        iou_sum_global += iou_sum
        img_counter_global += img_counter

    print("global Iou:", iou_sum_global/img_counter_global)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--test-videos-path', required=True)
    parser.add_argument('--results-dir', default="../results")
    parser.add_argument('--mode', default='gpu')
    args = parser.parse_args()    

    # args.model_path = "../data/saved_models/transparent_liquid_segmentation_unet_150x300_epoch_10_20220420_15_29_55" # Ours-full dataset
    # args.test_videos_path = "../datasets/pouring_testset_videos/video_" + str(i)

    run_all_test_videos(args)