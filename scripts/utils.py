import os
import cv2
import copy
import numpy as np

def get_cup_corners_contours(cupMask, cropPad=5):
    ret,thresh = cv2.threshold(cupMask,0,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contourImg = cv2.drawContours(rgbImage, contours, -1, (0,255,0), 3) # Draw all contours

    bigCnt = 0
    bigCntIdx = 0
    for k in range(len(contours)):
        if contours[k].shape[0] > bigCnt:
            bigCnt = contours[k].shape[0]
            bigCntIdx = k

    # print("len contours: ", len(contours))
    # print("bigCntIdx : ", bigCntIdx)
    # print("contours", contours)
    # print("Num of contours: ", len(contours))
    # print("Largest contour: ", bigCnt)
    colorMask = cv2.cvtColor(cupMask, cv2.COLOR_GRAY2BGR)
    contourImg = cv2.drawContours(copy.deepcopy(colorMask), contours, bigCntIdx, (0,255,0), 3)
    # contourImg = cv2.drawContours(copy.deepcopy(rgbImage), contours, -1, (0,255,0), 3)

    rect = cv2.minAreaRect(contours[bigCntIdx])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print("rect: ", rect)
    # print("box: ", box)

    x1 = box[:, 1].min()
    x2 = box[:, 1].max()
    y1 = box[:, 0].min()
    y2 = box[:, 0].max()

    imgCorners = np.array([x1-cropPad, x2+cropPad, y1-cropPad, y2+cropPad])
    # imgCorners = np.clip(imgCorners, a_min=0, a_max=480)
    # imgCorners = np.clip(imgCorners, a_min=0, a_max=cupMask.shape[0])

    return imgCorners, contourImg
