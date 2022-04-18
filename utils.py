import os
import cv2
import copy
import numpy as np

def get_cup_corners_contours(cupMask, cropPad=5):
    """
    Returns the corners of the largest mask in an image in pixel coordinates; this is used for cropping images.
    
    Args:
        cupMask: image input
        cropPad: padding for cropping
    
    Returns:
        imgCorners: corners in pixel coords
        contourImg: contour visualization using opencv
    """
    ret,thresh = cv2.threshold(cupMask,0,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bigCnt = 0
    bigCntIdx = 0
    for k in range(len(contours)):
        if contours[k].shape[0] > bigCnt:
            bigCnt = contours[k].shape[0]
            bigCntIdx = k

    colorMask = cv2.cvtColor(cupMask, cv2.COLOR_GRAY2BGR)
    contourImg = cv2.drawContours(copy.deepcopy(colorMask), contours, bigCntIdx, (0,255,0), 3)

    rect = cv2.minAreaRect(contours[bigCntIdx])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    x1 = box[:, 1].min()
    x2 = box[:, 1].max()
    y1 = box[:, 0].min()
    y2 = box[:, 0].max()

    imgCorners = np.array([x1-cropPad, x2+cropPad, y1-cropPad, y2+cropPad])
    return imgCorners, contourImg
