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

import unet
#from RopeDataLoader import RopeSegmentationDataset
#from RopeDataLoader import HandSegmentationDataset
# from RopeDataLoader import MujocoSawyerReachSegmentationDataset
from RopeDataLoader import FrankaSegmentationDataset_48x48
from RopeDataLoader import FrankaSegmentationDataset
from RopeDataLoader import FrankaSegmentationDataset_withDistractor

def applyMask(image, mask, threshold):
    tempImage = image.copy()
    tempImage[mask<threshold] = 0.0
    return tempImage

def getPuckPixels(image, fgMask, sawyerMask, threshold):
    tempImage = image.copy()
    tempImage[fgMask<threshold] = 0.0
    tempImage[sawyerMask>threshold] = 0.0
    return tempImage

# threshold = 0.9
#threshold = 0.8

#handTestDir = "../handSegmentationDataset/test/"
#handTestDir = "../handSegmentationDataset/test_hand3d/test_images"
#handTestDir = "../handSegmentationDataset/test_rope_and_hand_seg"
#handTestDir = "../handSegmentationDataset/train_hand_segmentation/test"
# handTestDir = "./datasets/puck_dataset/test"
# handTestDir = "./datasets/franka_wooden_board_dataset6/test"
# handTestDir = "./datasets/franka_wooden_board_puck_test2/"
# handTestDir = "./datasets/sawyer_door_dataset/test/"
# handTestDir = "./datasets/sawyer_door_test_dataset"
# handTestDir = "./datasets/sawyer_hurdle_dataset/test"
# handTestDir = "./datasets/sawyer_pickup_dataset/test"
# handTestDir = "./datasets/franka_wooden_board_segmentation_night1/test"
# handTestDir = "./datasets/franka_wooden_board_segmentation_night3/test
# handTestDir = "../franka_experiments/scripts/temp_data/goal_datasets/night_1/"
# handTestDir = "./datasets/franka_wooden_board_segmentation_newhouse_1/test"
# handTestDir = "../franka_experiments/scripts/temp_data/franka_wooden_board_segmentation_daytime1"
# handTestDir = "./datasets/franka_wooden_board_puck_test4/"
# handTestDir = "./datasets/sawyer_hurdle_middle_test"
handTestDir = "/home/gautham/programs/RPAD/trials/franka_experiments/scripts/temp_data/goal_datasets/new_house_1/"
# handTestDir = "/home/gautham/programs/RPAD/trials/franka_experiments/scripts/temp_data/franka_wooden_board_orange_puck/test"
#handTestDir = "../sawyer_push_dataset/test/"
#handTestDir = "../handSegmentationDataset/rope_segmentation_data/train"
# handTestDataset = FrankaSegmentationDataset_48x48(handTestDir)
# handTestDataset = FrankaSegmentationDataset(handTestDir)
handTestDataset = FrankaSegmentationDataset_withDistractor(handTestDir, distractor_type=None, imsize=48)
handTestDataLoader = DataLoader(handTestDataset, batch_size=1, shuffle=False, num_workers=5)

model = unet.UNet(n_channels=3, n_classes=1)
model = model.float()
model = model.cuda()

criterion = torch.nn.BCELoss()
#model.load_state_dict(torch.load("./saved_models/pytorchmodel_epoch20_20191001_19_52_24"))
#model.load_state_dict(torch.load("./saved_models/pytorchmodel_epoch20"))
#model.load_state_dict(torch.load("./saved_models/seuss_saved_models/pytorchmodel_epoch20_20191211_23_41_42"))
#model.load_state_dict(torch.load("./saved_models/pytorchmodel_rope_segmentation"))
#model.load_state_dict(torch.load("./saved_models/pytorchmodel_epoch30_20200113_handsegmentation"))
#model.load_state_dict(torch.load("./saved_models/pytorchmodel_epoch30_20200114_handsegmentation_rope_background"))
# model.load_state_dict(torch.load("./saved_models/pytorchmodel_franka_segmentation_48x48_epoch30_20200611_15_20_51"))
# model.load_state_dict(torch.load("./saved_models/pytorchmodel_franka_wooden_segmentation_48x48_epoch30_20200611_16_45_04"))
# model.load_state_dict(torch.load("./saved_models/pytorchmodel_franka_wooden_segmentation6_48x48_epoch20_20200623_04_32_53"))
# model.load_state_dict(torch.load("./saved_models/pytorchmodel_franka_wooden_segmentation6_epoch20_20200623_05_10_06"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_sawyer_door_segmentation_30_20200701_22_19_36"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_sawyer_door_segmentation_distractors_30_20200717_00_55_01"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_sawyer_door_segmentation_distractors_48x48_30_20200717_01_15_24"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_sawyer_pickup_segmentation_48x48_30_20200717_03_49_05"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_sawyer_hurdle_segmentation_10_20200704_20_11_16"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_franka_wooden_segmentationdataset_distractors_segmentation_480x480_20_20200715_08_36_03"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_franka_wooden_segmentation_night1_distractors_48x48_epoch20_20200722_07_04_13"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_franka_wooden_segmentation_night3_distractors_480x480_epoch30_20200724_04_05_30"))
# model.load_state_dict(torch.load("./saved_models2/pytorchmodel_franka_wooden_segmentation_newhouse_distractors_480x480_epoch30_20200726_21_10_45"))
model.load_state_dict(torch.load("./saved_models2/pytorchmodel_franka_wooden_segmentation_newhouse_distractors_48x48_epoch30_20200726_22_50_46"))

backSub = cv2.createBackgroundSubtractorMOG2() 

print("Testing started")
counter = 0

### Very important to set no_grad(), else model might train on testing data ###
model.eval()
with torch.no_grad():
    for batch, (image, depth, mask_true) in enumerate(handTestDataLoader):
        image = image.cuda()
        mask_true = mask_true.cuda().float()
        mask_pred = model(image.float())

        loss = criterion(mask_pred, mask_true)
        print("Batch: %d\tTesting Loss: %f" % (batch, loss.detach().cpu().item()))

        maskBatch = mask_pred.detach().cpu().numpy()
        maskTrueBatch = mask_true.detach().cpu().numpy()

        for i in range(maskBatch.shape[0]):
            mask = maskBatch[i].squeeze()
            maskTrue = maskTrueBatch[i].squeeze()
            #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            print(mask.shape)
            tempImage = image[i]
            tempImage = np.rollaxis(tempImage.detach().cpu().numpy(), 0,3)
            fgMask = backSub.apply(tempImage)

            # Uncomment to get puck mask.
            #puckImage = getPuckPixels(tempImage, fgMask ,mask, threshold)
            #compareImages = np.hstack([tempImage,puckImage])
            #print(maskedImage.shape)

            # Uncomment to view the actual mask from the model.
            # maskedImage = applyMask(tempImage, mask, threshold)
            # compareImages = np.hstack([tempImage,maskedImage])

            # groundTruthMask = tempImage.copy()
            # groundTruthMask[fgMask<threshold] = 0
            # groundTruthMask[fgMask<threshold] = 0
            # compareGT = np.hstack([tempImage,groundTruthMask])
            
            kernel = np.ones((5,5),np.uint8)
            maskOpened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            masked_image_pred = copy.deepcopy(tempImage)
            masked_image_pred_opened = copy.deepcopy(tempImage)
            masked_image_true = copy.deepcopy(tempImage)

            masked_image_pred[mask<0.9] = 0
            masked_image_pred_opened[maskOpened<0.2] = 0
            masked_image_true[maskTrue<0.4] = 0
            stacked_image = np.hstack([tempImage, masked_image_pred, masked_image_pred_opened])

            #cv2.imwrite("./test_results/sawyer_push_keep_puck/" + str(counter) + "_.jpg", compareImages)
            #cv2.imwrite("./temp_files/gtMask_" + str(counter) + "_.jpg", compareGT)
            # cv2.imwrite("./test_results/puck_test/compare_" + str(counter) + "_.jpg", stacked_image)
            counter += 1
           
            t = 2
            print(mask.dtype)
            maskCopy = copy.deepcopy(mask)*255
            maskCopy[maskCopy<t] = 0
            maskedImage = applyMask(tempImage, maskCopy, t)

            cv2.imshow("maskedImage", maskedImage)
            # cv2.imshow("maskedImage", compareImages)
            #cv2.imshow("maskedImage", compareGT)
            cv2.imshow("mask", mask*255)
            cv2.imshow("maskCopy", maskCopy)
            # cv2.imshow("maskTrue", maskTrue*255)
            # cv2.imshow("stacked_image", stacked_image)
            cv2.waitKey()


