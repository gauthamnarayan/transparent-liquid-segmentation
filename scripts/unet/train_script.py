import os
import sys
import numpy as np
import cv2
import time

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter

import unet
from RopeDataLoader import FrankaSegmentationDataset, FrankaSegmentationDataset_48x48, FrankaSegmentationDataset_withDistractor

# if not os.path.exists("saved_models"): os.makedirs("saved_models")

# handTrainDir = "./datasets/franka_wooden_board_dataset6/train"
# handTrainDir = "./datasets/sawyer_hurdle_dataset/train"
# handTrainDir = "./datasets/franka_wooden_board_segmentation_night3/train"
handTrainDir = "./datasets/franka_wooden_board_segmentation_newhouse_1/train"
# handTrainDir = "./datasets/sawyer_door_dataset/train"
# handTrainDir = "./datasets/sawyer_pickup_dataset/train"
# handTrainDataset = FrankaSegmentationDataset_48x48(handTrainDir)
# handTrainDataset = FrankaSegmentationDataset(handTrainDir)
handTrainDataset = FrankaSegmentationDataset_withDistractor(handTrainDir, distractor_type='bluePuck', imsize=256)
# handTrainDataLoader = DataLoader(handTrainDataset, batch_size=4, shuffle=True, num_workers=5)
handTrainDataLoader = DataLoader(handTrainDataset, batch_size=4, shuffle=True, num_workers=1)

# handValDir = "./datasets/franka_wooden_board_dataset6/val"
handValDir = "./datasets/franka_wooden_board_segmentation_newhouse_1/val"
# handValDir = "./datasets/sawyer_door_dataset/val"
# handValDir = "./datasets/sawyer_pickup_dataset/val"
# handValDataset = FrankaSegmentationDataset_48x48(handValDir)
# handValDataset = FrankaSegmentationDataset(handValDir)
handValDataset = FrankaSegmentationDataset_withDistractor(handValDir, distractor_type='bluePuck', imsize=256)
handValDataLoader = DataLoader(handValDataset, batch_size=2, shuffle=False, num_workers=5)

print(handTrainDir.__len__())
print(handValDir.__len__())

model = unet.UNet(n_channels=3, n_classes=1)
model = model.float()
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.BCELoss()

trainWriter = SummaryWriter("logs/train")
valWriter = SummaryWriter("logs/val")

#model.load_state_dict(torch.load("./pytorchmodel_epoch4"))

trainIter = 1
valIter = 1
globalIter = 1
for epoch in range(31):
    epochTrainLoss = 0
    model.train()
    print("training started")
    # for batch, (image, mask_true) in enumerate(handTrainDataLoader):
    for batch, (image, depth, mask_true) in enumerate(handTrainDataLoader):

        # temp_rgb = image.detach().numpy().squeeze()
        # temp_rgb = np.rollaxis(temp_rgb, 0, 3)
        # temp_mask = mask_true.detach().numpy().squeeze()
        # print(temp_rgb.shape)
        # print(temp_mask.shape)
        # print(image.shape)

        image = image.cuda()
        mask_true = mask_true.cuda().float()
        mask_pred = model(image.float())

        temp_mask_pred = mask_pred.detach().cpu().numpy().squeeze()
        # cv2.imshow("temp_rgb", temp_rgb[0])
        # cv2.waitKey()
        # cv2.imshow("temp_mask", temp_mask*255)
        # cv2.imshow("temp_mask_pred", temp_mask_pred)
        # cv2.waitKey()

        loss = criterion(mask_pred, mask_true)
        epochTrainLoss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: %d\tBatch: %d\tTraining Loss: %f" % (epoch, batch, loss.detach().cpu().item()))
        trainWriter.add_scalar('train_loss', loss.detach().cpu().item(), trainIter)
        trainWriter.add_scalar('train_val_loss_combined', loss.detach().cpu().item(), globalIter)
        #trainWriter.flush()
        trainIter += 1; globalIter += 1


    trainWriter.add_scalar('epoch_train_loss', epochTrainLoss/(batch+1), epoch)
    if(epoch%10 == 0):
        # torch.save(model.state_dict(), "saved_models/pytorchmodel_franka_wooden_segmentation6_48x48_epoch"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        # torch.save(model.state_dict(), "saved_models2/pytorchmodel_franka_wooden_segmentation6_distractors_epoch"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        # torch.save(model.state_dict(), "saved_models2/pytorchmodel_franka_wooden_segmentationdataset_distractors_segmentation_480x480_"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        # torch.save(model.state_dict(), "saved_models2/pytorchmodel_sawyer_door_segmentation_distractors_48x48_"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        # torch.save(model.state_dict(), "saved_models2/pytorchmodel_sawyer_pickup_segmentation_48x48_"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        # torch.save(model.state_dict(), "saved_models2/pytorchmodel_franka_wooden_segmentation_night1_distractors_48x48_epoch"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        # torch.save(model.state_dict(), "saved_models2/pytorchmodel_franka_wooden_segmentation_newhouse_distractors_480x480_epoch"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        # torch.save(model.state_dict(), "saved_models2/pytorchmodel_franka_wooden_segmentation_newhouse_distractors_48x48_epoch"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        # torch.save(model.state_dict(), "saved_models2/pytorchmodel_franka_wooden_segmentation_night3_distractors_48x48_epoch"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))
        torch.save(model.state_dict(), "saved_models2/pytorchmodel_franka_wooden_segmentation_night3_distractors_256x256_epoch"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S"))

    print("Validation started...")
    epochValLoss = 0
    model.eval()
    with torch.no_grad():
        # for batch, (image, mask_true) in enumerate(handValDataLoader):
        for batch, (image, depth, mask_true) in enumerate(handValDataLoader):

            image = image.cuda()
            mask_true = mask_true.cuda().float()
            mask_pred = model(image.float())

            loss = criterion(mask_pred, mask_true)
            epochValLoss += loss

            print("Epoch: %d\tBatch: %d\tValidation Loss: %f" % (epoch, batch, loss.detach().cpu().item()))
            valWriter.add_scalar('val_loss', loss.detach().cpu().item(), valIter)
            valWriter.add_scalar('train_val_loss_combined', loss.detach().cpu().item(), globalIter)
            #valWriter.flush()
            valIter += 1; globalIter += 1

        valWriter.add_scalar('epoch_val_loss', epochValLoss/(batch+1), epoch)

    trainWriter.close()
    valWriter.close()

