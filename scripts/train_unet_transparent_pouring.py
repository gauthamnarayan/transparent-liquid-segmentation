import os
import sys
import time
import argparse

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter

import unet
from data_loader import TransparentPouringSegmentationDataset

def train_model(args, train_dataloader, val_dataloader=None):

    print("Mode : ", args.mode)
    if args.mode == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

    trainWriter = SummaryWriter("logs/train")
    if args.do_validation:
        valWriter = SummaryWriter("logs/val")

    trainIter = 1
    valIter = 1
    globalIter = 1
    for epoch in range(11):
        epochTrainLoss = 0
        model.train()
        print("training started")
        for batch, (image, mask_true) in enumerate(train_dataloader):

            # temp_rgb = image[0].detach().numpy().squeeze()
            # temp_rgb = np.rollaxis(temp_rgb, 0, 3)
            # temp_mask = mask_true[0].detach().numpy().squeeze()
            # print(temp_rgb.shape)
            # print(temp_mask.shape)
            # print(image.shape)
            # cv2.imshow("temp_rgb", temp_rgb)
            # cv2.imshow("temp_mask", temp_mask*255)
            # cv2.waitKey()

            image = image.to(device)
            mask_true = mask_true.to(device).float()
            mask_true = torch.squeeze(mask_true)
            mask_pred = model(image.float())
            mask_pred = torch.squeeze(mask_pred)

            temp_mask_pred = mask_pred.detach().cpu().numpy().squeeze()
            # print(temp_mask_pred.shape)
            # cv2.imshow("temp_rgb", temp_rgb)
            # cv2.imshow("temp_mask", temp_mask)
            # cv2.imshow("temp_mask_pred", temp_mask_pred[0])
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
        if(epoch%5 == 0):
            torch.save(model.state_dict(), os.path.join(args.save_dir, 
                                                        "transparent_liquid_segmentation_unet_150x300_epoch_" 
                                                        + str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S")))

        if(args.do_validation):
            print("Validation started...")
            epochValLoss = 0
            model.eval()
            with torch.no_grad():
                for batch, (image, mask_true) in enumerate(val_dataloader):
                    
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
    if args.do_validation:
        valWriter.close()

def main(args):

    # train_dir = "../data/datasets/pouring_dataset/"
    train_dataset = TransparentPouringSegmentationDataset(args.train_data_dir, img_size=(150, 300), transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    train_model(args, train_dataloader)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-dir', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--mode', default='gpu')
    parser.add_argument('--do-validation', default=False)
    args = parser.parse_args()
    
    main(args)