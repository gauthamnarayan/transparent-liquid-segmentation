import os
import sys
import numpy as np
import cv2
import time
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import argparse

import unet
from dataLoader import TransparentPouringSegmentationDataset
from dataLoader import TransparentPouringSegmentationDataset_zeropadded
from dataLoader import TransparentPouringSegmentationDataset_verticalCrop

def train_unet(params):
    """
    Train a UNet model
    
    Args:
    params: training parameters
    """
    if params.mode == 'local':
        device = torch.device("cpu")
    elif params.mode == 'cluster':
        device = torch.device("cuda")

    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

    trainWriter = SummaryWriter("logs/train")
    valWriter = SummaryWriter("logs/val")

    trainIter = 1
    valIter = 1
    globalIter = 1
    for epoch in range(11):
        epochTrainLoss = 0
        model.train()
        print("training started")
        
        for batch, (image, depth, mask_true) in enumerate(params["trainDataLoader"]):
            image = image.to(device)
            mask_true = mask_true.to(device).float()
            mask_true = torch.squeeze(mask_true)
            mask_pred = model(image.float())
            mask_pred = torch.squeeze(mask_pred)

            temp_mask_pred = mask_pred.detach().cpu().numpy().squeeze()
            loss = criterion(mask_pred, mask_true)
            epochTrainLoss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: %d\tBatch: %d\tTraining Loss: %f" % (epoch, batch, loss.detach().cpu().item()))
            trainWriter.add_scalar('train_loss', loss.detach().cpu().item(), trainIter)
            trainWriter.add_scalar('train_val_loss_combined', loss.detach().cpu().item(), globalIter)
            trainWriter.flush()
            trainIter += 1; globalIter += 1


        trainWriter.add_scalar('epoch_train_loss', epochTrainLoss/(batch+1), epoch)
        if(epoch%5 == 0):
            torch.save(model.state_dict(), os.path.join(params["model_save_dir"], "epoch_"+str(epoch) + time.strftime("_%Y%m%d_%H_%M_%S")))

        print("Validation started...")
        model.eval()
        with torch.no_grad():
            for batch, (image, depth, mask_true) in enumerate(params["valDataLoader"]):

                image = image.cuda()
                mask_true = mask_true.cuda().float()
                mask_pred = model(image.float())

                loss = criterion(mask_pred, mask_true)
                epochValLoss += loss

                print("Epoch: %d\tBatch: %d\tValidation Loss: %f" % (epoch, batch, loss.detach().cpu().item()))
                valWriter.add_scalar('val_loss', loss.detach().cpu().item(), valIter)
                valWriter.add_scalar('train_val_loss_combined', loss.detach().cpu().item(), globalIter)
                valWriter.flush()
                valIter += 1; globalIter += 1

            valWriter.add_scalar('epoch_val_loss', epochValLoss/(batch+1), epoch)

    trainWriter.close()
    valWriter.close()


def main(params):

    transform = transforms.Compose([transforms.ColorJitter(brightness=params["brightness"], contrast=params["contrast"], hue=params["hue"])])
    trainDataset = TransparentPouringSegmentationDataset_Kai_robot_pouring_dataset_150x300(params["train_data_dir"], transform=transform)
    trainDataLoader = DataLoader(trainDataset, batch_size=4, shuffle=True, num_workers=4)

    valDataset = TransparentPouringSegmentationDataset_Kai_robot_pouring_dataset_150x300(params["val_data_dir"], transform=transform)
    valDataLoader = DataLoader(valDataset, batch_size=4, shuffle=False, num_workers=4)

    params["trainDataLoader"] = trainDataLoader
    params["valDataLoader"] = valDataLoader

    train_unet(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local', dest='mode',
                        help='directory to the training data')
    parser.add_argument('--val-data-dir', type=str, dest='val_data_dir',
                        help='directory to the validation data')
    parser.add_argument('--train-data-dir', type=str, dest='train_data_dir',
                        help='directory to the training data')
    parser.add_argument('--model-save-dir', type=int, default=300, dest='model_save_dir',
                        help='directory to save trained models')
    args = parser.parse_args()

    params = {}
    params["brightness"] = 0.1
    params["contrast"] = 0.3
    params["hue"] = 0.5
    params["mode"] = args.mode
    params["train_data_dir"] = args.train_data_dir
    params["val_data_dir"] = args.val_data_dir
    params["model_save_dir"] = args.model_save_dir
    
    main(params)
