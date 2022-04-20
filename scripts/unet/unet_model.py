# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import cv2
import copy
import numpy as np

from .unet_parts import *

def show_filters(x, nImgs, name):
    import matplotlib.pyplot as plt
    x = x.cpu().detach()
    x = np.squeeze(x, axis=0)

    fig = plt.figure(name)
    for i in range(1, nImgs+1):
        ax = fig.add_subplot(int(np.sqrt(nImgs)), int(np.sqrt(nImgs)), i)
        if(x.shape[0]==1):
            j = 0
        else:
            j = np.random.randint(1, x.shape[0])
        ax.imshow(x[j], cmap='gray')

    plt.show()

class UNet_wider_1(nn.Module):
    def __init__(self, n_channels, n_classes, width_factor = 1):
        super(UNet_wider_1, self).__init__()
        self.inc = inconv(n_channels, 64 * width_factor)
        self.down1 = down(64 * width_factor, 128 * width_factor)
        self.down2 = down(128 * width_factor, 256 * width_factor)
        self.down3 = down(256 * width_factor, 512 * width_factor)
        self.down4 = down(512 * width_factor, 512 * width_factor)
        self.up1 = up(1024 * width_factor, 256 * width_factor)
        self.up2 = up(512 * width_factor, 128 * width_factor)
        self.up3 = up(256 * width_factor, 64 * width_factor)
        self.up4 = up(128 * width_factor, 64 * width_factor)
        self.outc = outconv(64 * width_factor, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)

# class UNet(nn.Module):
    # def __init__(self, n_channels, n_classes):
        # super(UNet, self).__init__()
        # self.inc = inconv(n_channels, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        # self.outc = outconv(64, n_classes)

    # def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)

        # # show_filters(copy.deepcopy(x1), 25, 'x1')
        # # show_filters(copy.deepcopy(x3), 49, 'x3')
        # # show_filters(copy.deepcopy(x5), 100, 'x5')
        
        # x = self.up1(x5, x4)
        # show_filters(copy.deepcopy(x), 100,'up1')
        # x = self.up2(x, x3)
        # # show_filters(copy.deepcopy(x), 64,'up2')
        # x = self.up3(x, x2)
        # show_filters(copy.deepcopy(x), 49 ,'up3')
        # x = self.up4(x, x1)
        # # show_filters(copy.deepcopy(x), 25 ,'up4')
        # x = self.outc(x)
        # show_filters(copy.deepcopy(x), 1 ,'up5')

        # print("Printing shapes")
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # print("")
        # # cv2.imshow("x1", x1)
        # # cv2.imshow("x2", x2)
        # # cv2.imshow("x3", x3)
        # # cv2.imshow("x4", x4)
        # # cv2.imshow("x5", x5)

        # return F.sigmoid(x)
