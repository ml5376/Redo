"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams

        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(batch_size, 1, 96, 96), output:(batch_size, 64, 96, 96), (96-3+2*1)/1+1 = 96
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1), # 卷积层
            #nn.BatchNorm2d(num_features=12), # 归一化
            nn.RReLU(inplace=True), # 激活函数
            # output(batch_size, 64, 24, 24)
            nn.MaxPool2d(kernel_size=2, stride=2), # 最大值池化

            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(num_features=24),
            nn.RReLU(inplace=True),
            # output:(batch_size, 128, 24 ,24)
            nn.MaxPool2d(kernel_size=2, stride=2),

        )


        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            # input:(batch_size, 64, 48, 48), output:(batch_size, 128, 48, 48), (48-3+2*1)/1+1 = 48
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(num_features=24),
            nn.RReLU(inplace=True),
            # output:(batch_size, 128, 24 ,24)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        # 全连接层
        self.fc = nn.Sequential(
            
            nn.Linear(24*24*24,256),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 30),
        )



    def forward(self, x):

        x = self.conv1(x)
       
        x = x.view(-1).unsqueeze(0)
        
        y = self.fc(x)
      

        return y







class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)

      