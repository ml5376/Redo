"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.functional as F
import numpy as np
import torchvision.models as models


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23,hparams=None):
        super().__init__()
        self.__hparams = hparams

        self.pretrained_model=models.alexnet(pretrained=True).features
        self.model=nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,23,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.Upsample(size=(240,240), mode="bilinear")
        )

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
     
        x=self.pretrained_model(x)
        x = self.model(x)
        
        return x


    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

      