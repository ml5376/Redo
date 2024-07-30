import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from util import show_all_keypoints,save_model,download_url
from facialkeypoints_data import FacialKeypointsDataset
from nn import (DummyKeypointModel,KeypointModel)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

download_url = 'https://vision.in.tum.de/webshare/g/i2dl/facial_keypoints.zip'
i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
data_root = os.path.join(i2dl_exercises_path, "datasets", "facial_keypoints")

train_dataset = FacialKeypointsDataset(
    train=True,
    transform=transforms.ToTensor(),
    root=data_root,
    download_url=download_url
)
val_dataset = FacialKeypointsDataset(
    train=False,
    transform=transforms.ToTensor(),
    root=data_root,
)

loss_fn = torch.nn.MSELoss()

hparams = {
    # TODO: if you have any model arguments/hparams, define them here
    "batch_size":20,
    "n_hidden": 256,
    "learning_rate":1.3e-3
}  

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=0)

model = KeypointModel(hparams)
model.to(device)

import torchsummary
torchsummary.summary(model,(1,96,96))

import torch.nn as nn

cri = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)


num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for i,data in enumerate(train_loader,1):
        images = data['image']
        key_pts = data['keypoints']

        # flatten pts
        key_pts = key_pts.view(key_pts.size(0), -1)

        # convert variables to floats for regression loss
        labels = key_pts.type(torch.FloatTensor).to(device)
        inputs = images.type(torch.FloatTensor).to(device)
        outputs = model(inputs)
        loss = cri(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练完成！")


def show_keypoint_predictions(model, dataset, num_samples=3):
    for i in range(num_samples):
        image = dataset[i]["image"]
        key_pts = dataset[i]["keypoints"].to(device)

        print(image.shape)
        image = image.type(torch.FloatTensor).to(device)
        out=model(image)
        print(out.shape)
        predicted_keypoints = torch.squeeze(model(image).detach()).view(15,2)
   
        print(predicted_keypoints.shape)
        show_all_keypoints(image, key_pts, predicted_keypoints)

show_keypoint_predictions(model, val_dataset)