import os
import random
import tensorflow as tf
import numpy as np
import torch
from torchvision import datasets, transforms
import PIL
from PIL import Image
import torch.nn as nn
import tensorflow
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

LR = 1e-4
N_EPOCHS = 50
BATCH_SIZE = 4
PATIENCE = 5
DROPOUT = .2

class RpsClassifier(nn.Module):
    def __init__(self):
        super(RpsClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=DROPOUT, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=DROPOUT, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=DROPOUT, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=75, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(126*76, 120),
            nn.ReLU(),
            nn.Linear(120, 7),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        x=x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x
    def conv_block(self, c_in, c_out, dropout, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
            )
        return seq_block

def predict(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.Resize((800, 600)), # full size is 1600x1200, but that causes memory errors
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    loaded_model = RpsClassifier()
    loaded_model.load_state_dict(torch.load('model_DEnglish83.pt'))
    y_pred = torch.from_numpy(np.array([]))

    for img in x:
        image = Image.open(img)
        image = np.asarray(transform(image))
        trans = torch.from_numpy(image)
        trans.unsqueeze_(0)
        output = loaded_model(trans)
        temp = np.resize(np.asarray(output.detach()), (7))
        temp2 = torch.from_numpy(temp)
        for i in range(len(temp)):
            if temp[i] >= .5:
                temp[i] = 1
            else:
                temp[i] = 0

        if len(y_pred):
            y_pred = np.vstack((y_pred, np.asarray(torch.flatten(temp2))))
        else:
            y_pred = np.asarray(torch.flatten(temp2), dtype='int64')

    y_pred2 = torch.from_numpy(y_pred).type(torch.float)
    return y_pred2
