import os
import random
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

LR = 1e-4
N_EPOCHS = 500
BATCH_SIZE = 4
PATIENCE = 20
DROPOUT = .2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(101)
np.random.seed(101)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x_train_load = torch.load('x_train.pt')

x_np = np.array([])
i = 0
for obs in x_train_load:
    if len(x_np):
        x_np = np.append(x_np, np.array([np.asarray(obs[0])]), axis = 0)
    else:
        x_np=np.array([np.asarray(obs[0])])
    i+= 1
print('Objects:', i)

x_train = torch.from_numpy(x_np).to(device)
x_train.requires_grad = True
#print('Xtrain shape:', x_train.shape)


del x_train_load
del x_np


x_valid_load = torch.load('x_valid.pt')
x_np = np.array([])
i = 0
for obs in x_valid_load:
    if len(x_np):
        x_np = np.append(x_np, np.array([np.asarray(obs[0])]), axis = 0)
    else:
        x_np=np.array([np.asarray(obs[0])])
    i+= 1
print('Objects:', i)

x_valid = torch.from_numpy(x_np).to(device)

del x_valid_load
del x_np

y_train_load = torch.load('y_train.pt')
y_train = torch.from_numpy(np.asarray(y_train_load)).to(device)
#print(y_train.shape)

y_valid_load = torch.load('y_valid.pt')
y_valid = torch.from_numpy(np.asarray(y_train_load)).to(device)
#print(y_valid.shape)

model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
model.classifier[1] = nn.Conv2d(512, 7, kernel_size=(1,1), stride=(1,1))
model.to(device)
#print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

epoch_loss_mem = 999999
in_a_row = 0

print("Begin training.")
for e in range(N_EPOCHS):
    train_epoch_loss = 0
    valid_epoch_loss = 0
    model.train()
    for batch in range(len(x_train) // BATCH_SIZE + 1):
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        #print('x_train[inds] shape: ', x_train[inds].shape)
        y_train_pred = model(x_train[inds]).squeeze()
        #print('y_train_pred shape: ', y_train_pred.shape)
        y_train_pred = y_train_pred.type(torch.cuda.FloatTensor).detach()
        y_train_sub = y_train[inds].type(torch.cuda.FloatTensor)
        y_train_sub.requires_grad = True
        train_loss = criterion(y_train_sub, y_train_pred)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()

    print('Epoch:', e)
    print('Loss:', train_epoch_loss)

    BATCH_SIZE_VALID = 2
    for batch in range(len(x_valid) // BATCH_SIZE_VALID):
        inds = slice(batch * BATCH_SIZE_VALID, (batch + 1) * BATCH_SIZE_VALID)
        y_valid_pred = model(x_valid[inds]).squeeze()
        y_vali_pred = y_valid_pred.type(torch.cuda.FloatTensor).detach()
        y_valid_sub = y_valid[inds].type(torch.cuda.FloatTensor)
        y_valid_sub.requires_grad = True
        train_loss = criterion(y_valid_sub, y_valid_pred)
        valid_epoch_loss += train_loss.item()

    print('Valid loss this run:', valid_epoch_loss)
    print('Valid loss last run:', epoch_loss_mem)
    if valid_epoch_loss > epoch_loss_mem:
        in_a_row += 1
        if in_a_row >= PATIENCE:
            print('Early Stop, Epoch', e)
            break
        else:
            print('Valid Accuracy Decreased for', in_a_row,'Run(s)')
    else:
        epoch_loss_mem = valid_epoch_loss
        in_a_row = 0

torch.save(model.state_dict(), '/home/ubuntu/model_DEnglish83.pt')
torch.save(model.state_dict(), '/home/ubuntu/model_DEnglish83_Day7.pt')

