import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing	

class M1(nn.Module):
    def __init__(self):
        super(Encode,self).__init__()
        self.conv1 = nn.Conv1d(3,3,10)
        self.conv2 = nn.Conv1d(3,3,10)

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0,2,1)
        return x

device = 'cuda' if torch.cuda_is_available() else 'cpu'
# defining the model
model = M1()
# defining the optimizer
optimizer = optim.Adam(model.parameters())
# defining the loss function
criterion = nn.L1Loss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
epochs = 10

for i in range(epochs):
  arr = torch.empty((0,1,3)).to(device)
  outputs = []
  losses = 0.0
  for data in SAPdataloader:
  	if torch.cuda.is_available():
    	data = data.to(device)

    optimizer.zero_grad()

    output = model(data)
    #print(output.shape)
    L = criterion(output, data)
    #print(output.shape,data.shape)
    L.backward()

    optimizer.step()

    output = output.unsqueeze(1)

    arr = torch.vstack((arr,output))

    losses += L.item()
  
  if i%2 == 0:
    # printing the loss
  	print('Epoch : ',epoch+1, '\t', 'loss :', losses)

print("Training Over")
torch.save(model.state_dict(),'net.bin')