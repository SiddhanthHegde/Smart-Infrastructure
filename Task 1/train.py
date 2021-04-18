import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing	

class Encode(nn.Module):
    def __init__(self):
        super(Encode,self).__init__()
        self.lstm = nn.LSTM(3,1)
        self.fc1 = nn.Linear(20,10)
        self.fc2 = nn.Linear(10,1)

    def forward(self,x):
        output, _ = self.lstm(x)
        x = output.squeeze(2)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Decode(nn.Module):
    def __init__(self):
        super(Encode,self).__init__()
        self.convt1 = nn.ConvTranspose1d(1,2,10)
        self.convt2 = nn.ConvTranspose1d(2,3,11)

    def forward(self,x):
        x = x.unsqueeze(2)
        x = self.convt1(x)
        x = self.convt2(x)
        x = x.permute(0,2,1)
        return x

class Compressor(nn.Module):
  def __init__(self):
    super(Compressor,self).__init__()
    self.enc = Encode()
    self.dec = Decode()

  def forward(self,x):
    encoded = self.enc(x)
    decoded = self.dec(encoded)

    return encoded, decoded


# defining the model
model = Compressor()
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

    output = net(data)
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
torch.save(net.state_dict(),'net.bin')