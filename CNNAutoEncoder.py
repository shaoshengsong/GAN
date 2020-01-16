

import torch
import torchvision
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import os



# Hyper Parameters
EPOCH_COUNT = 20
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
TEST_IMG_COUNT =5 #test 5 image ,encoded_data.shape's shape



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=False,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=False,transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
model = AutoEncoder()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()


def train(epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):# data,_=(input,lable)   data=(b*channel*width*height)
        optimizer.zero_grad()               
        encoded, decoded = model(data)
        loss = criterion(decoded, data)           
        loss.backward()                    
        optimizer.step()                 

        if batch_idx % 5 == 0 :
            print('Epoch: ', epoch, '| train loss: %.6f' % loss.data.numpy())

def test(epoch):
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            #data = data.to(device)
            encoded, decoded  = model(data)
            loss = criterion(decoded, data)
            if i == 0:
                n = min(data.size(0), TEST_IMG_COUNT)
                print(n)
                comparison = torch.cat([data[:n],
                                      decoded.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                

            
    
    
if __name__ == "__main__":
    for epoch in range(1, EPOCH_COUNT + 1):
        train(epoch)
        test(epoch)