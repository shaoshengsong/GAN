
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import numpy as np
import os


# Hyper Parameters
EPOCH_COUNT = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
TEST_IMG_COUNT =5 #test  image count ,encoded_data.shape's shape

# MNIST dataset

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=False,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)



test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=False,transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

# torchvision.datasets.MNIST
# root（string）– 数据集的根目录，其中存放processed/training.pt和processed/test.pt文件。
# train（bool, 可选）– 如果设置为True，从training.pt创建数据集，否则从test.pt创建。
# download（bool, 可选）– 如果设置为True, 从网上下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
# transform（可被调用 , 可选）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：transforms.RandomCrop。
# target_transform （可被调用 , 可选）– 一种函数或变换，输入目标，进行变换。
    
    
# torchvision.transforms.ToTensor
# 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    
#torch.utils.data.DataLoader
#dataset（Dataset) – 要加载数据的数据集。
#batch_size（int, 可选) – 每一批要加载多少数据（默认：1）。
#shuffle（bool, 可选) – 如果每一个epoch内要打乱数据，就设置为True（默认：False）。


    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       #  range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


model = AutoEncoder()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()



def train(epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        input_data = data.view(-1, 28*28)
        encoded, decoded = model(input_data)

        loss = criterion(decoded, input_data)      
        optimizer.zero_grad()             
        loss.backward()                     
        optimizer.step()                   

        if batch_idx % 5 == 0 :
            print('Epoch: ', epoch, '| train loss: %.6f' % loss.data.numpy())



def test(epoch):
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            #data = data.to(device)
            input_data = data.view(-1, 28*28)
            encoded, decoded  = model(input_data)
            loss = criterion(decoded, input_data)
            if i == 0:
                n = min(data.size(0), TEST_IMG_COUNT)
                print(n)
                #(data.shape)#([64, 1, 28, 28])
                comparison = torch.cat([data[:n],
                                      decoded.view(BATCH_SIZE,-1,28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/ae_' + str(epoch) + '.png', nrow=n)
                
                
if __name__ == "__main__":
   for epoch in range(1, EPOCH_COUNT + 1):
       train(epoch)
       test(epoch)               
                
                
                
                
                