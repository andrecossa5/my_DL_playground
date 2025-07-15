"""
Convolutional Neural Network (CNN) for image classification (LeNet 5, example).
"""

import torch
import torchvision
import matplotlib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import plotting_utils as plu
from torch.utils.data import random_split, DataLoader
matplotlib.use('MacOSX')


##


# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),                     # 28×28→ tensor [0.0–1.0]
    transforms.Normalize((0.5), (0.5,))       # mean=0.5, std=0.5 for the single channel
])

# Datasets & DataLoaders
trainset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
testset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Look at classes
classes = trainset.classes  # ['T-shirt/top','Trouser',…,'Ankle boot']
classes


##


# NN
class NN(nn.Module):

    def __init__(self):
        super().__init__()
        # Conv layers: 1→16→32 channels, 3×3 kernels, pad=1 to keep spatial dims
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)   # 2×2 pooling, stride=2
        # After two poolings: 28→14→7, so features = 32 × 7 × 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)     # 10 fashion classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 1×28×28 → 16×28×28 → 16×14×14
        x = self.pool(F.relu(self.conv2(x)))  # 16×14×14 → 32×14×14 → 32×7×7
        x = x.view(x.size(0), -1)             # flatten to (batch, 1568)
        x = F.relu(self.fc1(x))               # → (batch,256)
        x = F.relu(self.fc2(x))               # → (batch,128)
        x = self.fc3(x)                       # → (batch,10)
        return x


##


# Training loop
def train(model, trainset, optimizer, loss_fn, 
          batch_size, n_epochs, device, random_state, train_size):
    """
    Basic trainer loop.
    """
    
    rng = torch.manual_seed(random_state)
    train_size = round(len(trainset) * .8)
    valid_size = len(trainset) - train_size
    trainset, validset = random_split(trainset, [train_size, valid_size], generator=rng)
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, num_workers=1, shuffle=False)
    train_stats = {'accuracy' : [], 'loss' : []}
    val_stats = {'accuracy' : [], 'loss' : []}

    for epoch in range(n_epochs):

        model.train()
        acc, loss_, count = 0, 0, 0
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            loss_ += loss.item() * X.size(0)
            y_pred = torch.max(out,1)[1]
            acc += (y_pred == y).sum().item()
            count += X.size(0)
        
        train_stats['accuracy'].append(acc/count) 
        train_stats['loss'].append(loss_/count)       
        
        model.eval()
        acc, loss_, count = 0, 0, 0
        with torch.no_grad():
            for X, y in validloader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss_ += loss_fn(out, y).item() * X.size(0)
                y_pred = torch.max(out,1)[1]
                acc += (y_pred == y).sum().item()
                count += X.size(0)
        
        val_stats['accuracy'].append(acc/count) 
        val_stats['loss'].append(loss_/count)       

        tl = train_stats['loss'][epoch]
        vl = val_stats['loss'][epoch]
        ta = train_stats['accuracy'][epoch]
        va = val_stats['accuracy'][epoch]

        print(f'Epoch: {epoch}. Mean train loss: {tl: 3f}, Mean validate loss: {vl: 3f}')
        print(f'Epoch: {epoch}. Mean train accuracy: {ta: 2f}, Mean validate accuracy: {va: 2f}')

    return pd.DataFrame(train_stats), pd.DataFrame(val_stats)


##


# Test loop
def test(model, testset, loss_fn, batch_size, device):

    test_stats = {'accuracy' : [], 'loss' : []}
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss_ = loss_fn(out, y).item() * X.size(0)
            y_pred = torch.max(out,1)[1]
            acc = (y_pred == y).sum().item()
            count = X.size(0)
            test_stats['accuracy'].append(acc/count) 
            test_stats['loss'].append(loss_/count)  

    return pd.DataFrame(test_stats)



# Instantiate model on device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NN().to(device)
model

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
loss_fn = nn.CrossEntropyLoss()
batch_size = 164
n_epochs = 10
random_state = 1234
train_size = .8
train_stats, val_stats = train(model, trainset, optimizer, loss_fn, 
                               batch_size, n_epochs, device, random_state, train_size)

# Test
test_stats = test(model, testset, loss_fn, batch_size, device)
test_stats


# Viz
fig, axs = plt.subplots(1,2,figsize=(6, 3.2), gridspec_kw={'width_ratios': [3,1]}, sharey=True)

colors = {'train':'darkorange', 'validate': 'green'}
axs[0].plot(train_stats['accuracy'], marker='x', color=colors['train'])
axs[0].plot(val_stats['accuracy'], marker='x', color=colors['validate'])
axs[0].set_ylim((.7,1))
plu.format_ax(ax=axs[0], title='Training accuracy', xlabel='Epochs', ylabel='Avg accuracy')
plu.add_legend(colors=colors, ax=axs[0], loc='lower right', bbox_to_anchor=(1,0), label='Set')

plu.violin(test_stats.assign(set='test'), x='set', y='accuracy', ax=axs[1], color='white')
plu.strip(test_stats.assign(set='test'), x='set', y='accuracy', ax=axs[1], color='darkgreen')
axs[1].set_ylim((.7,1))
plu.format_ax(ax=axs[1], title='Test accuracy', xlabel='', ylabel='Accuracy')

fig.tight_layout()
plt.show()


##