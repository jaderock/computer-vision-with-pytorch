import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split as split
import torch.optim.lr_scheduler as lr_scheduler


import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
from sklearn.metrics import confusion_matrix

n_epochs = 30
batch_size = 100
img_size = 32
img_channels = 3
n_class = 10

lr = 5e-3
n_train = 40000
n_val = 10000
s_p_e = int(n_train/batch_size)    # steps per epoch = 400 for lr_shceduler

data_path = './OCR/data/cifar10'

class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

# CIFAR-10 Datasets
dataset = CIFAR10(data_path, train=True, download=True,
                  transform=T.Compose([
                    T.RandomCrop(img_size, padding=4, padding_mode='reflect'),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                  ]))
test_dataset = CIFAR10(data_path, train=False, download=True,
                       transform=T.Compose([
                         T.ToTensor(),
                         T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                         ]))
train_dataset, val_dataset = split(dataset, [n_train, n_val], generator=torch.manual_seed(999))

# Slice datasets into dataloaders
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for imgs, labels in train_dl:
    print(labels.view(-1, 10))
    print('imgs.shape=', imgs.shape)
    break    # pick out the 1st batch of the train dataloader

# Show images in a batch of dataloader
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5
def show_imgs(images):
    fig, ax = plt.subplots(figsize=(16, 12))
    inputs = make_grid(denorm(images), nrow=10, padding=1)
    ax.imshow(inputs.permute(1, 2, 0))
    ax.set(xticks=[], yticks=[])
    plt.show()
show_imgs(imgs)


# ResNet9 model, loss_fn, optimizer and learning rate scheduler
def basic(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Rs_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.resnet = nn.Sequential(
            basic(in_channels, in_channels),
            basic(in_channels, in_channels)
        )
    def forward(self, x):
        return x + self.resnet(x)

class ResNet9(nn.Module):
    def __init__(self, in_channels=img_channels, num_classes=n_class):
        super().__init__()
        self.net = nn.Sequential(
            basic(in_channels, 64),             # batch_size * 64 * 32^2
            basic(64, 128), nn.MaxPool2d(2),    # batch_size * 128 * 16^2
            Rs_Block(128),                      # batch_size * 128 * 16^2
            basic(128, 256), nn.MaxPool2d(2),   # batch_size * 256 * 8^2
            basic(256, 512), nn.MaxPool2d(2),   # batch_size * 512 * 4^2
            Rs_Block(512),                      # batch_size * 512 * 4^2
            nn.MaxPool2d(4),                    # batch_size * 512 * 1^2
            nn.Flatten(),
            nn.Linear(512, num_classes)         # batch_size * 10
        )
    def forward(self, images):
        output = self.net(images)    # images.shape = batch_size * 3 * 32^2
        return output                # output.shape = batch_size * 10

model = ResNet9() #.cuda()
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = lr_scheduler.OneCycleLR(optimizer,
                                    max_lr=lr,
                                    steps_per_epoch=s_p_e,
                                    epochs=n_epochs,
                                    pct_start=0.4)

# Model training for the train_dataloader
def training(my_dataloader):
    total_loss = 0.0
    n_correct = 0.0
    n_samples = len(my_dataloader.dataset)
    model.train()
    for i, (images, labels) in enumerate(my_dataloader):
        #labels = labels.cuda(non_blocking=True)
        outputs = model(images)    # model(images.cuda(non_blocking=True))
        predictions = torch.argmax(outputs, dim=1)
        n_correct += torch.sum(predictions==labels).item()
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss/n_samples, 100*n_correct/n_samples


# Model evaluation for val_dataloader and test_dataloader
def evaluation(my_dataloader):
    n_samples = len(my_dataloader.dataset)
    with torch.no_grad():
        total_loss = 0.0
        n_correct = 0
        model.eval()
        for i, (images, labels) in enumerate(my_dataloader):
            # labels = labels.cuda(non_blocking=True)
            outputs = model(images)    # outputs=model(images.cuda(non_blocking=True))
            predictions = torch.argmax(outputs, dim=1)
            n_correct += torch.sum(predictions==labels).item()
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss/n_samples, 100*n_correct/n_samples

# The main training function
def fitting(epochs):
    df = pd.DataFrame(np.empty([epochs, 5]),
                      index = np.arange(epochs),
                      columns=['loss_train', 'acc_train', 'loss_val', 'acc_val', 'lr'])
    progress_bar = trange(epochs)
    for i in progress_bar:
        df.iloc[i,0], df.iloc[i,1] = training(train_dl)
        df.iloc[i,2], df.iloc[i,3] = evaluation(val_dl)
        df.iloc[i,4] = optimizer.param_groups[0]['lr']
        progress_bar.set_description("train_loss=%.5f" % df.iloc[i,0])
        progress_bar.set_postfix({'train_acc':df.iloc[i,1], 'test_acc':df.iloc[i,3]})
    return df

train_history = fitting(n_epochs)


# Graphs of Model training outputs
df = train_history
fig, ax = plt.subplots(1, 3, figsize=(12,3), sharex=True)
df.plot(ax=ax[0], y=[1,3], style=['r-+', 'b-d'])
df.plot(ax=ax[1], y=[0,2], style=['r-+', 'b-d'])
df.plot(ax=ax[2], y=[4], style=['r-+'])
for i in range(3):
    ax[i].set_xlabel('epoch')
    ax[i].grid(which='major', axis='both', color='g', linestyle=':')
ax[0].set_ylabel('accuracy(%)')
ax[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))


# Using test dataloader to test the trained model
n = 580
loss, accuracy = evaluation(test_dl)
print('test dataloader accuracy (%) = ', accuracy)
img, label = test_dataset[n]
def predict_image(img, model):
    print('img.shape=', img.shape)
    x = img.unsqueeze(0) #.cuda()
    print('img.unsqueeze(0).shape=', x.shape)
    y_hat = model(x)
    print('y_hat=', y_hat)
    pred_idx = torch.argmax(y_hat, dim=1)
    print('pred_idx=', pred_idx.item())
    return dataset.classes[pred_idx[0].item()]

print('Prediction is: {0}; label is {1}'.format(
    predict_image(img, model), dataset.classes[label]
))
plt.imshow(denorm(img).permute(1, 2, 0))

# save a trained model
model_file_name = './models/teacher.pth'
torch.save(model, model_file_name)
