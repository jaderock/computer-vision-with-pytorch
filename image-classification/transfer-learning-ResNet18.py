import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import make_grid
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

n_epochs = 10
img_size = (224, 224)
img_channels = 3
batch_size = 64
lr = 1e-2
data_path = './ImageSets/images/'

# datasets
train_dataset = ImageFolder(root=data_path+'train/',
                            transform=T.Compose([
                                T.Resize(img_size),
                                T.RandomVerticalFlip(0.5),
                                T.RandomHorizontalFlip(p=0.5),
                                T.ToTensor(),
                                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                            ]))
val_dataset = ImageFolder(root=data_path+'validation/',
                          transform=T.Compose([
                              T.Resize(img_size),
                              T.ToTensor(),
                              T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                          ]))
test_dataset = ImageFolder(root=data_path+'validation/',
                           transform=T.Compose([
                               T.Resize(img_size),
                               T.ToTensor(),
                               T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                           ]))
classes = train_dataset.classes
n_class = len(classes)
s_p_e = int(len(train_dataset)/batch_size)
n_samples = s_p_e * batch_size

# data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
n_batch = len(train_dataloader)

# display the 1st batch of the train_dataloader
for imgs, labels in train_dataloader:
    print(imgs.shape, '\nlabels=', labels)
    break
def denorm(img_tensors):    # shift image pixel values to [0,1]
    return img_tensors * 0.5 + 0.5
def show_imgs(images):
    fix, ax = plt.subplots(figsize=(16, 10))
    inputs = make_grid(denorm(images[:16]), nrow=8)
    ax.imshow(inputs.permute(1,2,0))
    ax.set(xticks=[], yticks=[])
show_imgs(imgs)

# Use a pre-trained ResNet
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
#print(model)
model.eval()
#for param in model.parameters():
#    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, n_class)
model = model.cuda()
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = lr_scheduler.OneCycleLR(optimizer,
                                    max_lr=lr,
                                    steps_per_epoch=s_p_e,
                                    epochs=n_epochs,
                                    pct_start=0.4)

# function to train the train_dataloader
def training(my_dataloader):
    total_loss = 0.0
    n_correct = 0.0
    n_samples = len(my_dataloader.dataset)
    model.train()
    for images, labels in my_dataloader:
        #labels = labels.cuda()
        outputs = model(images)
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

# evaluation
def evaluation(my_dataloader):
    n_samples = len(my_dataloader.dataset)
    with torch.no_grad():
        total_loss = 0.0
        n_correct = 0
        model.eval()
        for images, labels in my_dataloader:
            #labels = labels.cuda()
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            n_correct += torch.sum(predictions==labels).item()
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss/n_samples, 100*n_correct/n_samples

# main training function with pandas DataFrame
def fitting(epochs):
    df = pd.DataFrame(np.empty([epochs, 5]),
                      index=np.arange(epochs),
                      columns=['loss_train', 'acc_train', 'loss_val', 'acc_val', 'lr'])
    progress_bar = trange(epochs)
    for i in progress_bar:
        df.iloc[i,0], df.iloc[i,1] = training(train_dataloader)
        df.iloc[i,2], df.iloc[i,3] = evaluation(val_dataloader)
        df.iloc[i,4] = optimizer.param_groups[0]['lr']
        progress_bar.set_description("train_loss=%.5f" % df.iloc[i,0])
        progress_bar.set_postfix({'train_acc':df.iloc[i,1], 'val_acc':df.iloc[i,3]})
    return df

train_history = fitting(n_epochs)

# graphs of model training outputs
df = train_history
fig, ax = plt.subplots(1,3, figsize=(12,3), sharex=True)
df.plot(ax=ax[0], y=[1,3], style=['r-+', 'b-d'])
df.plot(ax=ax[1], y=[0,2], style=['r-+', 'b-d'])
df.plot(ax=ax[2], y=[4], style=['r-+'])
for i in range(3):
    ax[i].set_xlabel('epoch')
    ax[i].grid(which='major', axis='both', color='g', linestyle=':')
ax[0].set_ylabel('accuracy(%)')
ax[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# using test dataloader to test the trained model
loss, accuracy = evaluation(test_dataloader)
print('test dataloader accuracy (%)=', accuracy)
n = 1300
img, label = test_dataset[n]
def predict_image(img, model):
    print('img.shape=', img.shape)
    x = img.unsqueeze(0)
    print('img.unsqueeze(0).shape=', x.shape)
    y = model(x)
    preds = torch.argmax(y, dim=1)
    return train_dataset.classes[preds[0].item()]

print('Prediction is: {0}; label is {1}'.format(
    predict_image(img, model), train_dataset.classes[label]
))
plt.imshow(denorm(img).permute(1,2,0))
