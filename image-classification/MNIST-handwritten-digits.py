# MNIST handwritten digits classification
# training on CPU

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

n_epochs = 25
batch_size = 32
lr = 1e-2  # initial learning rate

img_size = 28
n_class = 10
n_hidden = 100
data_path = './OCR/data'

# MNIST train_dataset and test_dataset
train_dataset = MNIST(data_path, train=True, download=True, transform=T.Compose([T.ToTensor()]))
n_samples = len(train_dataset)  # n_samples = 60000
test_dataset = MNIST(data_path, train=False, download=True, transform=T.Compose([T.ToTensor()]))
n_tests = len(test_dataset)     # n_tests = 10000

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size*img_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_class)
        )  # output.shape = bach_size * n_class
    def forward(self, images):
        input = images.view(images.shape[0], -1)    # input.shape = batch_size * 784
        output = self.net(input)    # output.shape = batch_size * 10
        return output

model = Model()
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# training batch
def training():
    total_loss = 0.0
    n_correct = 0.0
    model.train()
    for images, labels in train_dataloader:
        targets = labels
        y_hat = model(images)
        predictions = torch.argmax(y_hat, dim=1)
        n_correct += torch.sum(predictions==targets).item()
        loss = loss_fn(y_hat, targets)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss/n_samples, 100*n_correct/n_samples

def evaluation():
    total_loss = 0.0
    n_correct = 0.0
    with torch.no_grad():
        model.eval()
        for images, labels in test_dataloader:
            targets = labels
            y_hat = model(images)
            predictions = torch.argmax(y_hat, dim=1)
            n_correct += torch.sum(predictions==targets).item()
            loss = loss_fn(y_hat, targets)
            total_loss += loss.item()
        return total_loss/n_tests, 100*n_correct/n_tests


# training function
def fitting(epochs):
    df = pd.DataFrame(np.empty([epochs, 4]),
                      index = np.arange(epochs),
                      columns=['loss_train', 'acc_train', 'loss_test', 'acc_test'])
    progress_bar = tqdm(range(epochs))
    for i in progress_bar:
        df.iloc[i, 0], df.iloc[i, 1] = training()
        df.iloc[i, 2], df.iloc[i, 3] = evaluation()
        progress_bar.set_description("train_loss=%.5f" % df.iloc[i, 0])
        progress_bar.set_postfix({'train_acc': df.iloc[i,1], 'test_acc': df.iloc[i,3]})
    return df

train_history = fitting(n_epochs)

# display training results
df = train_history
fig, ax = plt.subplots(1, 2, figsize=(10,4), sharex=True)
df.plot(ax=ax[0], y=[1,3], style=['r-o', 'b-+'])
df.plot(ax=ax[1], y=[0,2], style=['r-o', 'b-+'])
for i in range(2):
    ax[i].grid(which='major', axis='both', color='g', linestyle=':')
ax[0].set(xlabel='epoch', ylabel='accuracy(%)')
ax[1].set(xlabel='epoch', ylabel='loss')
plt.show()

# confusion matrix for image classification
def matrics(dataloader):
    y_true = []
    y_pred = []
    for images, labels in dataloader:
        y_true += labels
        model_output = model(images)
        y_hat = torch.argmax(model_output, dim=1)
        y_pred += y_hat.cpu()
    return confusion_matrix(y_true, y_pred)

cf = matrics(test_dataloader)
print('confusion_matrix=\n', cf)

# Metrics of confusion matrix for image classification
def metrics(cf, target_names):
    n = len(target_names)
    df = pd.DataFrame(
        np.zeros([n+2, 5]),
        index= target_names + ['macro_avg', 'weighted_avg'],
        columns=['precision', 'recall', 'f1_score', 'IoU', 'support']
    )
    for i in range(n):
        TP = cf[i, i]
        support = cf[i,:].sum()
        recall = TP/support
        preds = cf[:n, i].sum()
        precision = TP/preds
        f1_score = 2*TP/(preds + support)
        IoU = TP/(preds + support - TP)
        df.iloc[i, 0:5] = [np.round(precision, 3),
                           np.round(recall, 3),
                           np.round(f1_score, 3),
                           np.round(IoU, 3),
                           np.round(support, 0)]
        for j in range(4):    # for macro average
            df.iloc[n, j] = np.round(df.iloc[:n, j].mean(), 3)
        df.iloc[n, 4] = np.round(cf.sum(), 0)
        for j in range(4):
            df.iloc[n+1, j] = np.round(np.sum(
                [df.iloc[i,j]*df.iloc[i,4]/df.iloc[n,4] for i in range(n)]
            ), 3)
        df.iloc[n+1, 4] = np.round(np.sum([cf[i,i] for i in range(10)])/cf.sum(), 3)
        return df

df = metrics(cf, class_names)
print(df)

