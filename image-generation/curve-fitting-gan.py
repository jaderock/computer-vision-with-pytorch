import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")

n_epochs = 20000
batch_size = 64
z_dim = 2    # latent space dimensions
n_hidden = 256    # number of neurons in the only hidden layer of the GAN
n_points = 21     # number of points on each curve of training_data
n_batch = 5       # DAta loader batch number per epoch
lr = 1e-4

latent_noise = torch.randn(batch_size, z_dim, requires_grad=True, device=device)  #latent_noise.shape = 64*2

# creating training data from y=a*x^2+(a-1)
x = np.linspace(-1, 1, n_points)  #x=[-1.0, -0.9, ..., 1.0]
def training_data():
    a = np.random.uniform(1, 2, size=batch_size).reshape(-1,1)  # a.shape=64*1
    points = a*x*x + (a-1) #points.shape = 64*21
    return points
y = training_data()

# Show curves
def show_curves(x, y):
    fig, ax = plt.subplots(figsize=(4,4))
    for i in range(batch_size):
        ax.plot(x, y[i,:])
    ax.set(xlabel='x', ylabel='y', xticks=np.arange(-1.0, 1.1, 0.5))
    ax.grid(color='g', linestyle=':')
    plt.show()
show_curves(x, y)

# Generator class
class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_points)
        )
    def forward(self, z):
        fake_curves = self.net(z)    # z.shape = batch_size * z_dim = 64 *2
        return fake_curves           # fake_curves.shape = batch_size * n_points
G = generator() #.cuda()

# Discriminator class
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_points, n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, curves):
        output = self.net(curves)    # curves.shape = batch_size * n_points
        return output                # output.shape = batch_size * 1
D = discriminator() #.cuda()

# Model optimizers and BCELoss training loss
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)
loss_fn = nn.BCELoss(reduction='mean')
scheduler = lr_scheduler.OneCycleLR(
    optimizer_D,
    max_lr=lr,
    steps_per_epoch=n_batch,
    epochs=n_epochs,
    pct_start=0.6)

# Model D training function for a batch of dataloader
def train_D(curves, optimizer_D):
    # Using a batch of real images to calculate BCE loss with a target of ONEs
    batch_size = curves.shape[0]
    real_preds = D(curves)
    ones_target = torch.ones(batch_size, 1, device=device)
    real_loss = loss_fn(real_preds, ones_target)
    real_score = torch.mean(real_preds).item()
    # Generating a batch of fake images for BCE loss with a target of ZEROs
    noise = torch.randn(batch_size, z_dim, requires_grad=True, device=device)
    curves_fake = G(noise)
    fake_preds = D(curves_fake.detach()) # detach() for without training G
    zeros_target = torch.zeros(batch_size, 1, device=device)
    fake_loss = loss_fn(fake_preds, zeros_target)
    fake_score = torch.mean(fake_preds).item()

    loss_D = real_loss + fake_loss
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()
    scheduler.step()
    return loss_D.item(), real_score, fake_score

# Training function for the generator
def train_G(optimizer_G):
    noise = torch.randn(batch_size, z_dim, requires_grad=True, device=device)
    curves_fake = G(noise)
    fool_preds = D(curves_fake)
    ones_targets = torch.ones(batch_size, 1) #.cuda()
    loss_G = loss_fn(fool_preds, ones_targets)
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()
    return loss_G.item()

# Fitting function
def fit(epochs):
    #torch.cuda.empty_cache()
    df = pd.DataFrame(np.empty([epochs, 4]), index = np.arange(epochs),
                      columns=['Loss_G', 'Loss_D', 'D(x)', 'D(G(z))'])
    for i in trange(epochs):
        loss_G = 0.0
        loss_D = 0.0
        real_sc = 0.0
        fake_sc = 0.0
        for _ in range(n_batch):
            curves_real = torch.FloatTensor(training_data())
            loss_d, real_score, fake_score = train_D(curves_real, optimizer_D)
            loss_D += loss_d
            real_sc += real_score
            fake_sc += fake_score
            loss_g = train_G(optimizer_G)
            loss_G += loss_g
        df.iloc[i,0:3] = loss_G/n_batch, loss_D/n_batch, real_sc/n_batch
        df.iloc[i,3] = fake_sc/n_batch
        if i==0 or (i+1)%4000==0:
            print(
                "Epoch={:5}, Ls_G={:.3f}, Ls_D={:.3f}, D(x)={:.3f}, D(G(z))={:.3f}"
                .format(i+1, df.iloc[i,0], df.iloc[i,1], df.iloc[i,2], df.iloc[i,3])
            )
            fake_curves = G(latent_noise)
            show_curves(x, fake_curves.detach().numpy())
    return df

train_history = fit(n_epochs)

# show the training history in graph
df = train_history
fig, ax = plt.subplots(1, 2, figsize=(9,4), sharex=True)
df.plot(ax=ax[0], y=[0,1], style=['r:', 'b:'])
df.plot(ax=ax[1], y=[2,3], style=['b:', 'r:'])
for i in range(2):
    ax[i].set_xlabel('epoch')
    ax[i].grid(which='major', axis='both', color='g', linestyle=':')
ax[0].set(ylim=[0.4, 1.6], ylabel='Loss')
ax[0].axhline(y=2*np.log(2), color='k', linestyle='--') # theory D loss value
ax[0].axhline(y=np.log(2), color='k', linestyle='--')   # theory G loss value
ax[1].axhline(y=0.5, color='k', linestyle='--')    # theory D(x), D(G(z)) values
ax[1].set_ylim([0, 1.0])
plt.show()