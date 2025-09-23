import torch.nn as nn
def basic(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class Id_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.basic_block = basic(in_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.basic_block(x))
    
class Rs_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.basic_block = basic(in_channels, out_channels, stride=2)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.shortcut(x) + self.basic_block(x))
    
class ResNet(nn.Module):    # ResNet18  layer=[2,2,2,2]
    def __init__(self, image_channels=3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # layer0 out_shape=64*48^2
            nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # layer1 out_shape=64*24^2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Id_Block(64, 64, 1),
            Id_Block(64, 64, 1),
            # layer2 out_shape=128*12^2
            Rs_Block(64, 128, 2),
            Id_Block(128, 128, 1),
            # layer3 out_shape=256*6^2
            Rs_Block(128, 256, 2),
            Id_Block(256, 256, 1),
            # layer4 out_shape=512*3^2
            Rs_Block(256, 512, 2),
            Id_Block(512, 512, 1),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.net(x)
    
model = ResNet() #.cuda()