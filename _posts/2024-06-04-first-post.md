---
layout: post
title:  "Image classification on CIFAR 10 Dataset"
date:   2024-06-04 12:54:15 +0800
categories: jekyll update
---

This project implements different Image classification algorithms on the CIFAR 10 dataset. CIFAR 10 is a dataset of images containing 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The following are the algorithms used to classify these images. I have provided a summary of test accuracy and training time for each as well.

- VGG 16
- ResNet 18
- GoogleNet
(Note: Training was done on Google Colab GPU (free version).)

### VGG 16 Architecture
Visual Geometry Group or VGG16 has three fully connected layers and 13 convolutional layers with trainable parameters and 3 fully connected layers. It uses small 3x3 filters for consistent stacking and a deeper representation of learning features.

**The input image is of dimension 224x224x3.**

- Layer 1 and 2: They have the dimension 224x224x64, meaning they have 64 channels with a 3x3 filter size and the same padding as well as a stride of 2. Use the formulas below to calculate different components:
    
        Output for each layer: [(width — filter size + 2*padding)/Stride]+1

        Max pooling layer: (size — max pooling size)/stride + 1

        padding: filter size-1 / 2

        The output of this layer will be: [(224–3+6)/2] + 1 = 112x112x64

- Max pooling 1, layers 2 and 3: The output from layer 2 will be the input for the map pooling layer. The max pooling layer has a stride of 2, the two convolution layers have 128 channels and a filter size of 3x3.

        Output of the max pooling layer will be: (112–3)/2 + 1 = 56x56x128

- Max pooling 2, layers 4 and 5: This stack of layers has a max-pooling layer of stride (2, 2) which is the same as the previous layer. Then there are 2 convolution layers of filter size (3, 3) and 256 filters.

        The output of the max pooling layer will be: (56–3)/2 + 1 = 28 x 28x 256

- Rest of the layers: There are 2 sets of 3 convolution layers and a max pool layer. Each has 512 channels and filters of (3, 3) size with the same padding. The image is then passed to the stack of two convolution layers.

- After the stack of convolution and max-pooling layer, (7, 7, 512) feature map is generated, the output is flattened to make it a (1, 25088) feature vector (7x7x512 = 25088).

**After this, there are 3 fully connected layers**

1. The first layer takes input from the last feature vector of (1, 25088) and outputs a (1, 4096) vector.

2. The second layer also outputs a vector of size (1, 4096) but the third layer output 10 channels for 10 classes in the CIFAR dataset.

3. The third fully connected layer is different from the first two. Instead of producing a vector of size (1, 4096), it’s designed to output a vector of size (1, 10). In the context of the CIFAR dataset, this corresponds to the number of classes (10 classes).

Finally, the output from the third fully connected layer is passed through the softmax activation function which converts raw values into probability distribution over the classes.

```
import torch
import torch.nn as nn
from torch.autograd import Variable
# defining different configuration of VGG architecture. 
# Each configuration has different number of filters and convolutions
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Initializes the architecture by creating the convolutional layers
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

# Defines the forward pass of the network.
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
# Method creates the sequence of convolutional, batch normalization, and ReLU activation layers based on the configuration.
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
```
**Training time**: 1 hr 45 mins

**Accuracy**:| 99.98% training accuracy | 93.68% test accuracy.

### ResNet 18 Architecture
The architecture of ResNet is similar to the VGG architecture but also has Residual connections which help in solving the vanishing gradient problem. Residual/skip connections allow information to bypass certain layers, enabling the network to learn residual mapping and faster convergence by propagating gradients more effectively throughout the network.

Resnets are made by stacking these residual blocks together. This also allows the network to retain features in the long-term memory and make longer connections between different features in the network.

A residual block typically consists of the following components:

1. Input Feature Map
2. Two or more convolutional layers (with activation functions)
3. Shortcut Connection (residual connection) that directly adds the original input to the output of the convolutional layers

The convolutional layers mostly have 3×3 filters and follow two simple design rules:

1. For the same output feature map size, the layers have the same number of filters
2. If the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.

Downsampling is done directly by convolutional layers that have a stride of 2. The network ends with a global average pooling layer and a 10-way fully-connected layer with softmax. The total number of weighted layers is 18.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

#Each block consists of convolutional layers, batch normalization layers, and residual connections.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#The expansion parameter specifies the expansion factor for the number of channels in the bottleneck layer.
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

#The method takes the number of blocks, planes (number of filters), and stride (for downsampling).
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

#Input x is passed through convolutional layers, followed by each of the four layers with residual blocks.
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
```

**Training time**: 2 hrs 36 mins

**Accuracy**:| 99.99% training accuracy | 95.3% test accuracy.

**Hyperparameters**:| epochs = 200 | batch size = 128

### Google Net Architecture

22-layer deep network, and 5 max pooling layers.

The input layer is taken with an image size of 224x224x3. 

1. The first convolution layer has a filter size of 7x7, a stride of 2 and padding of 3 (filter size-1/2) 
        
        output = [(width — filter size + 2*padding)/Stride]+1 = (224–7 + 6 / 2) + 1 = 112x112x64.

2. The max pooling layer has filter size of 3x3 and input of 112x112x64  
        
        output size = (size — max pooling size)/stride + 1 = (112–3/2) + 1 = 56x56x64

**Inception layer**: Perform multiple convolutions of different kernel sizes (such as 1x1, 3x3, and 5x5) on the same input layer and then concatenate the resulting feature maps. This allows the network to capture features at various levels of abstraction and spatial resolution at the same time.

- 1x1 Convolution: Captures linear combinations of features.
- 3x3 Convolution: Captures more localized patterns.
- 5x5 Convolution: Captures larger patterns or structures.

**Max Pooling**: A max pooling operation is also included to capture the most important features in different regions.

**Dropout layer**: Used as a regularization technique to prevent overfitting. This layer drops out a fraction of units in each layer during the forward and backward passes. The fraction of the unit is a hyperparameter which is 40% in the figure above. The dropout units do not contribute in the forward pass and their weights are not updated during backpropagation. Dropout is done during the training only.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```
**Training time**: 1 hr 20 mins

**Accuracy**:| 90.92% training | 77.85% test

**Hyperparameters:** | epochs = 50 | batch size = 128

### Data Preparation + training and testing
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from models import *
#from utils import progress_bar
from tqdm import tqdm

# Set the desired values directly
lr = 0.1
resume = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# Data download and transform
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
# net = VGG('VGG19')
# net = ResNet18()
net = GoogLeNet()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr= lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training function
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

# Testing function 
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(testloader), total=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

# Start training and testing
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
```