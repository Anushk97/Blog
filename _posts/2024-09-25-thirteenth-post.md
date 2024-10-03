---
layout: post
title: Two stage training on MNIST
date:   2024-09-18 12:54:15 +0800
categories: jekyll update
---

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the TwoStageNetwork
class TwoStageNetwork(nn.Module):
    def __init__(self):
        super(TwoStageNetwork, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(784, 256),  # Flattened input size 28x28 = 784
            nn.ReLU(),
            nn.Linear(256, 128)  # Reconstruct back to 784 for autoencoder
        )
        self.stage2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 output classes
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x

# Define the Stage 1 Training (Autoencoder)
def train_stage1(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in dataloader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            optimizer.zero_grad()
            features = model.stage1(inputs)
            loss = criterion(features, features.detach())  # Self-supervised loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

# Define the Stage 2 Training (Classification)
def train_stage2(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            optimizer.zero_grad()
            features = model.stage1(inputs).detach()  # Use frozen features from stage1
            outputs = model.stage2(features)  # Classify using stage2
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])

# Download the MNIST training dataset
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)

# Instantiate the model, optimizer, and loss function
model = TwoStageNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_stage1 = nn.MSELoss()  # Autoencoder loss for Stage 1
criterion_stage2 = nn.CrossEntropyLoss()  # Classification loss for Stage 2

# Train Stage 1 (Feature Extraction)
print("Training Stage 1 (Feature Extraction)")
train_stage1(model, mnist_dataloader, optimizer, criterion_stage1, epochs=5)

# Train Stage 2 (Classification)
print("Training Stage 2 (Classification)")
train_stage2(model, mnist_dataloader, optimizer, criterion_stage2, epochs=5)

```

This is a neural network training procedure using a **two-stage architecture**. We can first train a network to extract features from data using a **self-supervised learning approach** (an autoencoder), and then use those features to perform classification on the MNIST dataset, a popular benchmark dataset for handwritten digits.

---

### 1. **What is the Two-Stage Network?**

The concept behind the two-stage network in this example is straightforward:
- **Stage 1:** The network is first trained to learn useful features from the input data. This is achieved via an **autoencoder** model, where the goal is to reconstruct the input data (in this case, 28x28 pixel images of handwritten digits) after passing it through a series of transformations.
- **Stage 2:** Once the network has learned to extract features, these features are used as inputs to a classifier that attempts to map them to one of the 10 digit classes (0-9). This is a standard **supervised learning** task.

By splitting the task into two stages, the network first focuses on learning a good internal representation of the data before trying to map that representation to specific labels. This approach can help the model generalize better, especially in situations where the labels are sparse or noisy.

---

### 2. **Stage 1: Feature Extraction with Autoencoder**

#### **Autoencoder Basics:**
An autoencoder is a type of neural network designed to encode input data into a compressed (lower-dimensional) representation, and then decode it back to the original data. The network is trained to minimize the difference between the input and the reconstructed output.

In this context, the **autoencoder** is used to compress a 784-dimensional input (flattened 28x28 image) down to a 128-dimensional vector. This 128-dimensional vector is the learned feature representation of the image.

#### **Why Use an Autoencoder for Feature Extraction?**
Feature extraction is critical in machine learning because it enables the network to transform raw data into more abstract representations that capture the underlying structure or patterns. Autoencoders force the network to encode the important information in a compressed form, leaving out the noise. Once trained, this compressed form can be used for tasks such as classification.

#### **Code Breakdown for Stage 1:**

```python
def train_stage1(model, dataloader, optimizer, criterion, epochs):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in dataloader:  # Ignore the labels during self-supervised training
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images (batch_size, 784)
            optimizer.zero_grad()
            features = model.stage1(inputs)  # Forward pass through stage1
            loss = criterion(features, features.detach())  # Self-supervised loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
```

- **Input Flattening:** 
  Each MNIST image is 28x28 pixels, which is flattened into a vector of size 784 (`inputs.view(inputs.size(0), -1)`). This flattening is required because the `nn.Linear` layers expect a 1D vector as input.

- **Feature Extraction via Stage 1:**
  `model.stage1(inputs)` passes the input through the first stage of the network. This consists of two linear layers:
  - A transformation from 784 dimensions (the input) to 256 dimensions, followed by a ReLU activation.
  - A transformation from 256 dimensions to 128 dimensions (the learned feature vector).

- **Loss Calculation:**
  The loss here is computed using **Mean Squared Error (MSE)** between the feature vector and its `detach()`ed version. While this setup appears as a self-supervised task, it's a placeholder for the autoencoder. For an actual autoencoder, we would typically compare the reconstructed output to the original input.

---

### 3. **Stage 2: Classification**

#### **From Feature Extraction to Classification:**
Once the network has been trained to learn useful features, those features are used as input for the second stage: classification. The output of `stage1` is a 128-dimensional vector, which is then fed into `stage2`. The job of `stage2` is to map this feature vector to one of 10 classes (digits 0 through 9).

#### **Supervised Learning Basics:**
In the classification stage, we train the model to minimize the **cross-entropy loss** between the predicted class probabilities and the true labels. Cross-entropy loss is commonly used for multi-class classification problems, and it works by measuring the difference between the predicted probability distribution and the actual distribution (represented as one-hot encoded labels).

#### **Code Breakdown for Stage 2:**

```python
def train_stage2(model, dataloader, optimizer, criterion, epochs):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            optimizer.zero_grad()
            features = model.stage1(inputs).detach()  # Use frozen features from stage1
            outputs = model.stage2(features)  # Forward pass through stage2
            loss = criterion(outputs, labels)  # Supervised loss (CrossEntropy)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
```

- **Frozen Features:**
  `model.stage1(inputs).detach()` ensures that the features extracted from `stage1` are used without updating the weights of `stage1`. This is important because we want to fine-tune only `stage2` in this stage.

- **Classification via Stage 2:**
  `model.stage2(features)` classifies the 128-dimensional feature vector into one of 10 classes, mapping the features down to 64 dimensions and then finally to 10 output classes.

- **CrossEntropy Loss:**
  `nn.CrossEntropyLoss()` computes the loss by comparing the predicted class probabilities with the actual labels. It measures how well the model is predicting the correct class out of the 10 possible classes.

---

### 4. **MNIST Dataset: The Data Source**

#### **Why MNIST?**
The **MNIST dataset** is widely used for benchmarking machine learning algorithms. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is grayscale and has a resolution of 28x28 pixels.

The dataset is small enough to be run efficiently on most hardware and large enough to demonstrate the principles of machine learning.

#### **Preprocessing with PyTorch:**

```python
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])
```

- **`ToTensor()`** converts the images from the dataset into PyTorch tensors, which are the format expected by the model.
- **`Normalize((0.5,), (0.5,))`** normalizes the pixel values to the range `[-1, 1]` (from the original `[0, 1]`), which can help the network learn more efficiently.

---

### 5. **Training Process**

Finally, the model is trained in two stages:

#### **Training Stage 1:**

```python
train_stage1(model, mnist_dataloader, optimizer, criterion_stage1, epochs=5)
```
This trains the model to extract features from the MNIST images. The features are compressed into a 128-dimensional vector.

#### **Training Stage 2:**

```python
train_stage2(model, mnist_dataloader, optimizer, criterion_stage2, epochs=5)
```
This trains the classifier to map the 128-dimensional feature vector to one of the 10 digit classes.
