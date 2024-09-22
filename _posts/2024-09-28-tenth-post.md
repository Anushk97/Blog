---
layout: post
title: Pytorch project - Training a handwritten digit recognition algorithm
date:   2024-09-18 12:54:15 +0800
categories: jekyll update
---


#### 🔥 Import libraries
Let's import the libraries to download the handwritten images

```
import torchvision
from torchvision.transforms import transforms
```

#### 😵 Transform data
Then we define a transform object which will be used to preprocess the image before training NN.

```
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
```

***transforms.ToTensor():***
- Converts a PIL Image or NumPy array into a PyTorch tensor.
- Scales the pixel values from a range of [0, 255] to [0.0, 1.0].

***transforms.Normalize((0.5,), (0.5,)):***
- This transformation normalizes the tensor values.
- Each channel is normalized by subtracting the mean (here 0.5) and dividing by the standard deviation (here also 0.5).

#### 💫 Download data from torchvision
Then define the train and tests datasets

```
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

- Number of train datapoints: 60k
- Number of test datapoints: 10k

#### ⚡ Define train and test loaders 

We will use 30 samples in one batch to train the NN
```
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False)
```

#### 🗯️ Creating a simple NN

Class names are defined as follows:
['0 - zero',
 '1 - one',
 '2 - two',
 '3 - three',
 '4 - four',
 '5 - five',
 '6 - six',
 '7 - seven',
 '8 - eight',
 '9 - nine']

 First, NN will have a single layer with RELU activation:

```
class MNIST_NN(nn.Module):
    def __init__(self, input_shape: int, 
                 hidden_units: int, 
                 output_shape: int):
        super().__init__()
        
        # Create a hidden layer with non-linearities
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
```

- nn.Flatten(): This layer flattens the input, converting a multi-dimensional image (28x28) into a 1D vector. 
- nn.Linear(in_features=input_shape, output_shape=hidden_units): A fully connected layer that maps the input (flattened image) to a hidden layer with hidden_units neurons.
- nn.ReLU(): The ReLU activation function, introduces non-linearity into the model. It returns the input if it’s positive, or zero otherwise.
- nn.Linear(in_features=hidden_units, output_shape=output_shape): Another fully connected layer that maps the hidden layer to the output layer.

```
model_non_linear = MNIST_NN(input_shape=784, hidden_units=100, output_shape=10)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
 lr=0.01)
```

For loss function - cross entropy loss is best for classification problems (see reference)

##### Train step 
```
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
  train_loss, train_acc = 0, 0
  model.train()

  for batch, (X, y) in enumerate(dataloader):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader.dataset)
  
  print(f"Train loss: {train_loss}, Train acc: {train_acc}")
  return train_loss, train_acc
```

1.	train_loss, train_acc = 0, 0:
- Initializes variables to accumulate the total training loss and accuracy across all batches.
2. model.train():
- Sets the model in “training mode.” This is important because certain layers like dropout and batch normalization behave differently during training and evaluation. In training mode, they perform their regular operations (e.g., dropout randomly zeroes out some connections), while in evaluation mode, they behave differently (e.g., dropout is turned off).
3. for batch, (X, y) in enumerate(dataloader)::
- Loops through each batch of data provided by the dataloader. The dataloader yields batches of input features X and corresponding labels y.
4. y_pred = model(X):
- Passes the input data X through the model to obtain the predicted output y_pred.
5.	loss = loss_fn(y_pred, y):
- Computes the loss by comparing the predicted values (y_pred) with the actual labels (y). This loss is then used to guide the model’s parameter updates.
6.	train_loss += loss:
- Adds the batch’s loss to the cumulative train_loss. This accumulates the total loss over all batches.
7. train_acc += (y_pred.argmax(dim=1) == y).type(torch.float).sum().item():
- Calculates the number of correct predictions by comparing the predicted class (the index of the highest predicted value, y_pred.argmax(dim=1)) to the actual class label y.
8.	optimizer.zero_grad():
- Resets the gradients of the model parameters. Gradients accumulate by default in PyTorch, so it’s important to zero them before calculating the new gradients during loss.backward().
9.	loss.backward():
- Performs backpropagation, calculating the gradients of the loss with respect to the model parameters.
10.	optimizer.step():
- Updates the model parameters using the gradients calculated during backpropagation, according to the chosen optimization algorithm (e.g., SGD, Adam).

11.	train_loss /= len(dataloader):
- Averages the total loss over all the batches by dividing the accumulated train_loss by the number of batches in the dataloader.

12.	train_acc /= len(dataloader.dataset):
- Averages the training accuracy over the entire dataset by dividing the accumulated train_acc by the total number of samples in the dataset (this assumes dataloader.dataset gives access to the full dataset).



##### Test step
```
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn):
  test_loss, test_acc = 0, 0
  model.eval()

  with torch.no_grad():
    for X, y in data_loader:
      test_pred = model(X)
      test_loss += loss_fn(test_pred, y).item()
      test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader.dataset)

    print(f"Test loss: {test_loss}, Test acc: {test_acc}")
    return test_loss, test_acc
```

##### Training loop
```
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_non_linear.parameters(), # Use the defined model_non_linear
 lr=0.01)

import tqdm

loss_hist = {}

loss_hist['train'] ={}
loss_hist['test'] ={}

epochs = 10
for epoch in tqdm.tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_loss = train_step(data_loader=train_loader,
        model=model_non_linear,
        loss_fn=loss_function,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )

    loss_hist['train'][epoch] = train_loss

    test_loss = test_step(data_loader=test_loader,
        model=model_non_linear,
        loss_fn=loss_function,
        accuracy_fn=accuracy_fn
    )

    loss_hist['test'][epoch] = test_loss
```

###### accuracy_fn
- torch.eq(y_true, y_pred): compares y_true and y_pred element-wise and returns a tensor of the same shape as the inputs, where each element is True (1) if the corresponding elements in y_true and y_pred are equal, and False (0) if they are not. For example, if y_true = [1, 0, 1] and y_pred = [1, 1, 1], then torch.eq(y_true, y_pred) would return [True, False, True] or [1, 0, 1] in tensor form.

- .sum(): This sums up the number of True values (i.e., correct predictions). In our example, torch.eq(y_true, y_pred).sum() would result in 2 because there are 2 correct predictions.

- .item(): converts the result from a tensor into a regular Python number (an integer in this case). So, correct is the total number of correct predictions.

###### loop
- loss_hist = {}: Initializes an empty dictionary loss_hist to store the training and testing loss for each epoch. The dictionary is structured to have two keys:
    - 'train' for storing the training loss.
	- 'test' for storing the testing loss.

- Loop through the number of epochs specified (epochs = 10)
	
- Training Step (train_loss = train_step(...)):
    - train_loader: The DataLoader that supplies batches of training data.
	-	model_non_linear: The model being trained.
	-	loss_function: The function that calculates how wrong the model’s predictions are.
	-	optimizer: The optimization algorithm that updates the model’s weights (e.g., SGD, Adam).
	-	accuracy_fn: A function to compute accuracy (not used directly in this snippet but likely part of the training step to track accuracy).
	-	train_loss: The function returns the training loss for the current epoch, which is accumulated over all training batches.

- loss_hist['train'][epoch] = train_loss:
	-	Stores the training loss for the current epoch in the loss_hist['train'] dictionary under the corresponding epoch key.

- Testing Step (test_loss = test_step(...)):
	-	test_loader: The DataLoader that supplies batches of testing data.
	-	model_non_linear: The same model used for training, but in evaluation mode (typically no gradient computation).
	-	loss_function: The same loss function used during training to compute the test loss.
	-	accuracy_fn: The accuracy function used to track performance on the test set.
	-	test_loss: The function returns the test loss for the current epoch, accumulated over all test batches.

- loss_hist['test'][epoch] = test_loss:
	-	Stores the test loss for the current epoch in the loss_hist['test'] dictionary under the corresponding epoch key.


<span style="color: red;">Training acc: 76.28%</span>

#### 💥 Creating a CNN model for better performance

```
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5) 
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```
Simple CNN
1.	Convolutional Layers (conv1 and conv2):
	-	conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1):
	-	The first convolutional layer takes input images with 1 channel (e.g., grayscale images like MNIST).
	-	It applies 32 filters (feature detectors) with a kernel size of 3x3 and adds padding of 1 pixel to maintain the original input size.
	-	conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1):
	-	The second convolutional layer takes the 32 feature maps from the first layer as input and applies 64 filters with a kernel size of 3x3 and padding of 1.

2.	Pooling Layer (pool):
	-	self.pool = nn.MaxPool2d(kernel_size=2, stride=2):
	-	The pooling layer performs max-pooling, which downsamples the feature maps by selecting the maximum value in each 2x2 region. This reduces the spatial dimensions (height and width) of the feature maps by half.
	-	The stride of 2 ensures that the pooling operation moves by 2 pixels, further reducing the size.

3.	Fully Connected Layers (fc1 and fc2):
	-	self.fc1 = nn.Linear(64 * 7 * 7, 128):
	-	After the convolution and pooling operations, the feature maps are flattened into a 1D vector.
	-	This fully connected layer takes the 64 feature maps, each of size 7x7 (reduced from 28x28 after two max-pooling operations), and outputs 128 units.
	-	self.fc2 = nn.Linear(128, 10):
	-	The second fully connected layer takes the 128 units from the previous layer and outputs 10 units. This corresponds to the number of classes in the classification task (e.g., for MNIST, the 10 digits from 0 to 9).

4.	Dropout (dropout):
	-	self.dropout = nn.Dropout(0.5):
	-	Dropout is a regularization technique used to prevent overfitting. It randomly sets 50% of the neurons in the fully connected layer (fc1) to zero during training, which forces the model to learn more robust features.

5.	Batch Normalization (batch_norm1 and batch_norm2):
	-	self.batch_norm1 = nn.BatchNorm2d(32) and self.batch_norm2 = nn.BatchNorm2d(64):
	-	Batch normalization normalizes the outputs of each convolutional layer (32 feature maps in conv1 and 64 feature maps in conv2) to have zero mean and unit variance. This helps in stabilizing and speeding up the training process.

Forward function
1.	First Convolution Block:
	-	x = self.pool(F.relu(self.batch_norm1(self.conv1(x)))):
	-	The input x is passed through conv1 to generate 32 feature maps, which are then normalized by batch_norm1.
	-	A ReLU activation function is applied to introduce non-linearity.
	-	Max pooling is applied to reduce the spatial dimensions.

2.	Second Convolution Block:
	-	x = self.pool(F.relu(self.batch_norm2(self.conv2(x)))):
	-	The output of the first block is passed through conv2 to generate 64 feature maps, normalized by batch_norm2, followed by ReLU activation and max pooling, further reducing the spatial dimensions.

3.	Flattening:
	-	x = torch.flatten(x, 1):
	-	The 2D feature maps are flattened into a 1D vector so that they can be fed into the fully connected layers.
	-	After two pooling operations, the feature maps are of size 7x7, and there are 64 of them. Flattening results in a vector of size 64 * 7 * 7 = 3136.

4.	Fully Connected Layers:
	-	x = F.relu(self.fc1(x)): The flattened input is passed through the first fully connected layer (fc1), resulting in 128 units, with ReLU activation applied.
	-	x = self.dropout(x):
	-	Dropout is applied to the 128 units to randomly set 50% of the units to zero during training, which helps prevent overfitting.
	-	x = self.fc2(x):
	-	The result is then passed through the second fully connected layer (fc2), resulting in 10 output units (one for each class in the classification task).

<span style="color: red;">Training acc: 98.98%</span>

#### Reference

| **Loss Function**              | **Task Type**                | **Use Case**                                                                                                 | **Description**                                                                                                                                  |
|---------------------------------|------------------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Mean Squared Error (MSE)**    | Regression                   | Predicting continuous values (e.g., house prices, temperature).                                              | Measures the average squared difference between predicted and actual values. Sensitive to outliers.                                               |
| **Mean Absolute Error (MAE)**   | Regression                   | Predicting continuous values where outliers are less important (e.g., median home price estimation).          | Measures the average absolute difference between predicted and actual values. Less sensitive to outliers than MSE.                                |
| **Huber Loss**                  | Regression                   | Regression tasks where you want a balance between MSE and MAE (e.g., when handling outliers).                 | Combines MSE and MAE to be more robust to outliers while still penalizing large errors.                                                           |
| **Cross-Entropy Loss**          | Classification               | Multi-class classification (e.g., image classification, digit recognition in MNIST).                         | Measures the difference between the predicted probability distribution and the actual distribution (one-hot encoded).                             |
| **Binary Cross-Entropy Loss**   | Binary Classification        | Binary classification tasks (e.g., spam detection, cancer detection).                                        | Similar to cross-entropy but specifically for binary classification.                                                                             |
| **Categorical Cross-Entropy**   | Multi-class Classification   | When you have multiple exclusive classes (e.g., one of 10 classes in MNIST).                                  | Computes the difference between predicted and actual probability distributions for multi-class classification problems.                           |
| **Sparse Categorical Cross-Entropy** | Multi-class Classification   | Multi-class classification with large class sets and sparse labels (e.g., NLP tasks with many possible words). | Similar to categorical cross-entropy but works when labels are integers instead of one-hot encoded.                                               |
| **Hinge Loss**                  | Binary Classification        | Support Vector Machines (SVMs), binary classification (e.g., separating two classes).                         | Used in SVMs, it penalizes predictions that are on the wrong side of the margin (margin-based loss).                                               |
| **Kullback-Leibler Divergence (KL-Divergence)** | Probabilistic Models       | Tasks where you want to measure the difference between two probability distributions (e.g., language modeling). | Measures how one probability distribution diverges from a second, reference probability distribution. Useful in generative models and reinforcement learning. |
| **Negative Log-Likelihood Loss**| Probabilistic Classification | Tasks where the model outputs probabilities (e.g., softmax outputs for classification).                       | Measures how likely the true label is under the predicted probability distribution. Used in probabilistic models.                                 |
| **Dice Loss**                   | Image Segmentation           | Medical image segmentation or tasks with imbalanced classes (e.g., segmenting cancerous regions).             | Measures overlap between predicted and ground truth segmentation maps. Helps handle imbalanced classes.                                           |
| **IoU Loss (Intersection over Union)** | Object Detection            | Object detection and image segmentation tasks (e.g., bounding box detection).                                | Measures the overlap between predicted and ground truth bounding boxes. Focuses on object localization.                                           |
| **Cosine Similarity Loss**      | Similarity Tasks             | Tasks where similarity between vectors is important (e.g., text similarity, recommendation systems).          | Measures the cosine of the angle between two non-zero vectors. Useful when the magnitude of vectors isn't as important as the direction.          |
| **Triplet Loss**                | Metric Learning              | Face recognition, image retrieval tasks where distinguishing between similar and dissimilar examples is key.  | Used in metric learning, where the model learns to push dissimilar examples apart and pull similar ones closer in embedding space.                |
| **CTC Loss (Connectionist Temporal Classification)** | Sequence Prediction         | Speech recognition or handwriting recognition tasks where alignment between input and target sequences is unclear. | Used in tasks where the length of the input and target sequences do not match, such as speech-to-text without predefined alignment.               |

-----

### 2. 