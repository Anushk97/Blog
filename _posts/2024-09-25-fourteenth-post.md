---
layout: post
title: Guide to Pytorch
date:   2024-09-18 12:54:15 +0800
categories: jekyll update
---

### NN blocks
1. `torch.nn.Module`: The base class for creating custom PyTorch neural network models. It is where we define our custom model architecture.

2. `F.relu(input)`: A rectified linear unit (ReLU) activation function, which is a popular activation function in deep learning models.

3. `F.sigmoid(input)`: A sigmoid activation function, which can be used for output activations in binary classification problems or hidden units in some neural network architectures.

4. `F.tanh(input)`: A hyperbolic tangent (tanh) activation function, which is commonly used as an activation function in hidden units of recurrent neural networks (RNNs), LSTM, and GRUs.

5. `torch.nn.Linear(in_features, out_features)`: Defines a new linear transformation layer with given input features (number of inputs) and output features (number of outputs). It is often the last layer in feedforward neural networks for regression tasks, or the penultimate layer for classification problems.

6. `torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`: Defines a 2-dimensional convolutional layer with given input channels (number of filters in previous layer), output channels (number of filters to learn), kernel size, and optional padding.

7. `torch.nn.MaxPool2d(kernel_size, stride=2, padding=0)`: Defines a 2-dimensional max pooling operation with specified kernel size, stride, and optional padding. It's often used after convolutional layers in convolutional neural networks (CNNs).

8. `F.dropout(input, p=0.5)`: Applies dropout during training to randomly set elements of the input tensor to zero, with a specified probability (p) for each element.

9. `torch.nn.BatchNorm2d(num_features)`: A batch normalization layer that is often used in convolutional neural networks, which can improve model performance and training stability by normalizing activations along the specified feature dimension.

10. `F.normalize(input, dim=1, p=2.0)`: Applies Lp-norm normalization (L1 or L2), element-wise, to the specified dimension of the input tensor. It can be useful for preprocessing data before feeding it into neural networks or as part of certain architectures, like WGANs.

### Tensor operation
1. Addition: `torch.add(tensor1, tensor2)`
2. Subtraction: `torch.sub(tensor1, tensor2)`
3. Multiplication: `torch.mul(tensor1, tensor2)`
4. Division: `torch.div(tensor1, tensor2)`
5. Powers: `torch.pow(tensor, exponent)`
6. Minimum: `torch.min(tensor1, tensor2, dim=0)` (element-wise) or `torch.min(input, dim=axis)` (along a specified dimension)
7. Maximum: `torch.max(tensor1, tensor2, dim=0)` (element-wise) or `torch.max(input, dim=axis)` (along a specified dimension)
8. Absolute values: `torch.abs(tensor)`
9. Negation: `torch.neg(tensor)`
10. Square root: `torch.sqrt(tensor)`
11. Summation: `torch.sum(input, dim=None, keepdim=False)`
12. Mean: `torch.mean(input, dim=None, keepdim=False)`
13. Transpose: `torch.transpose(input, dim0, dim1)` or `input.t()`
14. Repeat: `torch.repeat(input, repeats, dim=None)`
15. Permute dimensions: `torch.permute(input, dims)`
16. Element-wise operations: `torch.sigmoid(tensor)`, `torch.tanh(tensor)`, `torch.relu(tensor)`, `torch.log_softmax(tensor, dim=None)`, `torch.softmax(tensor, dim=None)`
17. Matrix multiplication: `input @ weights.t()` or `torch.matmul(input, weights.t())`
18. Inner product/Dot product: `torch.dot(input, weights)` or `input @ weights.t()`
19. Scaling a tensor: `tensor * scalar` or `tensor.mul_(scalar)`
20. Changing data type: `tensor.to(TorchDtype)` or `tensor.float()`, `tensor.double()`, etc.
21. Creating empty tensors of specific shapes and sizes: `torch.empty((3, 4), dtype=TorchDtype)` or `torch.zeros((3, 4), dtype=TorchDtype)`.


 ----

 ### LAB 

1. Creating Tensors as Matrices and Vectors:
   In PyTorch, tensors represent multidimensional arrays of data. To create matrices or vectors, you can use the `torch.tensor()` function with the specified shape and data type. For example:
   
   ```python
   import torch

   # Creating a 3x3 matrix
   A = torch.tensor(np.random.randn(3, 3), dtype=torch.float)

   # Creating a 1xN vector
   x = torch.tensor(np.random.randn(size=10), dtype=torch.float).T
   ```

2. Linear Operations:
   PyTorch offers various functions for linear operations, such as matrix multiplication (`@` operator or `torch.matmul()`) and matrix-vector multiplication (`torch.matmul(A, x)`). Transpose of a tensor can be computed using the `transpose()` method:
   ```python
   # Matrix Multiplication
   y = A @ x

   # Matrix-Vector Multiplication
   z = torch.matmul(A, x)

   # Transpose
   A_T = A.t()
   ```

3. Matrix Operations:
   PyTorch provides functions for common matrix operations like addition (`+`), subtraction (`-`), element-wise multiplication (`*`), and element-wise division (`/`). Functions like `torch.eye()` create identity matrices, while `torch.zeros()` and `torch.ones()` create matrices filled with zeros or ones, respectively:
   ```python
   # Matrix Addition
   C = A + B

   # Scalar Multiplication
   D = alpha * A
   ```

4. Transformations using Linear Algebra:
   PyTorch's `nn` module allows you to build neural networks with various transformations such as fully connected layers, convolutional layers, and linear transformations. For example, a simple feedforward neural network can be defined as follows:
   ```python
   import torch.nn as nn
   import torch.nn.functional as F

   class SimpleNN(nn.Module):
       def __init__(self, input_size, hidden_size, output_size):
           super().__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(hidden_size, output_size)

       def forward(self, x):
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return x

   # Instantiating the network and setting parameters
   model = SimpleNN(input_size=X.shape[1], hidden_size=50, output_size=1)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
   ```
    This example demonstrates the usage of `nn.Linear()` and the ReLU activation function (`nn.ReLU()`) to build a simple feedforward network. Note that in practice, you will also need to define a loss function, train your model, and make predictions with new data using this network.