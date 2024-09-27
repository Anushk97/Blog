---
layout: post
title: Deep learning models implemented in pytorch and TF
date:   2024-09-18 12:54:15 +0800
categories: jekyll update
---

## 1. logistic regression

### Pytorch

<span style="color: red;">steps: prepare and load data >> split into train and test set >> normalize >> define NN >> train and evaluate </span>

### 🏛️ Preparing dataset

#### Data source and download
```python
ds = np.lib.DataSource()
fp = ds.open('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

x = np.genfromtxt(BytesIO(fp.read().encode()), delimiter=',', usecols=range(2), max_rows=100)

y = np.zeroes(100)
y[50:] = 1

np.random.seed(1)
idx = np.arange(y.shape[0])
np.random.shuffle(idx)
```
- The np.lib.DataSource() is used to open and read the Iris dataset from the specified URL.
- fp = ds.open(...) fetches the file and stores it in fp as an open file-like object.

- The dataset is read into a NumPy array x using np.genfromtxt().
- BytesIO(fp.read().encode()): Converts the content of the file into bytes, so it can be processed as a file-like object in memory.
- delimiter=',': Specifies that the file is comma-separated.
- usecols=range(2): Only the first two columns of the dataset are used (sepal length and sepal width).
- max_rows=100: Only the first 100 rows are loaded (this includes data for the first two species of the Iris dataset).

- A label array y of 100 zeros is created, where the first 50 rows (corresponding to the first Iris species) are labeled as 0 and the next 50 rows (for the second species) are labeled as 1.
- This creates a binary classification problem between two species.

- np.random.seed(1) ensures that the shuffling is reproducible.
- idx = np.arange(y.shape[0]) creates an array of indices from 0 to 99.
- np.random.shuffle(idx) shuffles these indices, which will be used to randomly split the data into training and test sets.

#### Split into train and test 
```python
X_test, y_test = x[idx[:25]], y[idx[:25]]
X_train, y_train = x[idx[25:]], y[idx[25:]]
```
- The shuffled indices are used to split the data into training and test sets.
- The first 25 shuffled samples are assigned to the test set (X_test and y_test).
- The remaining 75 samples are used for training (X_train and y_train).

#### Normalize data
```python
mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train, X_test = (X_train - mu) / std, (X_test - mu) / std
```
- mu: The mean of the X_train data is computed along each feature (sepal length and sepal width).
- std: The standard deviation of X_train data is also computed.
- X_train, X_test = (X_train - mu) / std, (X_test - mu) / std: Both the training and test data are normalized using the mean and standard deviation of the training data. This step is important to standardize the data, ensuring that the features have a mean of 0 and a standard deviation of 1, which improves model performance.

### 🦒 Low level implementation using manual gradients

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def custom_where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)


class LogisticRegression1():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, 
                                   dtype=torch.float32, device=device)
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)

    def forward(self, x):
        linear = torch.add(torch.mm(x, self.weights), self.bias)
        probas = self._sigmoid(linear)
        return probas
        
    def backward(self, probas, y):  
        errors = y - probas.view(-1)
        return errors
            
    def predict_labels(self, x):
        probas = self.forward(x)
        labels = custom_where(probas >= .5, 1, 0)
        return labels    
            
    def evaluate(self, x, y):
        labels = self.predict_labels(x).float()
        accuracy = torch.sum(labels.view(-1) == y) / y.size()[0]
        return accuracy
    
    def _sigmoid(self, z):
        return 1. / (1. + torch.exp(-z))
    
    def _logit_cost(self, y, proba):
        tmp1 = torch.mm(-y.view(1, -1), torch.log(proba))
        tmp2 = torch.mm((1 - y).view(1, -1), torch.log(1 - proba))
        return tmp1 - tmp2
    
    def train(self, x, y, num_epochs, learning_rate=0.01):
        for e in range(num_epochs):
            
            #### Compute outputs ####
            probas = self.forward(x)
            
            #### Compute gradients ####
            errors = self.backward(probas, y)
            neg_grad = torch.mm(x.transpose(0, 1), errors.view(-1, 1))
            
            #### Update weights ####
            self.weights += learning_rate * neg_grad
            self.bias += learning_rate * torch.sum(errors)
            
            #### Logging ####
            print('Epoch: %03d' % (e+1), end="")
            print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
            print(' | Cost: %.3f' % self._logit_cost(y, self.forward(x)))
```
1. The first line sets the device (GPU if available or CPU) to be used for computation.
2. `custom_where` is a custom function that performs element-wise replacement of elements based on a given condition. This will be used later in the code to extract labels from predictions.
3. `LogisticRegression1` class definition: The Logistic Regression model with one hidden layer (weights) and a bias term is defined here. Initialization of weights, bias, forward propagation, backward propagation for calculating errors, predicting labels using custom_where function, and evaluating accuracy are all implemented within this class.
4. `__init__` method initializes the weights and bias with zeros.
5. `forward` method computes the output of the logistic regression model given an input x.
6. `backward` method calculates the errors based on true labels y.
7. `predict_labels` method extracts the labels using custom_where function.
8. `evaluate` method calculates the accuracy of the model on the given data (x and y).
9. `_sigmoid` method is a simple sigmoid activation function that will be used later in forward propagation.
10. `train` method defines the Logistic Regression training process which includes:
    - Computing outputs using forward propagation.
    - Calculating gradients based on errors and input transpose.
    - Updating weights and bias using learning rate and gradients.
    - Printing progress during each epoch (train accuracy, cost).

#### Evaluation
```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

logr = LogisticRegression1(num_features=2)
logr.train(X_train_tensor, y_train_tensor, num_epochs=10, learning_rate=0.1)

print('\nModel parameters:')
print('  Weights: %s' % logr.weights)
print('  Bias: %s' % logr.bias)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = logr.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc*100))
```
1. First, the device (GPU if available or CPU) to be used for computation is set.
2. `X_train_tensor` and `y_train_tensor` are created by converting numpy arrays `X_train` and `y_train` into PyTorch tensors and moving them to the specified device.
3. A `LogisticRegression1` instance called `logr` is initialized with 2 features.
4. The model is trained using the `train` method, which takes `X_train_tensor`, `y_train_tensor`, number of epochs (iterations), and learning rate as arguments.
5. After training, the model parameters (weights and bias) are printed to the console.
6. `X_test_tensor` and `y_test_tensor` are created by converting numpy arrays `X_test` and `y_test` into PyTorch tensors and moving them to the specified device.
7. The test set accuracy is calculated by calling the `evaluate` method on the trained model using `X_test_tensor` and `y_test_tensor`.

***Test set accuracy: 100.00%***

### 🦓 Low level implementation using Autograd
```python
def custom_where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)


class LogisticRegression2():
    def __init__(self, num_features):
        self.num_features = num_features
        
        self.weights = torch.zeros(num_features, 1, 
                                   dtype=torch.float32,
                                   device=device,
                                   requires_grad=True) # req. for autograd!
        self.bias = torch.zeros(1, 
                                dtype=torch.float32,
                                device=device,
                                requires_grad=True) # req. for autograd!

    def forward(self, x):
        linear = torch.add(torch.mm(x, self.weights), self.bias)
        probas = self._sigmoid(linear)
        return probas
                    
    def predict_labels(self, x):
        probas = self.forward(x)
        labels = custom_where((probas >= .5).float(), 1, 0)
        return labels    
            
    def evaluate(self, x, y):
        labels = self.predict_labels(x)
        accuracy = (torch.sum(labels.view(-1) == y.view(-1))).float() / y.size()[0]
        return accuracy
    
    def _sigmoid(self, z):
        return 1. / (1. + torch.exp(-z))
    
    def _logit_cost(self, y, proba):
        tmp1 = torch.mm(-y.view(1, -1), torch.log(proba))
        tmp2 = torch.mm((1 - y).view(1, -1), torch.log(1 - proba))
        return tmp1 - tmp2
    
    def train(self, x, y, num_epochs, learning_rate=0.01):
        
        for e in range(num_epochs):
            
            #### Compute outputs ####
            proba = self.forward(x)
            cost = self._logit_cost(y, proba)
            
            #### Compute gradients ####
            cost.backward()
            
            #### Update weights ####
            
            tmp = self.weights.detach()
            tmp -= learning_rate * self.weights.grad
            
            tmp = self.bias.detach()
            tmp -= learning_rate * self.bias.grad
            
            #### Reset gradients to zero for next iteration ####
            self.weights.grad.zero_()
            self.bias.grad.zero_()
    
            #### Logging ####
            print('Epoch: %03d' % (e+1), end="")
            print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
            print(' | Cost: %.3f' % self._logit_cost(y, self.forward(x)))
            
```
Each method is almost similar except the training loop where the gradients are computed using pytorch's autograd

- Forward Pass: The predicted probabilities proba are computed for the input x using the current weights and bias.
- Cost Calculation: The log-loss cost is computed by comparing the predicted probabilities proba with the true labels y.
- Backpropagation: cost.backward(): Computes the gradients of the weights and bias with respect to the cost function using PyTorch’s autograd system.
- Weight Update: Manual Update:
    - tmp = self.weights.detach(): Detaches the weights from the computation graph (to prevent further tracking of gradients).
	-	Gradient Descent: The weights and bias are updated by subtracting the gradient multiplied by the learning rate.
	-	Reset Gradients: After updating, the gradients are set to zero using zero_() for the next iteration.


### 🦄 High level implementation using nn.Module

```python
class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features,1)
```
- super(LogisticRegression3, self).__init__(): This calls the constructor of the parent class torch.nn.Module to initialize the model.
- self.linear = torch.nn.Linear(num_features, 1):
    - Defines a fully connected linear layer (torch.nn.Linear), which connects the input features to the output. The linear layer is essentially a matrix multiplication followed by a bias addition:
	-	num_features: The number of input features (e.g., for a 2D dataset, num_features=2).
	-	1: The output is a single value (for binary classification in logistic regression).
	    -	This layer automatically creates two parameters:
	    -	Weights (self.linear.weight): Initialized randomly but later set to zero.
	    -	Bias (self.linear.bias): Initialized randomly but later set to zero.

#### Model weight initialization
```python
self.linear.weight.detach().zero_()
self.linear.bias.detach().zero_()
```
- self.linear.weight.detach().zero_(): Sets the weights to zero in-place (_ indicates in-place operation). The .detach() ensures that this operation is done outside of the computation graph to prevent tracking of gradients.
- self.linear.bias.detach().zero_(): Similarly, sets the bias to zero.
- This mimics the manual logistic regression model where you started with weights initialized to zero.

#### Forward Prop
```python
def forward(self, x):
    logits = self.linear(x)
    probas = torch.sigmoid(logits)
    return probas
```
- logits = self.linear(x): Applies the linear layer to the input x. The linear layer computes the weighted sum of inputs (i.e., the dot product of the input features and the weights, plus the bias).
- probas = torch.sigmoid(logits): Applies the sigmoid function to the logits (the linear combination of inputs) to produce probabilities between 0 and 1, which are necessary for logistic regression.

#### Optimize and train
```python
model = LogisticRegression(num_features=2).to(device)
cost_fn = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```
- cost_fn = torch.nn.BCELoss(reduction='sum'):
	-	Defines the Binary Cross-Entropy Loss (BCELoss) function, which is commonly used for binary classification tasks.
	-	reduction='sum': The sum of the individual losses is used rather than the average. This mirrors the manual implementation, where the cost was not normalized by batch size.
- optimizer = torch.optim.SGD(model.parameters(), lr=0.1):
	-	Sets up the Stochastic Gradient Descent (SGD) optimizer, which will update the model parameters (weights and bias) using the gradients calculated during backpropagation.
	-	lr=0.1: The learning rate for the optimization process. It controls the step size of the weight updates.

#### Accuracy computation and training
```python
def comp_accuracy(label_var, pred_probas):
    pred_labels = custom_where((pred_probas > 0.5).float(), 1, 0).view(-1)
    acc = torch.sum(pred_labels == label_var.view(-1)).float() / label_var.size(0)
    return acc

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)

for epoch in range(num_epochs):
    
    #### Compute outputs ####
    out = model(X_train_tensor)
    
    #### Compute gradients ####
    cost = cost_fn(out, y_train_tensor)
    optimizer.zero_grad()
    cost.backward()
    
    #### Update weights ####  
    optimizer.step()
    
    #### Logging ####      
    pred_probas = model(X_train_tensor)
    acc = comp_accuracy(y_train_tensor, pred_probas)
    print('Epoch: %03d' % (epoch + 1), end="")
    print(' | Train ACC: %.3f' % acc, end="")
    print(' | Cost: %.3f' % cost_fn(pred_probas, y_train_tensor))
```

Epoch: 001 | Train ACC: 0.987 | Cost: 5.581
Epoch: 002 | Train ACC: 0.987 | Cost: 4.882
Epoch: 003 | Train ACC: 1.000 | Cost: 4.381
Epoch: 004 | Train ACC: 1.000 | Cost: 3.998
Epoch: 005 | Train ACC: 1.000 | Cost: 3.693
Epoch: 006 | Train ACC: 1.000 | Cost: 3.443
Epoch: 007 | Train ACC: 1.000 | Cost: 3.232
Epoch: 008 | Train ACC: 1.000 | Cost: 3.052
Epoch: 009 | Train ACC: 1.000 | Cost: 2.896
Epoch: 010 | Train ACC: 1.000 | Cost: 2.758

Model parameters:
  Weights: Parameter containing:
tensor([[ 4.2267, -2.9613]], device='cuda:0', requires_grad=True)
  Bias: Parameter containing:
tensor([0.0994], device='cuda:0', requires_grad=True)

### Tensorflow

```python
def iterate_minibatches(arrays, batch_size, shuffle=False, seed=None):
    rgen = np.random.RandomState(seed)
    indices = np.arange(arrays[0].shape[0])

    if shuffle:
        rgen.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
        index_slice = indices[start_idx:start_idx + batch_size]

        yield (ary[index_slice] for ary in arrays)
```
- rgen = np.random.RandomState(seed): Initializes a random number generator using the provided seed. This ensures reproducibility if shuffle=True.
- indices = np.arange(arrays[0].shape[0]): Creates an array of indices ranging from 0 to the number of samples in the dataset (arrays[0].shape[0]). This array will be used to slice the input arrays into mini-batches.
- If shuffle=True, the indices are shuffled. This allows the data to be processed in a random order for each epoch, which is helpful in training to reduce bias and improve generalization.
- Loop: The loop iterates through the data using a step size equal to batch_size. This ensures that mini-batches are created starting from start_idx and going up to the total number of samples.
	-	start_idx: The starting index for each mini-batch.
	-	indices[start_idx:start_idx + batch_size]: Slices the indices to form a mini-batch.
- yield (ary[index_slice] for ary in arrays):
	-	For each mini-batch, the function slices all arrays (e.g., features and labels) using the selected indices (index_slice).
	-	The function yields the mini-batches for further processing in the training loop. This is a generator, so it doesn’t store all mini-batches in memory at once but generates them on the fly when needed.

```python
import tensorflow as tf


##########################
### SETTINGS
##########################

n_features = x.shape[1]
n_samples = x.shape[0]
learning_rate = 0.05
training_epochs = 15
batch_size = 10


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default() as g:

   # Input data
    tf_x = tf.placeholder(dtype=tf.float32,
                          shape=[None, n_features], name='inputs')
    tf_y = tf.placeholder(dtype=tf.float32,
                          shape=[None], name='targets')
    
    # Model parameters
    params = {
        'weights': tf.Variable(tf.zeros(shape=[n_features, 1],
                                               dtype=tf.float32), name='weights'),
        'bias': tf.Variable([[0.]], dtype=tf.float32, name='bias')}

    # Logistic Regression
    linear = tf.matmul(tf_x, params['weights']) + params['bias']
    pred_proba = tf.sigmoid(linear, name='predict_probas')

    # Loss and optimizer
    r = tf.reshape(pred_proba, [-1])
    cost = tf.reduce_mean(tf.reduce_sum((-tf_y * tf.log(r)) - 
                                        ((1. - tf_y) * tf.log(1. - r))), name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')
                                                  
    # Class prediction
    pred_labels = tf.round(tf.reshape(pred_proba, [-1]), name='predict_labels')
    correct_prediction = tf.equal(tf_y, pred_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


##########################
### TRAINING & EVALUATION
##########################
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    avg_cost = np.nan
    count = 1
    
    for epoch in range(training_epochs):

        train_acc = sess.run('accuracy:0', feed_dict={tf_x: x_train,
                                                      tf_y: y_train})
        valid_acc = sess.run('accuracy:0', feed_dict={tf_x: x_test,
                                                      tf_y: y_test}) 

        print("Epoch: %03d | AvgCost: %.3f" % (epoch, avg_cost / count), end="")
        print(" | Train/Valid ACC: %.2f/%.2f" % (train_acc, valid_acc))
        
        avg_cost = 0.
        for x_batch, y_batch in iterate_minibatches(arrays=[x_train, y_train],
                                                    batch_size=batch_size, 
                                                    shuffle=True, seed=123):
            
            feed_dict = {'inputs:0': x_batch,
                         'targets:0': y_batch}
            _, c = sess.run(['train', 'cost:0'], feed_dict=feed_dict)

            avg_cost += c
            count += 1

    weights, bias = sess.run(['weights:0', 'bias:0'])
    print('\nWeights:\n', weights)
    print('\nBias:\n', bias)
```
- n_features: Number of features (columns) in the dataset.
- n_samples: Number of samples (rows) in the dataset.
- learning_rate: Learning rate for the gradient descent optimizer.
- training_epochs: Number of training epochs (iterations over the entire dataset).
- batch_size: Number of samples per batch for mini-batch gradient descent.
- weights: A variable initialized to zeros, with shape [n_features, 1] (for logistic regression).
- tf_x: Placeholder for input data (features). The shape [None, n_features] means it can take any number of rows but must have n_features columns.
- tf_y: Placeholder for target labels. The shape [None] allows any number of target values.
- bias: A scalar bias initialized to 0.
- linear = tf.matmul(tf_x, params['weights']) + params['bias']: This performs the linear combination of inputs and weights (i.e., XW + b).
- pred_proba = tf.sigmoid(linear, name='predict_probas'): Applies the sigmoid activation function to the linear output, converting it into a probability value between 0 and 1.
- r = tf.reshape(pred_proba, [-1]): Reshapes the predicted probabilities to a flat array.
- cost: Implements the binary cross-entropy loss function where p is the predicted probability and y is the true label.
	- optimizer: Uses gradient descent to minimize the cost.
	- train = optimizer.minimize(cost, name='train'): Defines the training operation, which updates the model’s weights and bias by minimizing the cost.
- pred_labels = tf.round(tf.reshape(pred_proba, [-1])): Converts the predicted probabilities to binary labels (0 or 1).
- correct_prediction: Compares the predicted labels with the true labels.
- accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)): Computes the accuracy by averaging the correct predictions.

Epoch: 000 | AvgCost: nan | Train/Valid ACC: 0.53/0.40
Epoch: 001 | AvgCost: 4.221 | Train/Valid ACC: 1.00/1.00
Epoch: 002 | AvgCost: 1.225 | Train/Valid ACC: 1.00/1.00
Epoch: 003 | AvgCost: 0.610 | Train/Valid ACC: 1.00/1.00
Epoch: 004 | AvgCost: 0.376 | Train/Valid ACC: 1.00/1.00
Epoch: 005 | AvgCost: 0.259 | Train/Valid ACC: 1.00/1.00
Epoch: 006 | AvgCost: 0.191 | Train/Valid ACC: 1.00/1.00
Epoch: 007 | AvgCost: 0.148 | Train/Valid ACC: 1.00/1.00
Epoch: 008 | AvgCost: 0.119 | Train/Valid ACC: 1.00/1.00
Epoch: 009 | AvgCost: 0.098 | Train/Valid ACC: 1.00/1.00
Epoch: 010 | AvgCost: 0.082 | Train/Valid ACC: 1.00/1.00
Epoch: 011 | AvgCost: 0.070 | Train/Valid ACC: 1.00/1.00
Epoch: 012 | AvgCost: 0.061 | Train/Valid ACC: 1.00/1.00
Epoch: 013 | AvgCost: 0.053 | Train/Valid ACC: 1.00/1.00
Epoch: 014 | AvgCost: 0.047 | Train/Valid ACC: 1.00/1.00

Weights:
 [[ 3.31176686]
 [-2.40808702]]

Bias:
 [[-0.01001291]]

 ----

## 2. Softmax regression (multinomial logistic regression)

## Pytorch

Softmax regression, also known as multinomial logistic regression, is a generalization of logistic regression used for multi-class classification problems. It extends the binary classification capabilities of logistic regression to handle multiple classes.

### Dataset
```python
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_seed = 123
learning_rate = 0.1
num_epochs = 10
batch_size = 256

num_features = 784
num_classes = 10

train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle-True)
```

- define hyperparamets, train and test datasets using the MNIST dataset from torchvision library
- in this case, the number of classes are 10 and features are 784 which is calculated from (28x28) dimensions

### Defining softmax regression class

```python
class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
        
    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

model = SoftmaxRegression(num_features=num_features,
                          num_classes=num_classes)

model.to(device)

##########################
### COST AND OPTIMIZER
##########################

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
```

1. `class SoftmaxRegression(torch.nn.Module):`: This line starts defining a new class named `SoftmaxRegression`, which inherits from the base `torch.nn.Module` class.
2. `__init__(self, num_features, num_classes):`: This is the constructor method for our custom class. It takes in two arguments - `num_features` and `num_classes`. The former represents the number of input features (columns) in the data, while the latter represents the number of output classes (or labels).
3. `super(SoftmaxRegression, self).__init__()`: This line initializes the parent `Module` class with our custom class, ensuring proper initialization and functionality.
4. `self.linear = torch.nn.Linear(num_features, num_classes)`: Here, we create a new linear layer (`torch.nn.Linear`) with the given number of input features and output classes. Linear layers are used to model the relationship between input features and output classes in machine learning models.
5. `self.linear.weight.detach().zero_()` and `self.linear.bias.detach().zero_()`: We initialize both the weights (matrix) and biases of our linear layer with zeros to ensure that the model starts from a clean state.
6. `def forward(self, x):`: This method is defined as the "forward pass" for our custom SoftmaxRegression model. The input data (`x`) is passed through our newly created linear layer using `logits = self.linear(x)`.
7. `probas = F.softmax(logits, dim=1)`: We apply the `F.softmax()` function from PyTorch to transform our logits (predicted log probabilities) into class probabilities (or softmax outputs) using the given `dim` (axis) to be 1 for multi-class problems.

#### Training 

```python
# Manual seed for deterministic data loader
torch.manual_seed(random_seed)


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    
    for features, targets in data_loader:
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        
    return correct_pred.float() / num_examples * 100
    

for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        
        # note that the PyTorch implementation of
        # CrossEntropyLoss works with logits, not
        # probabilities
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_dataset)//batch_size, cost))
            
    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader)))
```

1. `torch.manual_seed(random_seed)` is called to set the random seed, ensuring consistency across runs with different values of random seeds.
2. `compute_accuracy` function computes the training accuracy of the neural network model given a DataLoader. It iterates over each batch in the DataLoader and computes the maximum logits tensor for that batch. Then it checks which label is most frequent by comparing labels to targets, incrementing correct\_pred accordingly. Finally, this function computes and returns the percentage of correct predictions out of total number of examples.
3. Training loop:
1. `for epoch in range(num_epochs):` starts a for loop over defined number of epochs.
2. `for batch_idx, (features, targets) in enumerate(train_loader):` begins iterations through batches in the training DataLoader. Each batch consists of input features and target labels.
3. Inside each epoch iteration:
    a) `features = features.view(-1, 28*28).to(device)` reshapes input tensor to size [N, flattened_MNIST], then sends it to the specified device.
    b) `targets = targets.to(device)` does the same for target tensor.
    c) Calling `model(features)` executes forward pass with given inputs, and returns logits (log-probabilities) and probabilities.
    d) Computing cross entropy loss `cost`: CrossEntropyLoss computes loss between the ground truth targets and the predicted labels using logits instead of probabilities.
    e) Clearing gradient computation for optimizer with `optimizer.zero_grad()`.
    f) Backward propagation: `cost.backward()` calculates gradients based on the computed cost.
    g) Optimizer's update: `optimizer.step()` updates the neural network model's parameters based on computed gradients.
    h) If not batch\_idx % 50, logs progress through training iterations.
4. At end of each epoch, `compute_accuracy(model, train_loader)` is called to compute and log the percentage of correct predictions on the full MNIST validation dataset.


------
### 3. RNNs

Recurrent Neural Networks (RNNs) are a type of neural network architecture designed to process sequential data, where the output at each step depends on both the current input and the previous hidden state. RNNs have a unique architecture that allows them to maintain an internal memory or "hidden state" which carries information from one time step to the next, enabling them to model temporal dependencies in data.

The basic building blocks of an RNN include:

1. Input Layer: This layer consists of input units that take a single data point (feature vector) x\_t as input at each time step t.

2. Hidden State and Cell or Gating Vector: The hidden state h\_t and the cell or gating vector g\_t are the internal variables of an RNN. The hidden state h\_t represents the network's understanding or memory of the previous time step, while the gating vector g\_t determines how much of the previous hidden state should be retained for the next time step and how much new information from the current input should be incorporated.

3. Output Layer: This layer produces an output y\_t based on the current hidden state h\_t and the weighted sum of the inputs (x\_t and h_(t-1)). In many applications, the output y\_t represents a probability distribution over classes or a prediction for the next time step.

4. Weights: RNNs use weights to determine the connections between different layers. The weights are updated during training using backpropagation through time (BPTT).

5. Activation Functions: The activation functions applied at each layer help introduce non-linearity into the model, enabling it to learn complex representations. Common choices for activation functions in RNNs include sigmoid, tanh, and ReLU.


```python
class RNN(torch.nn.Module):
    def __init__(self, input_size, embed_size,
                 hidden_size, output_size, num_layers):
        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embed = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(input_size=embed_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.init_hidden = torch.nn.Parameter(torch.zeros(
                                              num_layers, 1, hidden_size))
    
    def forward(self, features, hidden):
        embedded = self.embed(features.view(1, -1))
        output, hidden = self.gru(embedded.view(1, 1, -1), hidden)
        output = self.fc(output.view(1, -1))
        return output, hidden
      
    def init_zero_state(self):
        init_hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(DEVICE)
        return init_hidden
```
This RNN (Recurrent Neural Network) class definition in PyTorch follows the architecture of processing sequential data using hidden states and cell vectors. The code defines a neural network with input layers, hidden layers, and output layers, allowing for processing one time step at a time.

1. Inheritance: The constructor inherits from torch.nn.Module which provides the foundation for defining a custom neural network.

2. Initialization: Self-defined variables, including input size, embed size, hidden size, and output size, along with the number of layers.

3. Embedding Layer: Defining an embedding layer using torch.nn.Embedding to transform input features into hidden states for further processing by RNN.

4. GRU (Gated Recurrent Unit) Layer: A gated recurrent unit is defined as a part of the network using torch.nn.GRU, providing non-linear transformations based on previous hidden states and cell vectors.

5. Linear Layer: A fully connected layer for converting hidden states to output dimensions.

6. Initial Hidden State: This variable initializes zero state for the RNN with the given number of layers, dimensions, and device.

7. Forward Method: The forward method defines one time step inference by applying embedding, GRU, and linear transformations on input features and previous hidden states while returning output and updated hidden states.

8. Initial Hidden State Initialization: A zero state initialization method for RNN to initialize its hidden state.


```python
torch.manual_seed(RANDOM_SEED)
model = RNN(len(string.printable), EMBEDDING_DIM, HIDDEN_DIM, len(string.printable), NUM_HIDDEN)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate(model, prime_str='A', predict_len=100, temperature=0.8):
    ## based on https://github.com/spro/practical-pytorch/
    ## blob/master/char-rnn-generation/char-rnn-generation.ipynb

    hidden = model.init_zero_state()
    prime_input = char_to_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p].to(DEVICE), hidden.to(DEVICE))
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = model(inp.to(DEVICE), hidden.to(DEVICE))
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = string.printable[top_i]
        predicted += predicted_char
        inp = char_to_tensor(predicted_char)

    return predicted

start_time = time.time()
for iteration in range(NUM_ITER):

    
    ### FORWARD AND BACK PROP

    hidden = model.init_zero_state()
    optimizer.zero_grad()
    
    loss = 0.
    inputs, targets = draw_random_sample(textfile)
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    for c in range(TEXT_PORTION_SIZE):
        outputs, hidden = model(inputs[c], hidden)
        loss += F.cross_entropy(outputs, targets[c].view(1))

    loss /= TEXT_PORTION_SIZE
    loss.backward()
    
    ### UPDATE MODEL PARAMETERS
    optimizer.step()

    ### LOGGING
    with torch.set_grad_enabled(False):
      if iteration % 1000 == 0:
          print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
          print(f'Iteration {iteration} | Loss {loss.item():.2f}\n\n')
          print(evaluate(model, 'Th', 200), '\n')
          print(50*'=')
```
1. Set the random seed for reproducibility
2. Build a model using RNN architecture, with input size from printable strings (length of alphabet), embedding dimension, and hidden layer dimensions and numbers.
3. Move the model to a specific device for faster computation.
4. Initialize an optimizer Adam for learning rate.
5. Evaluation function to generate text, using prime string 'A', prediction length of 100 tokens, and temperature for randomness. This function builds up hidden state by taking each character of the prime string as input, then generates output from this hidden state until predict_len is reached, and returns generated sequence.
6. Start a timer to record elapsed time during training.
7. Training loop: for i in range(number of iterations), 
    a) Draw random samples from the text file (input and target strings).
    b) Move these tensors to GPU device.
    c) For each character index, compute outputs and hidden state using the model inputs and previous hidden state.
        1) Compute cross entropy loss for this step with targets.
        2) Sum up losses from all characters in the text portion size.
        3) Divide total loss by the number of characters to get average loss per character.
    d) Backward pass grads through computation graph, compute gradient with loss.
    e) Apply optimizer step to update model parameters.
8. After each thousand iterations, log training progress: print iteration, loss, and generated text using evaluate function.