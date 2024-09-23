---
layout: post
title: Deep learning models implemented in pytorch and TF
date:   2024-09-18 12:54:15 +0800
categories: jekyll update
---

## 1. logistic regression

### Pytorch
### 🏛️ Preparing dataset

#### Data source and download
```
ds = np.lib.DataSource()
fp = ds.open('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
```
- The np.lib.DataSource() is used to open and read the Iris dataset from the specified URL.
- fp = ds.open(...) fetches the file and stores it in fp as an open file-like object.

#### Loading the data
```
x = np.genfromtxt(BytesIO(fp.read().encode()), delimiter=',', usecols=range(2), max_rows=100)
```
- The dataset is read into a NumPy array x using np.genfromtxt().
- BytesIO(fp.read().encode()): Converts the content of the file into bytes, so it can be processed as a file-like object in memory.
- delimiter=',': Specifies that the file is comma-separated.
- usecols=range(2): Only the first two columns of the dataset are used (sepal length and sepal width).
- max_rows=100: Only the first 100 rows are loaded (this includes data for the first two species of the Iris dataset).

#### Creating labels
```
y = np.zeroes(100)
y[50:] = 1
```
- A label array y of 100 zeros is created, where the first 50 rows (corresponding to the first Iris species) are labeled as 0 and the next 50 rows (for the second species) are labeled as 1.
- This creates a binary classification problem between two species.

#### Shuffling the data
```
np.random.seed(1)
idx = np.arange(y.shape[0])
np.random.shuffle(idx)
```
- np.random.seed(1) ensures that the shuffling is reproducible.
- idx = np.arange(y.shape[0]) creates an array of indices from 0 to 99.
- np.random.shuffle(idx) shuffles these indices, which will be used to randomly split the data into training and test sets.

#### Split into train and test 
```
X_test, y_test = x[idx[:25]], y[idx[:25]]
X_train, y_train = x[idx[25:]], y[idx[25:]]
```
- The shuffled indices are used to split the data into training and test sets.
- The first 25 shuffled samples are assigned to the test set (X_test and y_test).
- The remaining 75 samples are used for training (X_train and y_train).

#### Normalize data
```
mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train, X_test = (X_train - mu) / std, (X_test - mu) / std
```
- mu: The mean of the X_train data is computed along each feature (sepal length and sepal width).
- std: The standard deviation of X_train data is also computed.
- X_train, X_test = (X_train - mu) / std, (X_test - mu) / std: Both the training and test data are normalized using the mean and standard deviation of the training data. This step is important to standardize the data, ensuring that the features have a mean of 0 and a standard deviation of 1, which improves model performance.

### 🦒 Low level implementation using manual gradients
```
def custom_where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)
```
- This function mimics the behavior of torch.where. It selects values from x_1 where the condition cond is True, and values from x_2 where cond is False. The condition cond should be a tensor of 0s and 1s, and the result is computed using element-wise multiplication and addition.

#### Model initialization
```
class LogisticRegression():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, dtype=torch.float32)
        self.bias = torch.zeros(1, dtype=torch.float32)
```
- self.weights: A tensor of zeros representing the weights of the logistic regression model, with a size corresponding to the number of features.
- self.bias: A scalar bias term, also initialized to zero.

#### Forward prop
This method implements the forward pass of logistic regression:
- It computes the linear combination of inputs and weights: z = Wx + b
- It applies the sigmoid function to this linear combination: σ(z) = 1 / (1 + e^(-z))
Theoretically, this represents the probability estimate P(Y=1|X) for binary classification.

```
def forward(self, x):
    linear = torch.add(torch.mm(x, self.weights), self.bias)
    probas = self._sigmoid(linear)
    return probas
```
- torch.mm(x, self.weights): Performs matrix multiplication between the input x and the weights.
- torch.add(...): Adds the bias term to the result of the matrix multiplication.
- self._sigmoid(linear): Applies the sigmoid activation function to the linear combination to get probabilities (values between 0 and 1).

#### Sigmoid activation func
```
def _sigmoid(self, z):
    return 1. / (1. + torch.exp(-z))
```
- converts the linear output to probabilities between 0 and 1, following the logistic regression equation.

#### Backward prop: compute gradients
```
def backward(self, probas, y):
    errors = y - probas.view(-1)
    return errors
```
- computes the error (residual) between the true labels y and the predicted probabilities probas.
- errors = y - probas.view(-1): The difference between actual labels y and the predicted probabilities probas. The .view(-1) flattens the probability tensor to ensure it aligns with the shape of y.

#### Make predictions
```
def predict_labels(self, x):
    probas = self.forward(x)
    labels = customer_where(probas >= .5, 1, 0)
    return labels
```
- probas = self.forward(x): Computes the predicted probabilities.
- custom_where(probas >= .5, 1, 0): Converts probabilities to binary labels (1 if the probability is greater than or equal to 0.5, 0 otherwise).

#### Evaluate: compute accuracy
```
def evaluate(self, x, y):
    labels = self.predict_labels(x).float()
    accuracy = torch.sum(labels.view(-1) == y) / y.size()[0]
    return accuracy
```
- predict_labels(x): Predicts the binary labels for input x.
- torch.sum(labels.view(-1) == y): Compares predicted labels to actual labels y and counts the number of correct predictions.
- Accuracy is calculated as the proportion of correct predictions.

#### Log-loss cost function
```
def _logit_cost(self, y, proba):
    tmp1 = torch.mm(-y.view(1,-1), torch.log(proba))
    tmp2 = torch.mm((1-y).view(1,-1), torch.log(1-proba))
    return tmp1 - tmp2
```
- This computes the logistic regression cost function (also called binary cross-entropy or log-loss).
- The cost function compares the true labels y and predicted probabilities proba:
    - tmp1: Computes the cost for class 1 predictions.
    - tmp2: Computes the cost for class 0 predictions.
- The result is the sum of both terms, which represents the total cost for the current predictions.

#### Training
```
def train(self, x,y,num_epochs, learning_rate = 0.01):
    for e in range(num_epochs):
        probas = self.forward(x)
        errors = self.backward(probas, y)
        neg_grad = torch.mm(x.transpose(0,1), errors.view(-1,1))
        self.weights += learning_rate * neg_grad
        self.bias += learning_rate * torch.sum(errors)

        print('Epoch: %03d' % (e+1), end="")
        print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
        print(' | Cost: %.3f' % self._logit_cost(y, self.forward(x)))
```

- probas = self.forward(x): Computes the predicted probabilities for the current input x.
- errors = self.backward(probas, y): Calculates the error between the predictions and actual labels.
- Gradients:
    - neg_grad = torch.mm(x.transpose(0, 1), errors.view(-1, 1)): Computes the negative gradient of the weights with respect to the error.
    - self.weights += learning_rate * neg_grad: Updates the weights using the gradient and learning rate.
    - self.bias += learning_rate * torch.sum(errors): Updates the bias term using the gradient and learning rate.

##### Convert training data to tensor
```
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
```
- torch.tensor(): This converts the X_train and y_train NumPy arrays into PyTorch tensors, which are required for the model.
- dtype=torch.float32: Ensures the tensors are of type float32, which is needed for the mathematical operations in the model.

##### Instantiate and train
```
logr = LogisticRegression(num_features=2)
logr.train(X_train_tensor, y_train_tensor, num_epochs=10, learning_rate=0.1)
```
- Forward Propagation: The model computes the predicted probabilities using the current weights and bias.
- Backward Propagation: The model calculates the error between the predicted probabilities and actual labels, computes the gradient, and updates the weights and bias using the specified learning rate.
- Accuracy and Cost Logging: After each epoch, the model prints the training accuracy and cost (loss) for monitoring training progress.

Epoch: 001 | Train ACC: 0.000 | Cost: 5.581
Epoch: 002 | Train ACC: 0.000 | Cost: 4.882
Epoch: 003 | Train ACC: 1.000 | Cost: 4.381
Epoch: 004 | Train ACC: 1.000 | Cost: 3.998
Epoch: 005 | Train ACC: 1.000 | Cost: 3.693
Epoch: 006 | Train ACC: 1.000 | Cost: 3.443
Epoch: 007 | Train ACC: 1.000 | Cost: 3.232
Epoch: 008 | Train ACC: 1.000 | Cost: 3.052
Epoch: 009 | Train ACC: 1.000 | Cost: 2.896
Epoch: 010 | Train ACC: 1.000 | Cost: 2.758

Model parameters:
  Weights: tensor([[ 4.2267],
        [-2.9613]], device='cuda:0')
  Bias: tensor([0.0994], device='cuda:0')

##### Evaluate on test set
```
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = logr.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc*100))
```
***Test set accuracy: 100.00%***

### 🦓 Low level implementation using Autograd
```
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

```
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
```
self.linear.weight.detach().zero_()
self.linear.bias.detach().zero_()
```
- self.linear.weight.detach().zero_(): Sets the weights to zero in-place (_ indicates in-place operation). The .detach() ensures that this operation is done outside of the computation graph to prevent tracking of gradients.
- self.linear.bias.detach().zero_(): Similarly, sets the bias to zero.
- This mimics the manual logistic regression model where you started with weights initialized to zero.

#### Forward Prop
```
def forward(self, x):
    logits = self.linear(x)
    probas = torch.sigmoid(logits)
    return probas
```
- logits = self.linear(x): Applies the linear layer to the input x. The linear layer computes the weighted sum of inputs (i.e., the dot product of the input features and the weights, plus the bias).
- probas = torch.sigmoid(logits): Applies the sigmoid function to the logits (the linear combination of inputs) to produce probabilities between 0 and 1, which are necessary for logistic regression.

#### Optimize and train
```
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
```
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

```
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

```
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
 
 ## 2. Softmax regression