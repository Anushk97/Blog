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
ds = np.lib.DataSource() #open and read iris dataset
fp = ds.open('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

# convert data into numpy array
x = np.genfromtxt(BytesIO(fp.read().encode()), delimiter=',', usecols=range(2), max_rows=100)

# create a binary classification problem with 50 0 labels and 50 1 labels
y = np.zeroes(100)
y[50:] = 1

np.random.seed(1)

#create and shuffle indices
idx = np.arange(y.shape[0])
np.random.shuffle(idx)

# split into train and test sets
X_test, y_test = x[idx[:25]], y[idx[:25]]
X_train, y_train = x[idx[25:]], y[idx[25:]]

#normalize the data by computing mean and std and then standatdize it in train and test sets
mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train, X_test = (X_train - mu) / std, (X_test - mu) / std
```

### 🦒 Low level implementation using manual gradients

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#perform element wise replacement based on a condition. used to extract labels from prediction
def custom_where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

# model with one hidden layer and bias term
class LogisticRegression1():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, 
                                   dtype=torch.float32, device=device)
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)

#compute output given input x
    def forward(self, x):
        linear = torch.add(torch.mm(x, self.weights), self.bias)
        probas = self._sigmoid(linear)
        return probas

#calculte errors based on true labels
    def backward(self, probas, y):  
        errors = y - probas.view(-1)
        return errors
            
#extract labels using custom where function
    def predict_labels(self, x):
        probas = self.forward(x)
        labels = custom_where(probas >= .5, 1, 0)
        return labels    
            
    def evaluate(self, x, y):
        labels = self.predict_labels(x).float()
        accuracy = torch.sum(labels.view(-1) == y) / y.size()[0]
        return accuracy
    
#for forward prop 
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

#convert numpy arrays into pytorch tensor for training
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

#initialize lr model with 2 features 
logr = LogisticRegression1(num_features=2)

#train the model for 10 epochs
logr.train(X_train_tensor, y_train_tensor, num_epochs=10, learning_rate=0.1)

test_acc = logr.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc*100))

print('\nModel parameters:')
print('  Weights: %s' % logr.weights)
print('  Bias: %s' % logr.bias)
```
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
            
            #### manual update weights ####
            
            #detach the weights from computation graph
            tmp = self.weights.detach()
            tmp -= learning_rate * self.weights.grad #perform gradient descent
            
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

### 🦄 High level implementation using nn.Module

```python
class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features,1) #fully connected layer with weights and bias initialized randomly
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x):
        logits = self.linear(x) #apply linear layer to input x
        probas = torch.sigmoid(logits) #apply sigmoid to logits to produe probabilities
        return probas


def comp_accuracy(label_var, pred_probas):
    pred_labels = custom_where((pred_probas > 0.5).float(), 1, 0).view(-1)
    acc = torch.sum(pred_labels == label_var.view(-1)).float() / label_var.size(0)
    return acc

#initialize model with 2 features
model = LogisticRegression(num_features=2).to(device)

#Binary cross entropy loss used for binary classif
cost_fn = torch.nn.BCELoss(reduction='sum')

#stochastic gradient descent for optimization algo
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#convert array to tensors for training
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)

#training loop
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
import tensorflow as tf

def iterate_minibatches(arrays, batch_size, shuffle=False, seed=None):
    rgen = np.random.RandomState(seed)
    #array used to slice inputs to mini batches
    indices = np.arange(arrays[0].shape[0])

    if shuffle:
        rgen.shuffle(indices)

    #create mini batches
    for start_idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
        index_slice = indices[start_idx:start_idx + batch_size]

        yield (ary[index_slice] for ary in arrays)

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
    # perform linear combination 
    linear = tf.matmul(tf_x, params['weights']) + params['bias']
    
    #apply sigmoid for probabilities
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

#hyper parameters
random_seed = 123
learning_rate = 0.1
num_epochs = 10
batch_size = 256

num_features = 784
num_classes = 10

#download and transform datasets
train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle-True)

class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
        
    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1) #transform logits into class probabilities
        return logits, probas

model = SoftmaxRegression(num_features=num_features,
                          num_classes=num_classes)

model.to(device)

##########################
### COST AND OPTIMIZER
##########################

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

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
    
#training loop
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

------
### 3. RNNs

Recurrent Neural Networks (RNNs) are a type of neural network architecture designed to process sequential data, where the output at each step depends on both the current input and the previous hidden state. RNNs have a unique architecture that allows them to maintain an internal memory or "hidden state" which carries information from one time step to the next, enabling them to model temporal dependencies in data.

The basic building blocks of an RNN include:

1. Input Layer: This layer consists of input units that take a single data point (feature vector) x\_t as input at each time step t.

2. Hidden State and Cell or Gating Vector: The hidden state h\_t and the cell or gating vector g\_t are the internal variables of an RNN. The hidden state h\_t represents the network's understanding or memory of the previous time step, while the gating vector g\_t determines how much of the previous hidden state should be retained for the next time step and how much new information from the current input should be incorporated.

3. Output Layer: This layer produces an output y\_t based on the current hidden state h\_t and the weighted sum of the inputs (x\_t and h_(t-1)). In many applications, the output y\_t represents a probability distribution over classes or a prediction for the next time step.

4. Weights: RNNs use weights to determine the connections between different layers. The weights are updated during training using backpropagation through time (BPTT).

5. Activation Functions: The activation functions applied at each layer help introduce non-linearity into the model, enabling it to learn complex representations. Common choices for activation functions in RNNs include sigmoid, tanh, and ReLU.

download data: 
!wget http://www.gutenberg.org/files/98/98-0.txt

```python
#settings and parameters
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT_PORTION_SIZE = 200

NUM_ITER = 20000
LEARNING_RATE = 0.005
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
NUM_HIDDEN = 1

with open('./98-0.txt', 'r') as f:
    textfile = f.read()

# convert special characters
textfile = unidecode.unidecode(textfile)

# strip extra whitespaces
textfile = re.sub(' +',' ', textfile)

TEXT_LENGTH = len(textfile)

def random_portion(textfile):
    start_index = random.randint(0, TEXT_LENGTH - TEXT_PORTION_SIZE)
    end_index = start_index + TEXT_PORTION_SIZE + 1
    return textfile[start_index:end_index]

print(random_portion(textfile))

#convert chars to tensor for training
def char_to_tensor(text):
    lst = [string.printable.index(c) for c in text]
    tensor = torch.tensor(lst).long()
    return tensor

print(char_to_tensor('abcDEF'))

### Model ###
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
