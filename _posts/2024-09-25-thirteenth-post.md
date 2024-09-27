---
layout: post
title: All about Autoencoders 
date:   2024-09-18 12:54:15 +0800
categories: jekyll update
---

### What are autoencoders?
 Autoencoders are a type of artificial neural network (ANN) used primarily for unsupervised learning, feature extraction, and dimensionality reduction. They are designed to learn efficient codings of input data by training the model to reconstruct its inputs from an encoded representation.

 The term "autoencoder" refers to the fact that the network learns to encode and decode its inputs automatically without being explicitly told what features are important or how the data should be transformed.

 Autoencoders consists of:
 1. Encoder: The encoder is the part of the network that takes the input data and maps it to a lower-dimensional latent space or hidden representation. It usually consists of several fully connected dense layers with progressively fewer neurons. ReLU activation functions are commonly used between the layers, but other non-linearities like LeakyReLU or tanh can also be employed.

 2. Latent space: This is the hidden layer or representation space where the neural network compresses the input data to a lower dimension while retaining essential information for accurate reconstruction.

 3. Decoder: The decoder is responsible for reconstructing the original input from its encoded representation in the bottleneck. It consists of dense layers with progressively more neurons than the encoder, allowing the network to recover the original input features as closely as possible. Activation functions like sigmoid, tanh, or ReLU can be used between the decoder's layers depending on the specific application.

 <span style="color: red;">Q: what are latent space?</span>

### Loading the MNIST data 
```python
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 123
learning_rate = 0.005
num_epochs = 5
batch_size = 256

# Architecture
num_features = 784
num_hidden_1 = 32


##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
```

### Defining the model (vanilla autoencoder)
```python
class Autoencoder(torch.nn.Module):

    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        
        ### ENCODER
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # The following to lones are not necessary, 
        # but used here to demonstrate how to access the weights
        # and use a different weight initialization.
        # By default, PyTorch uses Xavier/Glorot initialization, which
        # should usually be preferred.
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        
        ### DECODER
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        

    def forward(self, x):
        
        ### ENCODER
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)
        
        ### DECODER
        logits = self.linear_2(encoded)
        decoded = torch.sigmoid(logits)
        
        return decoded

```
- ```self.linear_1``` takes the input shape num_features and has a hidden layer size ```num_hidden_1```
- ```self.linear_1.weight.detach().normal_(0.0,0.1)``` - sets the initial random weights and biases for the encoder's first Linear layer using the normal method, which generates random values from a normal distribution (mean=0, std=0.1). The detach() method is used to prevent these weights and biases from being automatically moved to the computation graph during backpropagation.
- the decoder part has ```self.linear_2``` which has hidden layer size of ```num_hidden_1``` and output shape of num_features. The weights and bias are initialized the same way as encoder.
- the ```forward``` function for the network  performs the encoding and decoding operations. It takes an input tensor x, passes it through the encoder, and saves the encoded tensor as encoded. The decoder is then applied to this tensor and saves the resulting decoded tensor as decoded.

### Initate and train
```python
torch.manual_seed(random_seed)
model = Autoencoder(num_features=num_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

start_time = time.time()
for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        # don't need labels, only the images (features)
        features = features.view(-1, 28*28).to(device)
            
        ### FORWARD AND BACK PROP
        decoded = model(features)
        cost = F.binary_cross_entropy(decoded, features)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))
            
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
```



1. The script initializes some variables, sets the number of epochs (iterations over the entire dataset), and gets a DataLoader object to efficiently load batches from the training dataset.

2. A for loop runs through all the epochs. For each epoch, another nested loop iterates through each batch in the dataset.

3. The features tensor is extracted from the current batch without requiring the corresponding targets (labels) and reshaped to fit the input dimensions of the model. This step ensures that the input shape matches the expected size for the Autoencoder network.

4. Inside this loop, we perform a forward pass by passing the features through the Autoencoder network using the `model(features)` function call. The output is stored in the decoded tensor.

5. We calculate the loss or cost (binary cross-entropy in this case) between the original input features and the decoded output. This value measures how well our model has performed during the forward pass.

6. The gradients for the entire computation graph are zeroed out using `optimizer.zero_grad()` to prepare for backpropagation (backward pass). The aim is to compute the gradients with respect to each weight and bias in the network, allowing us to update them later.

7. In the backward pass, we call the cost tensor's `backward()` method, which computes the gradients using the chain rule of calculus. This gradient computation flows through the entire neural network, enabling us to find the partial derivatives (gradients) with respect to all trainable parameters.

8. We update the model parameters by invoking the optimizer's `step()` function, which applies the gradients from the backward pass and updates the model weights and biases according to the specified learning rate.

### Use cases for Autoencoders
 
Autoencoders are a type of neural network architecture that can be used for various applications due to their ability to learn efficient and compact representations of input data. Here are some common use cases for Autoencoders:


1. Dimensionality reduction: Autoencoders can be employed as an unsupervised dimensionality reduction method, where the encoder maps the input data into a lower-dimensional latent space (hidden representation), and the decoder reconstructs the original data from this reduced representation. This can help to preserve important features of the data while reducing its complexity for further analysis or visualization.



2. Denoising: Autoencoders can be trained on noisy versions of their input data, enabling them to learn robust representations that can remove noise and recover the original data. This is particularly useful in applications where data may be corrupted or contaminated.



3. Anomaly detection: By training an Autoencoder on normal data, it can learn a typical representation of the input data. New, unseen data can then be compared against this learned representation to identify anomalies. This is done by measuring the reconstruction error between the original data and the reconstructed data.



4. Generation of new data: Autoencoders can generate new data by sampling from the latent space and passing it through the decoder. This can be useful for applications such as content creation, image synthesis, or text generation.



5. Compression: By learning a compressed representation of input data using an Autoencoder, the encoded data can be transmitted or stored more efficiently. This is particularly useful in applications where large amounts of data need to be processed and communicated, such as image processing, video compression, or speech recognition.



6. Feature extraction: The bottleneck layer of an Autoencoder's encoder can be considered a feature extractor that maps the input data into a lower-dimensional space, preserving important features for further analysis or modeling. This can be helpful in various applications such as classification, clustering, and regression.



7. Pretraining: Autoencoders can serve as powerful pretraining models for other deep learning architectures, enabling them to learn better representations of their input data before fine-tuning on specific tasks. This can lead to improved performance and faster convergence during the later stages of training.

### Variational autoencoders

Variational Autoencoders can be explained through their architecture and training process:

1. **Architecture**: A VAE consists of an encoder network, a decoder network, and a reparameterization trick. The encoder network takes the input data x as input and outputs a latent variable z (mean μ and standard deviation σ), while the decoder network reconstructs the input from the sampled latent variables z.

2. **Loss function**: The objective of VAE training is to minimize two losses: reconstruction loss L_recon and KL divergence loss L_kl. The reconstruction loss measures the difference between the original input data x and the reconstructed data ŷ. The KL divergence loss enforces a Gaussian distribution on the latent variables, keeping them close to a standard normal distribution. This ensures that the learned representation is not too complex or unrealistic.

The overall loss function for a single training example can be expressed as:

L = L_recon(x, ŷ) + β * L_kl(z; μ, σ)

where β is a hyperparameter controlling the weight of KL divergence loss.

3. **Training process**: During training, the input data x and its corresponding reconstruction ŷ are passed through both the encoder and decoder networks to calculate the reconstruction loss. The latent variables z are then sampled using the reparameterization trick, which involves adding a noise vector e to the mean μ and passing it through a deterministic function (σ = σ(μ)) before computing the KL divergence loss. This ensures that the training process remains deterministic.

4. **Generative modeling**: Once trained, a VAE can be used for generative modeling by sampling z from their learned distribution using Rejection Sampling or another sampling algorithm. The decoder network then reconstructs a new data point x' from the sampled latent variables z.

5. **Theory behind Variational Autoencoders**: Variational Autoencoders extend the concept of traditional Autoencoders by incorporating probabilistic representations of input data. This allows VAEs to learn more complex and meaningful latent representations compared to simple reconstruction tasks in standard Autoencoders.

By combining the strengths of both Autoencoders (reconstruction loss) and Variational Bayes (probabilistic latent variables), VAEs can achieve generative modeling capabilities while keeping the training process computationally efficient and stable

```python
class ConditionalVariationalAutoencoder(torch.nn.Module):

    def __init__(self, num_features, num_latent, num_classes):
        super(ConditionalVariationalAutoencoder, self).__init__()
        
        self.num_classes = num_classes
        
        
        ###############
        # ENCODER
        ##############
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.enc_conv_1 = torch.nn.Conv2d(in_channels=1+self.num_classes,
                                          out_channels=16,
                                          kernel_size=(6, 6),
                                          stride=(2, 2),
                                          padding=0)

        self.enc_conv_2 = torch.nn.Conv2d(in_channels=16,
                                          out_channels=32,
                                          kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=0)                 
        
        self.enc_conv_3 = torch.nn.Conv2d(in_channels=32,
                                          out_channels=64,
                                          kernel_size=(2, 2),
                                          stride=(2, 2),
                                          padding=0)                     
        
        self.z_mean = torch.nn.Linear(64*2*2, num_latent)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use 
        # an exponential function
        self.z_log_var = torch.nn.Linear(64*2*2, num_latent)
        
        
        
        ###############
        # DECODER
        ##############
        
        self.dec_linear_1 = torch.nn.Linear(num_latent+self.num_classes, 64*2*2)
               
        self.dec_deconv_1 = torch.nn.ConvTranspose2d(in_channels=64,
                                                     out_channels=32,
                                                     kernel_size=(2, 2),
                                                     stride=(2, 2),
                                                     padding=0)
                                 
        self.dec_deconv_2 = torch.nn.ConvTranspose2d(in_channels=32,
                                                     out_channels=16,
                                                     kernel_size=(4, 4),
                                                     stride=(3, 3),
                                                     padding=1)
        
        self.dec_deconv_3 = torch.nn.ConvTranspose2d(in_channels=16,
                                                     out_channels=1,
                                                     kernel_size=(6, 6),
                                                     stride=(3, 3),
                                                     padding=4)        


    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def encoder(self, features, targets):
        
        ### Add condition
        onehot_targets = to_onehot(targets, self.num_classes, device)
        onehot_targets = onehot_targets.view(-1, self.num_classes, 1, 1)
        
        ones = torch.ones(features.size()[0], 
                          self.num_classes,
                          features.size()[2], 
                          features.size()[3], 
                          dtype=features.dtype).to(device)
        ones = ones * onehot_targets
        x = torch.cat((features, ones), dim=1)
        
        x = self.enc_conv_1(x)
        x = F.leaky_relu(x)
        #print('conv1 out:', x.size())
        
        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)
        #print('conv2 out:', x.size())
        
        x = self.enc_conv_3(x)
        x = F.leaky_relu(x)
        #print('conv3 out:', x.size())
        
        z_mean = self.z_mean(x.view(-1, 64*2*2))
        z_log_var = self.z_log_var(x.view(-1, 64*2*2))
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded
    
    def decoder(self, encoded, targets):
        ### Add condition
        onehot_targets = to_onehot(targets, self.num_classes, device)
        encoded = torch.cat((encoded, onehot_targets), dim=1)        
        
        x = self.dec_linear_1(encoded)
        x = x.view(-1, 64, 2, 2)
        
        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())
        
        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)
        #print('deconv2 out:', x.size())
        
        x = self.dec_deconv_3(x)
        x = F.leaky_relu(x)
        #print('deconv1 out:', x.size())
        
        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, features, targets):
        
        z_mean, z_log_var, encoded = self.encoder(features, targets)
        decoded = self.decoder(encoded, targets)
        
        return z_mean, z_log_var, encoded, decoded
```
1. `self.enc_conv_1` to `self.enc_conv_3`: These are the three convolutional layers in the encoder network, each followed by a LeakyReLU activation function. Each convolutional layer reduces the spatial dimensions of the input feature map and increases the number of channels. The purpose is to extract more abstract features from the input image as we go deeper into the network.
2. `self.z_mean` and `self.z_log_var`: These are the two fully connected (linear) layers used in the encoder network for the latent variable representation of the input image. The first layer, `self.z_mean`, outputs the mean of the latent variable distribution, while the second layer, `self.z_log_var`, outputs the log-standard deviation of the latent variable distribution. These two representations are used to sample from the latent variable distribution and encode the input image into a lower dimensional representation.
3. `self.reparameterize`: This is a utility function that takes the mean and log-standard deviation of the latent variable distribution and samples from it using the reparameterization trick to make the training process deterministic.
4. `self.dec_linear_1` to `self.dec_deconv_3`: These are the three fully connected layers and deconvolutional layers in the decoder network, each followed by a LeakyReLU activation function (except for the final sigmoid activation function). The purpose of these layers is to learn the inverse mapping from the lower dimensional latent space back to the original image space.
5. `to_onehot`: This is a utility function used to convert target labels into one-hot encoded representations, which are then concatenated with the latent representation before passing it through the decoder network. This helps the decoder network learn to conditionally generate images based on the input target labels.
6. `self.forward`: This is the main forward pass function that takes an input image (features) and its corresponding label (targets), encodes the image using the encoder network, decodes the latent representation using the decoder network, and outputs the generated image as well as the reconstruced target label.

The functions used in this architecture include: `nn.Conv2d`, `nn.BatchNorm2d`, `nn.LeakyReLU(0.2)`, `nn.Linear()`, `nn.LogSoftmax()`, and a custom function `to_onehot`. These functions are PyTorch equivalents of TensorFlow operations and help us define, initialize, and apply each layer in the network architecture.


### Training

```python
start_time = time.time()

for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        z_mean, z_log_var, encoded, decoded = model(features, targets)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean**2 + 
                                torch.exp(z_log_var) - z_log_var - 1)).sum()
        
        
        ### Add condition
        # Disabled for reconstruction loss as it gives poor results
        """
        onehot_targets = to_onehot(targets, num_classes, device)
        onehot_targets = onehot_targets.view(-1, num_classes, 1, 1)
        
        ones = torch.ones(features.size()[0], 
                          num_classes,
                          features.size()[2], 
                          features.size()[3], 
                          dtype=features.dtype).to(device)
        ones = ones * onehot_targets
        x_con = torch.cat((features, ones), dim=1)
        """
        
        ### Compute loss
        #pixelwise_bce = F.binary_cross_entropy(decoded, x_con, reduction='sum')
        pixelwise_bce = F.binary_cross_entropy(decoded, features, reduction='sum')
        cost = kl_divergence + pixelwise_bce
        
        ### UPDATE MODEL PARAMETERS
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))
            
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
```

The code first sets up some necessary parameters such as the device (CPU or GPU), batch size, number of epochs, data loader, and defines a function `to_onehot` for converting target labels into one-hot encoded representations.

Next, it initializes the VAE model by defining the encoder and decoder networks using convolutional layers for feature extraction and fully connected layers for encoding and decoding. The `ReconstructionLoss` function is also defined to compute the pixelwise binary cross-entropy loss between the decoded and original images, as well as the Kullback-Leibler (KL) divergence between the encoded latent variables and a standard normal distribution.

The main training loop starts by loading the data using the data loader and initializing some variables for logging. The VAE model is then put in train mode and the optimizer is set up with Adam as the optimization algorithm.

The training loop iterates over the number of epochs, and within each epoch, it iterates through the batches of images using the data loader. For each batch, the features are extracted using the encoder network and passed through the decoder network to obtain the reconstructed image. The cost is then computed as the sum of the reconstruction loss (pixelwise binary cross-entropy) and KL divergence.
    
    - KL divergence measures the amount of information lost when approximating one distribution (P) with another distribution (Q). It is not symmetric, meaning that P(X)|KLQ(X) ≠ Q(X)|KLP(X), where P and Q are probability distributions over some random variable X. 

The model parameters are updated by backpropagating the gradients and optimizing using the Adam optimizer. The training progress is logged after every 50 batches, including the epoch number, batch number, cost, and time elapsed.

