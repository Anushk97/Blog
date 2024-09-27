---
layout: post
title: Activation functions
date:   2024-09-18 12:54:15 +0800
categories: jekyll update
---
Activation functions are essential components of artificial neural networks (ANNs) and deep learning models. They introduce non-linearity to these models, allowing them to learn complex relationships between input features and output classes.

Neural networks with only linear operations can only model linearly separable data. However, most real-world problems involve non-linear relationships, making it necessary to incorporate non-linearities in our neural network models. This is where activation functions come into play.

An activation function, f(x), takes a weighted sum of the inputs (the output of a previous layer) and applies a non-linear transformation to introduce non-linearity:

Activation(Wx + b) = f(Wx + b)

Where:
- W is the weight matrix,
- x is the input feature vector or the output from the previous layer,
- b is the bias term,
- Activation() is the activation function applied element-wise to each neuron in a given layer.

The introduction of an activation function at each layer enables the network to learn non-linear transformations and model complex relationships between input features and output classes.

Some common types of activation functions used in machine learning include:

1. Sigmoid Function:
   The sigmoid function maps any real number input to a value within the range [0, 1]. It is defined as:

    σ(x) = f(x) = 1/(1 + e‶(-b)) = 1/(1 + e‶(−Wx)).

   The sigmoid function introduces non-linearity and enables the network to learn complex relationships between input features and output classes. However, its gradients become smaller closer to zero. This can cause problems in deep networks, making other alternatives more common.

2. ReLU (Rectified Linear Unit) Function:
   The ReLU function maps every positive input value to 0, and each negative input value with a threshold to 0. It is defined as:

    σ(x)=f(x)=max(0,x) =ReLU(x)=Max(0,x)

   This activation function introduces non-linearity while preserving computational efficiency and enabling faster training compared to sigmoid functions. However, it has no saturation and its gradients are large for negative input values, causing problems in deep networks called "vanishing gradient problem." To address this issue, alternative activation functions like LeCu (Leaky Rectified Linear Unit), PReLU (Piecewise Rectified Linear Unit) and other non-saturated alternatives with smooth gradients are used.

3. tanCos(x) = sigma(x) = 1/[1 + e‶(-Wx)] = 1/[1 + e‶(-Wx)]

The tanCos function, also called the hyperbolic tangent, is another alternative non-linear activation. This function maps input values to output values within the range (-∞, 0) and (0, ∞). It introduces non-linearity similar to sigmoid functions but with better gradients for negative input values compared to sigmoid. However, it still faces issues like vanishing and saturated gradients.

To address these challenges and introduce new types of activation functions, researchers proposed alternatives such as:

1. ReLU (Rectified Linear Unit) and its variants: PReLU, LeCu (Leaky Rectified Linear unit), ELiU (Extreme Linearly Unbounded unit), etc.

2. Max(0, x) or MAX(0, x) functions: MAx (Maximum Absolute-value) and others.

3. ReLU with randomized negative slopes and other variants: LeGi (Leaky Gaussian-like), ReS (Rectified Stochastic-wise), ReLU_u(ReLU with unclipped gradients), etc.

4. SiLk and its relatives: LSiL (Leaky SILU), SiLN, SiLuG (Significantly Leakier), etc.

5. ELu (Extremely Linear unit) and its variations: ELu\_1 (Extremely Large-input unit), ELu2 (Extremely Large-output unit), ELu_0 (Extremely Larger-than zero unit), etc.

6. Swish, SiLgW and related activation functions: swish(x) = x * exp(1/d), sigmoid(x) = 1 / [1 + e‶(-x)] = sigmoid(x), sigmoid(x / d) = 1 / (1 + e (-x / d)), SiLgW(x)=exp(-x/d)/max[0, exp(-x/d), x], etc.

6. ReLU_p and its derivatives: PReLU_p, LSiLgP (Leaky SiLG-P), SiLu_p (Significantly Leakier-p)
