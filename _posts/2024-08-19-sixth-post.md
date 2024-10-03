---
layout: post
title:  "ML resources 📘"
date:   2024-08-19 12:54:15 +0800
categories: jekyll update
---

### Current Learning resources
[Deep learning Berkeley course](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A)

[Word embeddings](https://lena-voita.github.io/nlp_course/word_embeddings.html#main_content)

[Julia](https://julialang.org/learning/classes/)

[full stack open](https://fullstackopen.com/en/#course-contents)

[transformer visualization](https://bbycroft.net/llm)

## Reading list
### Math for ML books
1. "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy
2. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
3. "Pattern Recognition and Machine Learning" by Christopher Bishop
4. "Mathematics for Machine Learning" by Brian D. Irving
5. "An Introduction to Machine Learning with Applications in Engineering" by Andrew M. Moore



----

### Linear alg

 Linear Algebra plays a fundamental role in Machine Learning (ML) and Deep Learning (DL), providing essential mathematical foundations for various algorithms and models. 

1. Vectors and Matrices:
   A vector is an ordered sequence of numbers, typically real or complex, arranged in a single column or row. In ML/DL, vectors often represent data points or features. A matrix, on the other hand, is a two-dimensional array of numbers that can be thought of as a collection of vectors or rows and columns. Matrices can be used to represent linear transformations, coefficients in linear models, or weights in neural networks.

2. Vector Operations:
   a) Addition and Subtraction: Two vectors of the same size can be added or subtracted element-wise. This operation is fundamental for data preprocessing and feature manipulation.
   b) Scalar Multiplication: Multiplying a vector by a scalar results in the same numbers multiplied by that scalar.
   c) Dot Product (Inner Product): Given two vectors, the dot product computes the sum of the products of corresponding elements. It is used to measure similarity between vectors and calculate the angle between them.

3. Matrix Operations:
   a) Addition and Subtraction: Two matrices of the same size can be added or subtracted element-wise. This operation is used to create new feature representations or combine multiple matrices.
   b) Transpose: The transpose of a matrix swaps its rows and columns. In ML/DL, transposition is often used for data representation changes or when dealing with symmetry in covariance matrices.
   c) Matrix Multiplication: Multiplying two matrices results in a new matrix formed by summing the products of corresponding elements in each row of the first matrix and each column of the second matrix. It's crucial for various ML/DL algorithms, such as neural networks and linear regression.

4. Linear Systems and Solving Linear Equations:
   A system of linear equations represents a set of equations where every equation is a linear combination of variables. In ML/DL, these systems often appear in the form of overdetermined or underdetermined systems that need to be solved for model coefficients. Techniques like Gauss-Jordan elimination, matrix inversion, and QR decomposition are used to find solutions.

5. Eigenvalues and Eigenvectors:
   For a square matrix A, the eigenvalues λ and corresponding eigenvectors x satisfy Ax = λx. In ML/DL, eigenvalues and eigenvectors provide important insights into the underlying structure of data and can be used to find principal components in Principal Component Analysis (PCA), or solve systems with ill-conditioned matrices using Singular Value Decomposition (SVD).

6. Determinants:
   The determinant of a square matrix is a scalar value that describes essential properties such as volume scaling and orientation preservation/reflection in linear transformations. In ML, it can be used to calculate the absolute value of the Jacobian for change-of-basis calculations.

7. Inverses:
   The inverse of a square matrix A, denoted as A⁻¹, is another square matrix that satisfies AA⁻¹ = A⁻¹A = I (the identity matrix). In ML/DL, the inverse of a matrix is used to solve linear systems Ax = b or find the weights in Bayesian inference.

8. Norms and Distances:
   The norms of vectors measure their magnitudes, while distances between vectors provide a measure of similarity. Commonly used norms include Euclidean, Manhattan, and Chebyshev distances, which are used for various tasks like clustering, dimensionality reduction, and similarity search.

   ---
### Questions from Hands on ML

   1. How would you define Machine Learning? 

   2. Can you name four types of problems where it shines?
   
   3. What is a labeled training set? 
   
   4. What are the two most common supervised tasks? 
   
   5. Can you name four common unsupervised tasks? 
   
   6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains? 
   
   7. What type of algorithm would you use to segment your customers into multiple groups? 
   
   8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem? 
   
   9. What is an online learning system? 
   
   10. What is out-of-core learning? 
   
   11. What type of learning algorithm relies on a similarity measure to make predictions? 
   
   12. What is the difference between a model parameter and a learning algorithm’s hyperparameter? 

   13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions? 
   
   14. Can you name four of the main challenges in Machine Learning? 
   
   15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions? 
   
   16. What is a test set, and why would you want to use it? 
   
   17. What is the purpose of a validation set? 
   
   18. What is the train-dev set, when do you need it, and how do you use it? 
   
   19. What can go wrong if you tune hyperparameters using the test set? 

