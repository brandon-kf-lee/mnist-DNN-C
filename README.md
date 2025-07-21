# MNIST Neural Network in C

This source code seeks to replicate the (now removed) [MNIST For ML Beginners](https://web.archive.org/web/20180801165522/https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners) tutorial from the Tensorflow website using plain C code.

The task is to recognise digits, such as the ones below, as accurately as possible.

![MNIST digits](https://web.archive.org/web/20180801165522im_/https://www.tensorflow.org/versions/r1.1/images/MNIST.png)

This code modifies the original from Andrew Carter in order to add support for Compute-in-Memory, a new computing paradigm that can compute matrix multiplication within memory. 

By Brandon Lee: [email](mailto:brandon.kf.lee@gmail.com), [portfolio](https://brandon-kf-lee.github.io/)
- Code derived from [AndrewCarterUK ![(Twitter)](http://i.imgur.com/wWzX9uB.png)](https://twitter.com/AndrewCarterUK)
- [Original repo](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c)

## Contents

- [mnist_train.c](mnist_train.c): Glue code that trains the network
- [mnist_inference.c](mnist_inference.c): Glue code that uses pre-trained weights and reports algorithm accuracy
- [mnist_file.c](mnist_file.c): Retrieves images and labels from the MNIST dataset
- [neural_network.c](neural_network.c): Implements training and prediction routines for a simple neural network

## Usage

```sh
make
./mnist_train.c
./mnist_inference.c
```

## Description

The neural network implemented has one output layer and no hidden layers. Softmax activation is used, and this ensures that the output activations form a probability vector corresponding to each label. The cross entropy is used as a loss function.

The algorithm reaches an accuracy of around 90-92% over 1000 steps.

## Expected Output

```
Training neural network in 1000 steps...
Neural network saved as "mnist_network.bin"
Accuracy: 0.908
```