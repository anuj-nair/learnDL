# Deep Learning
Important point in Deep learning

## Activation Functions
* The artificial neuron does the weighted sums calculation
* Activation function determines if a neuron fires
### Step
* If the output is above a certain values, neuron has fired
* If the output is less than the threshold, then it has not fired
### Linear
* No matter how many layers--if all of them are linear, the final activation of last layer is linear
### Sigmoid
* Output of activation function between 0 and 1
* Vanishing gradient problem--near boundaries the network doesn't learn quickly
``` 
 sigma(x) = 1/(1 + e^(-x)) 
```
### Tanh
* Output of activation function between -1,1
* Similar problems to sigmoid
```
tanh(x) = (2/(1 + e^(-2x))) - 1
```
or
```
tanh(x) = 2sigma(2x) - 1
```
### ReLU (Rectified Linear Unit)
```
f(x) = 0 for x < 0
f(x) = x for x >=0

```
## Keras Compile
compile(self,optimizer,loss,metrics=['accuracy'])

### Optimizer
An algorithm that, given a set of parameters, returns one with a smaller loss function

#### SGD
* Stochastic gradient descent optimizer
* Includes support for momentum, learning rate decay and Nesterov momentum

#### RMSprop
* A good choice for recurrent neural network

#### Adam
* An algorithm for first-order gradient-based optimization of stochastic objective functions

### Loss
The objective function for measuring the accuracy of performance error of a neural network

#### mean_squared_error
Regression problem

#### categorical_crossentropy 
When your target has multiple classes

#### binary_crossentropy
When your target has two classes


### Metrics
The list of metrics
Its mostly gonna be `accuracy`


## Convolutional Neural Network

### Zero Padding

```
((N - F + 2p) / S) + 1
```
F: Size of filter
S: Stride
N: Size of image
P: Amount of padding

### Pooling
* Form of nonlinear downsampling
* Reduces chances of overfitting
* As the height and width decrease, you want to increase the number kernels

#### Max Pooling
* Convolutions "Light up" when they detect a particular feature in a region of an image
* When downsampling, it makes sense to preserve the parts that were most activated

### Image Augmentation
* Taking images in training data set and manipulating them, which means		
	*	There are more images for our model to train on
	*	Image manipulations make our model more robust



### Dropout
* Randomly kill each neuron in layer of a training set with probability p
* Purpose: to prevent over-fitting; only done on training data


