# Lesson 2

Sum of Squared Errors (SSE) ensures positive error and penalises larger errors:

$\displaystyle E = \frac{1}{2}\sum_{\mu} \sum_j \left[ y^{\mu}_j - \hat{y} ^{\mu}_j \right]^2 $

Where:
* $\mu$ are the datapoints
* $j$ are the output neurons

For intuition, consider a single layer network:

$ \displaystyle \hat{y}^{\mu}_j = f \left( \sum_i{ w_{ij} x^{\mu}_i }\right) $

Where $f$ is the activation funciton and the $i$s represent input nodes to the layer we are considering. The sum over $i$ is because a node's activation is the weighted sum of all input nodes.
The bias term has a weight of $1$.

Substituting into the previous formula, we see that the error depends solely on the weights:

$ \displaystyle E = \frac{1}{2}\sum_{\mu} \sum_j \left[ y^{\mu}_j - f \left( \sum_i{ w_{ij} x^{\mu}_i }\right) \right]^2 $

So the weights are the knobs that we turn to adjust the overall error (the inputs $x$ are fixed).


We use the Mean Squared Error (MSE) instead of the SSE to prevent the error being (approx) proportional to the number of datapoints, and needing to scale the learning rate.


## Gradient Descent

Algorithm for updating the weights with batch gradient descent:

1. Set the weight step to zero: $ \Delta w_i = 0 $
1. For each record in the training data:
  * Make a forward pass through the network, calculating the output $ \hat y = f(\sum_i w_i x_i) $
  * Calculate the error term for the output unit, $ \delta = (y - \hat y) * f'(\sum_i w_i x_i)$
  * Update the weight step $\Delta w_i = \Delta w_i + \delta x_i $
1. Update the weights $w_i = w_i + \eta \Delta w_i / m$

   where $\eta$ is the learning rate and mm is the number of records. Here we're averaging the weight steps to help reduce any large variations in the training data.
1. Repeat for $e$ epochs.

## Initialising weights

Initialise weights from a normal distribution with $\mu$ = $0$ and $\sigma =  1/ \sqrt n$ where $n$ is the number of input units. This keeps the input to the sigmoid low for increasing numbers of input units.

## Gradient Descent (without bias term)

```
   import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

# TODO: We haven't provided the sigmoid_prime function like we did in
#       the previous lesson to encourage you to come up with a more
#       efficient solution. If you need a hint, check out the comments
#       in solution.py from the previous lecture.

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Activation of the output unit
        #   Notice we multiply the inputs and the weights here
        #   rather than storing h as a separate variable
        output = sigmoid(np.dot(x, weights))

        # The error, the target minus the network output
        error = y - output

        # The error term
        #   Notice we calulate f'(h) here instead of defining a separate
        #   sigmoid_prime function. This just makes it faster because we
        #   can re-use the result of the sigmoid function stored in
        #   the output variable
        error_term = error * output * (1 - output)

        # The gradient descent step, the error times the gradient times the inputs
        del_w += error_term * x

    # Update the weights here. The learning rate times the
    # change in weights, divided by the number of records to average
    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```

## Weights matrix representation

Each *row* in the matrix will correspond to the weights leading *out* of a single input unit, and each *column* will correspond to the weights leading *in* to a single hidden unit.

![l2-multilayer-diagram-weights.png](l2-multilayer-diagram-weights.png)

However, I prefer to use:

![l2-inputs-matrix.png](l2-inputs-matrix.png)

Numpy converts lists and tuples to row vectors, with an unspecified 2nd dimension:

```
>>> a = np.array([1, 2, 3])
>>> a
array([1, 2, 3])
>>> a.shape
(3,)
```

Even transposing still returns a row vector:
```
>>> a.T
array([1, 2, 3])
>>> a.T.T
array([1, 2, 3])
```

To get a column vector, use `a[:, None]` (Note that `-1` does *not* work in place of `None`):
```
>>> a[:, None]
array([[1],
       [2],
       [3]])

>>> a = np.array([1, 2, 3])[:, np.newaxis]
>>> a
array([[1],
       [2],
       [3]])
```

Alternatively:

```
>>> np.array([1, 2, 3], ndmin=2).T
array([[1],
       [2],
       [3]])

>>> a=np.array([[1, 2, 3]]).T
>>> a
array([[1],
       [2],
       [3]])
```

## Gradient descent formulae

$ \displaystyle \frac{\delta E}{\delta w_i} = -(y - \hat y)f'(h)x$

$ W_{new} = W + \Delta W / m\ $  ` # Undo effect of summing each training example's gradient`

Where $\Delta W$ is the negative gradient times learning rate.

$ \displaystyle \Delta W = \eta (y - \hat y)f'(h)x$

### Vanishing gradients

Because we multiply by the derivative of the activation function, using sigmoid will mean that the gradient reduces to a **maximum** of 25% of the following layer's gradient.

### Backprop example code

In this course:

* `x_error` is the derivative of the output of the unit x, *after the activation function is applied*
* `x_error_term` is the derivative of the weighted sum term, or $z$ in Ng's courses.

Polished assignment submission of a trainable network:

Use `np.outer()` to calculate the $\Delta W$ matrix from the inputs and the layer's error term.


```
import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.output_nodes))

        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # Regression problem, no activation function

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        '''

        error = y - final_outputs  # Output layer error is the difference between desired target and actual output.
        output_error_term = error  # No multiply by derivative of sigmoid since regression problem
        hidden_error = self.weights_hidden_to_output.dot(output_error_term)  # element-wise
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        delta_weights_i_h += np.outer(X, hidden_error_term)
        delta_weights_h_o += np.outer(hidden_outputs, output_error_term)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weights_hidden_to_output += self.lr * delta_weights_h_o  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden  += self.lr * delta_weights_i_h  # update input-to-hidden weights with gradient descent step
```

## Multi-layer backprop

Real-world example with more than one training set record.

This may not be as good in the internals as the above example, but has some extra functionality.


```
import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
# features.shape == (360, 6)

last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = x.dot(weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(hidden_output.dot(weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output

        # TODO: Calculate error term for the output unit, x * (1-x) == derivative of sigmoid
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        # OET shape is ()    # scalar
        # WHO shape is (2,)  # defined above
        # With explicit shapes, .dot() arguments are (hidden, output) and (output, 1)
        hidden_error = np.dot(weights_hidden_output, output_error_term)

        # hidden_error.shape == (2,)

        # TODO: Calculate the error term for the hidden layer (the weighted sum or Ng's z term)
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden  += np.outer(x, hidden_error_term)

    # TODO: Update weights  (don't forget to division by n_records or number of samples)
    weights_input_hidden  += learnrate * del_w_input_hidden  / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```
