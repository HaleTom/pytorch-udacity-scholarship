# Lesson 2

Sum of Squared Errors (SSE) ensures positive error and penalises larger errors:

<img src="/notes/tex/fb2c541ebb8cdd36232c9eb64ae3b40d.svg?invert_in_darkmode&sanitize=true" align=middle width=178.14223679999998pt height=43.42856099999997pt/>

Where:
* <img src="/notes/tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> are the datapoints
* <img src="/notes/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> are the output neurons

For intuition, consider a single layer network:

<img src="/notes/tex/19bbf75ee02655fc74405b6f2ada991f.svg?invert_in_darkmode&sanitize=true" align=middle width=146.00566694999998pt height=57.53473439999999pt/>

Where <img src="/notes/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the activation funciton and the <img src="/notes/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>s represent input nodes to the layer we are considering. The sum over <img src="/notes/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> is because a node's activation is the weighted sum of all input nodes.
The bias term has a weight of <img src="/notes/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>.

Substituting into the previous formula, we see that the error depends solely on the weights:

<img src="/notes/tex/0311e753bd1ccd1a54a30faa43146138.svg?invert_in_darkmode&sanitize=true" align=middle width=272.7817785pt height=64.23797490000003pt/>

So the weights are the knobs that we turn to adjust the overall error (the inputs <img src="/notes/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> are fixed).


We use the Mean Squared Error (MSE) instead of the SSE to prevent the error being (approx) proportional to the number of datapoints, and needing to scale the learning rate.


## Gradient Descent

Algorithm for updating the weights with batch gradient descent:

1. Set the weight step to zero: <img src="/notes/tex/15a47aead42a6c4d9d5e47642f6fdcd6.svg?invert_in_darkmode&sanitize=true" align=middle width=61.076817449999986pt height=22.465723500000017pt/>
1. For each record in the training data:
  * Make a forward pass through the network, calculating the output <img src="/notes/tex/14f809905520fcbdb8f731562c59612e.svg?invert_in_darkmode&sanitize=true" align=middle width=110.84287334999999pt height=24.657735299999988pt/>
  * Calculate the error term for the output unit, <img src="/notes/tex/a8f0eba78f962aa1c9b56a65ce6bdd8d.svg?invert_in_darkmode&sanitize=true" align=middle width=180.43361819999998pt height=24.7161288pt/>
  * Update the weight step <img src="/notes/tex/431f64af2e152747c19c99ebdb6d3bfc.svg?invert_in_darkmode&sanitize=true" align=middle width=125.86273424999999pt height=22.831056599999986pt/>
1. Update the weights <img src="/notes/tex/fff7595a58506a199732745896782ab9.svg?invert_in_darkmode&sanitize=true" align=middle width=138.83565464999998pt height=24.65753399999998pt/>

   where $\eta$ is the learning rate and mm is the number of records. Here we're averaging the weight steps to help reduce any large variations in the training data.
1. Repeat for <img src="/notes/tex/8cd34385ed61aca950a6b06d09fb50ac.svg?invert_in_darkmode&sanitize=true" align=middle width=7.654137149999991pt height=14.15524440000002pt/> epochs.

## Initialising weights

Initialise weights from a normal distribution with <img src="/notes/tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> = <img src="/notes/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> and <img src="/notes/tex/00ad64261eafb079393e48a756021bcf.svg?invert_in_darkmode&sanitize=true" align=middle width=71.90448045pt height=24.995338500000003pt/> where <img src="/notes/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the number of input units. This keeps the input to the sigmoid low for increasing numbers of input units.

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

<img src="/notes/tex/a8e4f1f11f38363f1363ff01fd5124f2.svg?invert_in_darkmode&sanitize=true" align=middle width=158.10086819999998pt height=45.072403200000004pt/>

<img src="/notes/tex/6bf6c6c51c481e7b931ca353a5cbd468.svg?invert_in_darkmode&sanitize=true" align=middle width=152.67906884999996pt height=24.65753399999998pt/>  ` # Undo effect of summing each training example's gradient`

Where <img src="/notes/tex/8d9b176fd459e329c57782488a188936.svg?invert_in_darkmode&sanitize=true" align=middle width=31.50693314999999pt height=22.465723500000017pt/> is the negative gradient times learning rate.

<img src="/notes/tex/9c7a4a1acca03a485fb788d3e94a99e5.svg?invert_in_darkmode&sanitize=true" align=middle width=158.43235155pt height=26.359964399999996pt/>

### Vanishing gradients

Because we multiply by the derivative of the activation function, using sigmoid will mean that the gradient reduces to a **maximum** of 25% of the following layer's gradient.

### Backprop example code

Simple example: only one training set recored, forward and backward pass:

```
import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = target - output

# TODO: Calculate error term for output layer
output_error_term = error * output * (1 - output)

# TODO: Calculate error term for hidden layer

# OET shape is ()    # scalar
# WHO shape is (2,)  # defined above
hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
                    hidden_layer_output * (1 - hidden_layer_output)

# HET shape is (2,)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * np.outer(x, hidden_error_term)

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)

# delta_w_i_h.shape == (3, 2)
```

## Multi-layer backprop

Real-world example with more than one training set record.

Use `np.outer()` to calculate the <img src="/notes/tex/8d9b176fd459e329c57782488a188936.svg?invert_in_darkmode&sanitize=true" align=middle width=31.50693314999999pt height=22.465723500000017pt/> matrix from the inputs and the layer's error term:


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

        # TODO: Calculate error term for the output unit
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # hidden_error_term.shape == (2,)

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

## fdsfas
fdsaf

