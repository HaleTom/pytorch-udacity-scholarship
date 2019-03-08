# Introduction to Neural Networks

## What is a NN?

A NN draws a line between red and blue data points.

When a linear model can separate points, it has a formula like:

<img src="/notes/pytorch-scholarship/tex/1c4973c62463edbb1e74014872ed26cc.svg?invert_in_darkmode&sanitize=true" align=middle width=128.51574614999998pt height=21.18721440000001pt/>

Points where the equation <img src="/notes/pytorch-scholarship/tex/bb0cb7aa9f90d9fd13946da9ffa97ed2.svg?invert_in_darkmode&sanitize=true" align=middle width=25.570741349999988pt height=21.18721440000001pt/> are accepted by convention. <img src="/notes/pytorch-scholarship/tex/1757afe2b054e59c6d5c465cf82bd885.svg?invert_in_darkmode&sanitize=true" align=middle width=25.570741349999988pt height=21.18721440000001pt/> is rejected.

Generally, a linear equation is:

<img src="/notes/pytorch-scholarship/tex/587d9779086724f044a15127cd59f2df.svg?invert_in_darkmode&sanitize=true" align=middle width=84.48607365pt height=22.831056599999986pt/>

where <img src="/notes/pytorch-scholarship/tex/f1bbfed182fa9659bb2ac9e5ea634635.svg?invert_in_darkmode&sanitize=true" align=middle width=98.10314084999999pt height=24.65753399999998pt/> and <img src="/notes/pytorch-scholarship/tex/0a203f5b135a9327d49ec3d66425ef78.svg?invert_in_darkmode&sanitize=true" align=middle width=84.94283159999999pt height=24.65753399999998pt/>

<img src="/notes/pytorch-scholarship/tex/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=22.465723500000017pt/> are the labels of the given data points.

Each point is of the form <img src="/notes/pytorch-scholarship/tex/f9c4ac8137c16d58619a9815768ed439.svg?invert_in_darkmode&sanitize=true" align=middle width=69.58530149999999pt height=24.65753399999998pt/>.

The purpose of the learning algorithm is to find a solution which has <img src="/notes/pytorch-scholarship/tex/6b950307e24405f963ef75ad1217dcb1.svg?invert_in_darkmode&sanitize=true" align=middle width=9.347490899999991pt height=22.831056599999986pt/> as close as possible to <img src="/notes/pytorch-scholarship/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>.

In 3D, the boundary will be a plane, in the form:

<img src="/notes/pytorch-scholarship/tex/587d9779086724f044a15127cd59f2df.svg?invert_in_darkmode&sanitize=true" align=middle width=84.48607365pt height=22.831056599999986pt/>

where <img src="/notes/pytorch-scholarship/tex/67db38f7e9dd08a2d2d6a9ac05acfccf.svg?invert_in_darkmode&sanitize=true" align=middle width=124.55199239999997pt height=24.65753399999998pt/> and <img src="/notes/pytorch-scholarship/tex/b0728cbf6fe023c5e8833e97fb75c50f.svg?invert_in_darkmode&sanitize=true" align=middle width=109.01816145pt height=24.65753399999998pt/>

with <img src="/notes/pytorch-scholarship/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> dimensions, the solution is a <img src="/notes/pytorch-scholarship/tex/efcf8d472ecdd2ea56d727b5746100e3.svg?invert_in_darkmode&sanitize=true" align=middle width=38.17727759999999pt height=21.18721440000001pt/>-dimensional hyperplane.

### Perceptron

![l2-perceptron](l2-perceptron.png)

The edges from input nodes <img src="/notes/pytorch-scholarship/tex/ae3bb9d3169eb1257957c2d604a4d775.svg?invert_in_darkmode&sanitize=true" align=middle width=82.85389529999999pt height=24.65753399999998pt/> are numbered with the values of <img src="/notes/pytorch-scholarship/tex/4b4518f1b7f0fb1347fa21506ebafb19.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/> and <img src="/notes/pytorch-scholarship/tex/f7eb0e840408d84a0c156d6efb611f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.32105549999999pt height=14.15524440000002pt/>, and the perceptron node itself is labelled with the bias term.

The bias could also be an edge label on an input node which is set to the constant <img src="/notes/pytorch-scholarship/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>.

![l2-perceptrion-linear-and-activation](l2-perceptrion-linear-and-activation.png)

There is the linear part and the non-linear or activation function (above it's the non-linear step function).

Some logical operations can be represented by perceptrons:
![l2-OR-perceptron](l2-OR-perceptron.png)

The AND perceptron can be changed to an OR by:
 * Decreasing <img src="/notes/pytorch-scholarship/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> (think the <img src="/notes/pytorch-scholarship/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>-intercept term in <img src="/notes/pytorch-scholarship/tex/3f0405965dd2f9e1dd62f6258e0a5294.svg?invert_in_darkmode&sanitize=true" align=middle width=81.54091274999998pt height=22.831056599999986pt/>) moves the line down
 * Since <img src="/notes/pytorch-scholarship/tex/52506fecc0860a807aa3051af9f1e266.svg?invert_in_darkmode&sanitize=true" align=middle width=159.6744963pt height=22.831056599999986pt/>, increasing the weights will require decreasing <img src="/notes/pytorch-scholarship/tex/abc574bb2f36d89e011d4a52627367cf.svg?invert_in_darkmode&sanitize=true" align=middle width=40.02286529999999pt height=14.15524440000002pt/> to keep the equality, which moves the line diagonally toward the origin.

NOT requires only a negative weight (which is applied in the non-0 case). In the 0 case, >= 0 implies true.

XOR requires a more complex solution because there is no straight line which can correctly separate the blue and red points.

![l2-XOR-logic](l2-XOR-logic.png)

For XOR, we have a neural network of 4 perceptrons: XOR = AND(OR, NOT(AND))

### Automating the decision boundary line equation

We want an algorithm to automatically learn the decision boundary line.

If a particular point is misclassified, we want the decision boundary line to move closer to that point, and eventually past it.

To move the line closer:

* to a false positive, *subtract* the point
* to a false negative, *add* the point

Use 1 as the value for the bias.

![l2-perceptron-move-line-to-point](l2-perceptron-move-line-to-point.png)

To dampen the movement, multiply by a learning rate.

![l2-perceptron-algorithm](l2-perceptron-algorithm.png)

### Non-linear regions

![l2-non-linear-regions](l2-non-linear-regions.png)

Non-linear regions require a more complex solution than a simple straight line (linear equation).

### Error functions

An error function tells us how far we are from the perfect solution. We "move" one step in the direction which reduces the error the most by changing the model's weights.

### Gradient Descent

Gradient descent is taking a step towards the steepest slope down the mountain. Local minima exist, and are talked about later, but still give a pretty good solution.

### Discrete vs Continuous error functions

The error function needs to be continuous so that it can be differentiated, as the derivative gives the slope, and we want to take a step down the steepest slope.

If the error function gave discrete values (eg the count of misclassifcations), it wouldn't be possible to know which direction to move the weights.

So we want the style on the right:

![l2-binary-vs-probability-scores](l2-binary-vs-probability-scores.png)

![l2-step-vs-sigmoid](l2-step-vs-sigmoid.png)

* When the sum is 0, the sigmoid output is <img src="/notes/pytorch-scholarship/tex/cde2d598001a947a6afd044a43d15629.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/>.
* Large positive numbers give values close to <img src="/notes/pytorch-scholarship/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>.
* Large negative numbers give values close to <img src="/notes/pytorch-scholarship/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>.

![l2-probability-of-prediction](l2-probability-of-prediction.png)

![l2-perceptron-step-vs-sigmoid](l2-perceptron-step-vs-sigmoid.png)

### Softmax

Softmax is used if there are more than two possible labels.  It ensures that the sum of all probabilities equals 1.

<img src="/notes/pytorch-scholarship/tex/3c38bf9376d371a9b9910d76f08b40ff.svg?invert_in_darkmode&sanitize=true" align=middle width=240.48757754999997pt height=50.15378279999997pt/>

```
def softmax(L):
    expL = np.exp(L)
    return np.divide (expL, expL.sum())
```

The exponential function ensures that all negative inputs map to a positive number.

With 2 classes, the output is the same as the sigmoid function.

### One-hot encoding

One-hot encoding ensures that there are no dependencies between the variables.

Each variable has its own numeric range, a probability between 0 and 1.

Eg, if `(duck, beaver, walrus)` were assigned values of `(1, 2, 3)` then does prediction of `2` mean `beaver` or .5 probability of `duck` and .5 of `walrus`?

It removes any concept that `beaver` is more `walrus`-y than `duck`, or that it is half way between the two.

### Maximum Likelihood - evaluating models

Maximum Likelihood allows for model selection.

We want to pick the model which gives the highest probability to the provided, correct labels.

![l2-maximimum-likelihood-example](l2-maximimum-likelihood-example.png)

Multiply the probabilities of the points getting the ground-truth label. The maximum likelihood is denoted: P(all).

(The required assumption for this to be valid is that all points are independent events)

The goal is to maximise P(all).

### Maximising probabilities

So, we want to move our model toward having a higher P(all).

If there were thousands of points, then the product would be very small (all probabilities are <img src="/notes/pytorch-scholarship/tex/d0d9fabe406e63c42d47cfb1433b78b6.svg?invert_in_darkmode&sanitize=true" align=middle width=25.570741349999988pt height=21.18721440000001pt/>).

Also, if any one probability were almost 0, it would bring the whole product down drastically, without knowing which it was.

So we avoid products.

Instead, we use <img src="/notes/pytorch-scholarship/tex/ea88241ebbb313d3abc36ace084515de.svg?invert_in_darkmode&sanitize=true" align=middle width=21.626772749999994pt height=22.831056599999986pt/> to turn products into sums as <img src="/notes/pytorch-scholarship/tex/610925506c18dd84b30e89987e340c48.svg?invert_in_darkmode&sanitize=true" align=middle width=176.733282pt height=24.65753399999998pt/>.

Since <img src="/notes/pytorch-scholarship/tex/e627e810866eace349d376c6696d114e.svg?invert_in_darkmode&sanitize=true" align=middle width=66.23670404999999pt height=24.65753399999998pt/>, we expect that the logarithms of values in the range <img src="/notes/pytorch-scholarship/tex/e88c070a4a52572ef1d5792a341c0900.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/> will be negative.

So to get positive numbers, we take the negative values of <img src="/notes/pytorch-scholarship/tex/226b61226dce4a485a9a1d80382bb7db.svg?invert_in_darkmode&sanitize=true" align=middle width=76.11327899999999pt height=24.65753399999998pt/>.

High probabilities will give values close to 0.

The sum of negative logs is called the *cross entropy*.

A bad model will have a high cross entropy.

![l2-cross-entropy](l2-cross-entropy.png)

The arrangement of points on the right is much more likely to happen, and has a low cross-entropy.

[LHS has more realistic probability values. RHS top-right point should be 0.7]

Points with higher <img src="/notes/pytorch-scholarship/tex/0361cb5c1623c2137200302920f25de0.svg?invert_in_darkmode&sanitize=true" align=middle width=60.03439199999999pt height=24.65753399999998pt/> scores are those which have the worst (least likely) predictions.

We can think of the <img src="/notes/pytorch-scholarship/tex/0361cb5c1623c2137200302920f25de0.svg?invert_in_darkmode&sanitize=true" align=middle width=60.03439199999999pt height=24.65753399999998pt/> value as the error of each point's classification.

Cross entropy tells us how likely it is that the events we observed occurred based on the probabilities of them happening.

The goal has changed: from maximising probability to minimising cross entropy.

Assume 3 doors have different probabilities of having a gift behind them:
![l2-gifts-behind-doors](l2-gifts-behind-doors.png)

The probability of the outcome as recorded by red circles is <img src="/notes/pytorch-scholarship/tex/dc8d575fa52da35eb77faac1ee169f50.svg?invert_in_darkmode&sanitize=true" align=middle width=162.55700339999999pt height=21.18721440000001pt/>

![l2-doors-and-probabilities](l2-doors-and-probabilities.png)
Above the 8 rows are the <img src="/notes/pytorch-scholarship/tex/a0d387340c219f798d8ad8a0fee3a5cf.svg?invert_in_darkmode&sanitize=true" align=middle width=14.771756999999988pt height=26.76175259999998pt/> possible outcomes. The probabilities column sums to <img src="/notes/pytorch-scholarship/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>.

Note the cross entropy is highest when the probability is lowest.

Cross-entropy is inversely proportional to the total probability of an outcome.

![l2-cross-entropy-formula](l2-cross-entropy-formula.png)

As <img src="/notes/pytorch-scholarship/tex/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode&sanitize=true" align=middle width=12.710331149999991pt height=14.15524440000002pt/> is either <img src="/notes/pytorch-scholarship/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> or <img src="/notes/pytorch-scholarship/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>, [either the first or second term of the sum is cancelled out](https://stats.stackexchange.com/a/287933/162527).

Note the bottom right notation.

```
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
```

### Multi-class cross entropy

So far, we've looked only at binary outcomes (gift or not).  Here for 3 possible outcomes:

![l2-multi-class-cross-entropy-example](l2-multi-class-cross-entropy-example.png)

![ls-multi-class-cross-entropy-formula](ls-multi-class-cross-entropy-formula.png)

In the bottom right formula, <img src="/notes/pytorch-scholarship/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> is the number of classes, <img src="/notes/pytorch-scholarship/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> the number of observations.

### Logistic regression cost function

![l2-error-function-example](l2-error-function-example.png)

By convention, we multiply by <img src="/notes/pytorch-scholarship/tex/d74399227ae7f06fb8b82d97e71fc538.svg?invert_in_darkmode&sanitize=true" align=middle width=11.664849899999997pt height=27.77565449999998pt/> to get the average value. (Also done if using the multi-class formula.)

Substituting <img src="/notes/pytorch-scholarship/tex/e83a22ed58285fe191996e66e2918230.svg?invert_in_darkmode&sanitize=true" align=middle width=107.68438559999998pt height=24.65753399999998pt/> we get:

![l2-error-function-2-classes](l2-error-function-2-classes.png)

A loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set, and may be regularised (see later).

### Gradient descent

![l2-gradient-descent-example](l2-gradient-descent-example.png)

The negative derivative tells us the direction of steepest slope to follow down to a lower cost.

![l2-gradient-descent-algorithm](l2-gradient-descent-algorithm.png)

The derivative of the cost / error function at a point <img src="/notes/pytorch-scholarship/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> with label <img src="/notes/pytorch-scholarship/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/> an prediction <img src="/notes/pytorch-scholarship/tex/6b950307e24405f963ef75ad1217dcb1.svg?invert_in_darkmode&sanitize=true" align=middle width=9.347490899999991pt height=22.831056599999986pt/> is:

<img src="/notes/pytorch-scholarship/tex/8859617bddaf61921865b7e0ffd3a09e.svg?invert_in_darkmode&sanitize=true" align=middle width=96.09652304999999pt height=45.072403200000004pt/>

<img src="/notes/pytorch-scholarship/tex/dbc4862d3442aced58ec009bf267fe10.svg?invert_in_darkmode&sanitize=true" align=middle width=74.98429125pt height=45.072403200000004pt/>

<img src="/notes/pytorch-scholarship/tex/9124633801f2cd43ecbebc304cdb51da.svg?invert_in_darkmode&sanitize=true" align=middle width=175.99500104999998pt height=24.65753399999998pt/>

The <img src="/notes/pytorch-scholarship/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> gives the derivative of the bias term.

* The gradient is a scalar times the coordinates of the point.
* The scalar is the difference between the label and prediction.
* The closer to the prediction, the smaller the gradient.

The updates of gradient descent are:

<img src="/notes/pytorch-scholarship/tex/93b09c28dd34d948abc61b32fdb8689b.svg?invert_in_darkmode&sanitize=true" align=middle width=98.70352799999999pt height=24.65753399999998pt/>

<img src="/notes/pytorch-scholarship/tex/32c6d5f56d75e2d4306635400a754903.svg?invert_in_darkmode&sanitize=true" align=middle width=64.28462369999998pt height=24.65753399999998pt/>

Note: Since we've taken the average of the errors, the term we are adding should be <img src="/notes/pytorch-scholarship/tex/d95141f3577d39e0f79051bf123fdb4e.svg?invert_in_darkmode&sanitize=true" align=middle width=36.085930649999995pt height=27.77565449999998pt/> instead of <img src="/notes/pytorch-scholarship/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>, but as <img src="/notes/pytorch-scholarship/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is a constant, then in order to simplify calculations, we'll just take <img src="/notes/pytorch-scholarship/tex/d95141f3577d39e0f79051bf123fdb4e.svg?invert_in_darkmode&sanitize=true" align=middle width=36.085930649999995pt height=27.77565449999998pt/> to be our learning rate, and abuse the notation by just calling it <img src="/notes/pytorch-scholarship/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>.

### Gradient Descent vs Perceptron Algorithm

In a perceptron the labels and predictions are either 1 or 0, then the difference is either 0, 1 or -1. In the 0 case, there is no update.

Hence these two are equivalent:

![l2-perceptron-vs-grad-desc](l2-perceptron-vs-grad-desc.png)

Except on the left, any value in the range <img src="/notes/pytorch-scholarship/tex/acf5ce819219b95070be2dbeb8a671e9.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/> is possible.

![l2-gradient-descent-intuition](l2-gradient-descent-intuition.png)

The "go further away" is because a correct point's classification wants to be closer to 1 if it is already correct.

```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias
```
### Non-linear models

Neural networks allow for a complex, non-linear boundary.

To combine two perceptrons regions, add their probability outputs. This will give numbers in a range greater than 1, so use the sigmoid function to reduce the values to the range <img src="/notes/pytorch-scholarship/tex/acf5ce819219b95070be2dbeb8a671e9.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/>.

![l2-combine-two-regions](l2-combine-two-regions.png)

Weights and biases can also be added:

![l2-combine-perceptrons.png](l2-combine-perceptrons.png)

Thus a neural network uses perceptrons as inputs to perceptrons.

![l2-combining-3-regions](l2-combining-3-regions.png)

![l2-3-input-variables](l2-3-input-variables.png)

With many input variables, there is a high-dimensional space, which is split with a highly non-linear boundary if there are many hidden layers in the network.

Notation: <img src="/notes/pytorch-scholarship/tex/001a280007c321b3d36d23ee80338d63.svg?invert_in_darkmode&sanitize=true" align=middle width=40.838382749999994pt height=34.337843099999986pt/> is the <img src="/notes/pytorch-scholarship/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/>-th layer matrix of weights from the <img src="/notes/pytorch-scholarship/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>th input of the previous node to the <img src="/notes/pytorch-scholarship/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>th neuron.

![l2-backprop-boundary-movement](l2-backprop-boundary-movement.png)

In the above, we want to reduce the weighting given to the top linear model, as it it causing the output to be incorrect.

Also we want the linear model boundaries to move in the directions shown.

![l2-grad-of-error-function](l2-grad-of-error-function.png)

![l2-chain-rule](l2-chain-rule.png)

The derivative of the sigmoid function is:

<img src="/notes/pytorch-scholarship/tex/2281792d9014975e6c0d2c06b6418601.svg?invert_in_darkmode&sanitize=true" align=middle width=141.9348282pt height=24.7161288pt/>

### Overfitting and underfitting

Underfitting is using a too-simple solution - it will make too many errors. A.k.a. *error due to bias*.

Overfitting is an over-complex too-specific solution which fits the training data extremely well but will not generalise to the test data. A.k.a. *high variance*.

![l2-over-under-fitting](l2-over-under-fitting.png)

![l2-over-under-fitting2](l2-over-under-fitting2.png)

Because it it near to impossible to get the complexity of architecture just right, we err on the side of an overly complex model and then reduce overfitting.

### Early stopping

![l2-train-vs-test-error](l2-train-vs-test-error.png)

Do gradient descent only until the test error starts to rise.

### Regularisation

![l2-sigmoid-with-high-coefficients](l2-sigmoid-with-high-coefficients.png)

When classifying only to points, the model on the right with larger coefficients gives a better result as the output values are closer to 0 and 1.

However, the gradients will generally be very close to <img src="/notes/pytorch-scholarship/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>, except in a very small region.

The LHS model will be more effective for gradient descent.

The RHS model will generate large errors because it is overly confident.

![l2-err-func-regularised](l2-err-func-regularised.png)

The top formula (absolute values) is called L1 Regularisation.

The one with the squared weights is called L2 Regularisation.

L1 is sparse - has small weights tend to go to zero. If we have too many features, L1 can will set some weights to 0 so that some are unused.

L2 generally gives better results for training models.  It prefers small homogeneous weights.

Given the choice between weights <img src="/notes/pytorch-scholarship/tex/1e5ba49ae6981862f61b4d510dcf29af.svg?invert_in_darkmode&sanitize=true" align=middle width=36.52973609999999pt height=24.65753399999998pt/> and <img src="/notes/pytorch-scholarship/tex/423f5d8f8e80c6d51ffd3bca0bea1c54.svg?invert_in_darkmode&sanitize=true" align=middle width=62.10060284999999pt height=24.65753399999998pt/>, L2 will prefer the latter since <img src="/notes/pytorch-scholarship/tex/c761d43998cd1bbddadec5a0723e7c89.svg?invert_in_darkmode&sanitize=true" align=middle width=81.41536919999999pt height=26.76175259999998pt/> while <img src="/notes/pytorch-scholarship/tex/24b22deb33a9a9fae33f6afb3ed3ef0c.svg?invert_in_darkmode&sanitize=true" align=middle width=119.77167014999998pt height=26.76175259999998pt/> is smaller.

### Dropout

Dropout ensures that there is less specialistion in the neurons. Large weights will dominate the network, meaning the other parts don't really get trained.

Dropout randomly turns off parts of the network ensuring a more even contribution of neurons to the final output.

A hyperparameter is the probability that each node will get dropped.

### Local Minima and Random Restart

Gradient descent can have us stop in a local minima. 

Random Restart starts gradient descent again from another random location, increasing the likelihood that a better minima is found.

### Vanishing Gradient

The sigmoid function's gradient becomes very shallow at small distances from <img src="/notes/pytorch-scholarship/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>.

The derivatives tell us in which direction to move, and determine the movement amount, so we don't want them to be too small.

Because the chain rule is used to calculate the derivative of a first layer weight, this product of derivatives can get smaller and smaller as a network gets deeper, making our descent steps too small.

Tanh is a scaled sigmoid, giving outputs between [-1, 1], instead of [0, 1] and with a steeper slope, between [0, 1] instead of [0, 0.25].

[Why is tanh almost always better than sigmoid as an activation function?](https://stats.stackexchange.com/q/330559/162527)

ReLU returns <img src="/notes/pytorch-scholarship/tex/3426369de3b1497efeb8c94526be2ae0.svg?invert_in_darkmode&sanitize=true" align=middle width=70.22275589999998pt height=24.65753399999998pt/>.  It can improve training signivicantly without sacrificing much accuracy. The gradient is either 1 or 0.

ReLU or Tanh will give larger derivatives, making the chain rule calculated derivatives less small, giving a faster gradient descent.

If ReLU is used, the final layer will still need to be sigmoid or softmax to get probabilities between 0 and 1.

Keeping the final layer as a ReLU helps build regression models which predict their value (used in RNNs).

### Batch vs Stochastic Gradient Descent

Instead of using all the input data before taking a gradient descent step, we can use a single point or mini-batches of input to speed the iteration process with slightly less accurate steps.

### Learning rate decay

What learning rate to use is a non-trivial research question.

A high learning rate can actually bounce out of a local minima. A too low rate will take too long to converge.

If the model isn't working (error is not decreasing), decrease the learning rate.

An optimisation is to decrease learning rates are the model gets closer to a solution.

### Momentum

Momentum uses inertia to bounce out of a local minima based on the speed that we've built up already going downhill.

![l2-momentum](l2-momentum.png)

With a <img src="/notes/pytorch-scholarship/tex/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode&sanitize=true" align=middle width=10.16555099999999pt height=22.831056599999986pt/> or momentum hyperparameter between 0 and 1, we can have the last step have a lot of influence on the momentum, and a previous step have less by raising <img src="/notes/pytorch-scholarship/tex/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode&sanitize=true" align=middle width=10.16555099999999pt height=22.831056599999986pt/> to a power based on the age of the step.
