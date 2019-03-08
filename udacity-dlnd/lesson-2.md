# Lesson 2

Sum of Squared Errors (SSE) ensures positive error and penalises larger errors:

<img src="/udacity-dlnd/tex/fb2c541ebb8cdd36232c9eb64ae3b40d.svg?invert_in_darkmode&sanitize=true" align=middle width=178.14223679999998pt height=43.42856099999997pt/>

Where:
* <img src="/udacity-dlnd/tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> are the datapoints
* <img src="/udacity-dlnd/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> are the output neurons

For intuition, consider a single layer network:

<img src="/udacity-dlnd/tex/19bbf75ee02655fc74405b6f2ada991f.svg?invert_in_darkmode&sanitize=true" align=middle width=146.00566694999998pt height=57.53473439999999pt/>

Where <img src="/udacity-dlnd/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> represents an input node.

Substituting into the previous formula, we see that the error depends solely on the weights:

<img src="/udacity-dlnd/tex/0311e753bd1ccd1a54a30faa43146138.svg?invert_in_darkmode&sanitize=true" align=middle width=272.7817785pt height=64.23797490000003pt/>

