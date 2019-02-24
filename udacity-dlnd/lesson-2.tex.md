# Lesson 2

Sum of Squared Errors (SSE) ensures positive error and penalises larger errors:

$\displaystyle E = \frac{1}{2}\sum_{\mu} \sum_j \left[ y^{\mu}_j - \hat{y} ^{\mu}_j \right]^2 $

Where:
* $\mu$ are the datapoints
* $j$ are the output neurons

For intuition, consider a single layer network:

$ \displaystyle \hat{y}^{\mu}_j = f \left( \sum_i{ w_{ij} x^{\mu}_i }\right) $

Where $i$ represents an input node.

Substituting into the previous formula, we see that the error depends solely on the weights:

$ \displaystyle E = \frac{1}{2}\sum_{\mu} \sum_j \left[ y^{\mu}_j - f \left( \sum_i{ w_{ij} x^{\mu}_i }\right) \right]^2 $

