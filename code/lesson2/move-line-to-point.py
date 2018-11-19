#!/usr/bin/env python

import sys
import numpy as np

class State:
    def __init__(self, weights, bias):
        self._weights = np.array(weights, np.float)
        self._bias = bias

    def weights(self):
        return self._weights

    def bias(self):
        return self._bias

    def __str__(self):
        return "w1=%5.2f w2=%5.2f b=%6.2f" % (self._weights[0], self._weights[1], self._bias)

    def weighted_sum(self, x):
        return x.dot(self._weights) + self._bias

    def increment_weights(self, x, alpha=0.1):
        return State(self._weights + np.multiply(alpha, x), self._bias + alpha)

def move_line(x, weights, bias, iteration=1):
    x = np.array(x, float)
    current_state = State(weights, bias)

    output = current_state.weighted_sum(x)
    current_state = current_state.increment_weights(x)

    print("%-2d: %s output=%6.3f" % (iteration, str(current_state), output))

    if output >= 0:
        return iteration
    # Algorithm is guaranteed to converge
    return move_line(x, current_state.weights(), current_state.bias(), iteration + 1)

move_line((1, 1), (3, 4), -10)
