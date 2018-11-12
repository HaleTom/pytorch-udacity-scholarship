#!/usr/bin/env python

import sys
import numpy as np

class MoveLine:
    def __init__(self):
        self.x = np.array([1, 1], float)
        self.w = np.array([3, 4], float)
        self.b = -10

    def weighted_sum(self):
        return np.add(self.x.dot(self.w), self.b)

    def increment_weights(self, alpha=0.1):
        self.w += np.multiply(alpha, self.x)
        self.b += alpha

    def iterate(self):
        iteration=1
        while True:
            output = self.weighted_sum()
            print("%-2d: w1=%5.2f w2=%5.2f b=%6.2f output=%6.3f" %
                    (iteration, self.w[0], self.w[1], self.b, output))
            self.increment_weights()
            if output >= 0:
                break
            iteration += 1
        return iteration

MoveLine().iterate()
