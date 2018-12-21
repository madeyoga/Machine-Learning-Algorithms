import numpy as np

class BaseRegression:
    def __init__(self):
        self.coef = -1
        self.bias = -1
        self.x = None
        self.y = None
