import numpy as np

class BasePreprocessing:
    def __init__(self):
        self.memo_ = {}

    def fit(self, array):
        return 0

    def transform(self, array):
        return 0

    def fit_transform(self, array):
        return 0
