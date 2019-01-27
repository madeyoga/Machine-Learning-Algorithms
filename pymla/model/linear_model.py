from pymla.model.base.estimator import BaseRegression
import numpy as np

class LinearRegression(BaseRegression):
    def fit(self, X : np.array, y : np.array):
        X = np.array(X)
        y = np.array(y)
        
        mean_x = np.mean(X)
        mean_y = np.mean(y)
        
        numerator = 0
        denominator = 0

        numerator = ((X - mean_x) * (y - mean_y)).sum(0)
        denominator = ((X - mean_x) ** 2).sum(0)
        
        self.coef = numerator / denominator
        self.bias = mean_y - self.coef * mean_x

    def predict(self, X_test : np.array):
        x = np.array(X_test)
        return self.bias + self.coef * x 
        
            
