import numpy as np
import math
from base.activation_function import sigmoid

class MLPClassifier:
    """Multi-Layer Perceptron Classifier"""

    def __init__(self, activation="tanh"):
        self.num_examples = 0
        self.nn_input_dim = 2
        self.nn_output_dim = 2
        self.gd_learn_rate = 0.01 # gradient descent learning rate
        self.reg_lambda = 0.01 # regularization strength
        self.activation_function = self.get_activation(activation)
        self.nn_nodes = 2
        self.max_iterations=2000
        self.print_loss=False

    @staticmethod
    def get_activation(activation):
        if activation == 'tanh':
            return np.tanh
        else:
            return sigmoid

    def fit(self, X_train, y_train):
        """
        nn_nodes : Number of nodes in the hidden layer
        max_iterations : for gradient descent
        """

        model = {}
        # 1. pick training sample
        # 2. feed forward / forward propagation

        # 3. calculate error
        # 4. back propagation
        # 5. update weight
