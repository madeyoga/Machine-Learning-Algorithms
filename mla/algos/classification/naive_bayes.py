import numpy as np
from mla.base.statistic import std_dev

class GaussianNB:
    """
    Proba:
        P(A|B) = P(B|A).P(A)/P(B)
        P(A|B) = P(AnB)/P(B)
        P(A|B) = Count(AnB)/Count(B)
    Additive Smoothing:
        P(A|B) = (Count(AnB) + 1)/Count(B)+|{A class}|
    """
    def __init__(self):
        self.memo_ = {}
        self.separated_instances = {}

    def fit(self, X_train, y_train):
        dataset = np.append(X_train, y_train, axis=1)
        print(dataset)

        for i, X_train_i in enumerate(X_train):
            print(i, X_train_i , y_train[i])
            if not y_train[i][0] in self.separated_instances:
                self.separated_instances[y_train[i][0]] = []
                self.separated_instances[y_train[i][0]].append(X_train_i.tolist())
            else:
                self.separated_instances[y_train[i][0]].append(X_train_i.tolist())
        print(self.separated_instances)

        for key in self.separated_instances:
            instance = np.array(self.separated_instances[key])
            avg = np.mean(instance.astype(np.float), axis=0)
            stddev = std_dev(instance.astype(np.float))
            self.memo_[key] = (avg.tolist(), stddev.tolist())
        print(self.memo_)

        return

    def predict(self, X_test):
        return
