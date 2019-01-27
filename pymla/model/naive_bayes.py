import numpy as np
import math
from pymla.model.base.statistic import std_dev

def gaussian_proba(xi, avg, variance):
    # print((2*math.pi*variance)**0.5)
    # print(math.exp(-((xi-avg)**2)/(2*variance)))
    return 1/((2*math.pi*variance)**0.5) * math.exp(-((xi-avg)**2)/(2*variance))

class GaussianNB:
    """
    Proba:
        P(A|B) = P(B|A).P(A)/P(B)
        P(A|B) = P(AnB)/P(B)
        P(A|B) = Count(AnB)/Count(B)
    Additive Smoothing:
        P(A|B) = (Count(AnB) + 1)/Count(B)+|{A class}|
    GaussianProba:
        P(xi|ck) = 1/((2.phi.variance)**0.5) * e ** (-((xi-meank)**2)/2.variance)
    """
    def __init__(self):
        self.memo_ = {}
        self.separated_instances = {}
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        self.y_train = y_train
        for i, X_train_i in enumerate(X_train):
            print(i, X_train_i , y_train[i])
            if not y_train[i][0] in self.separated_instances:
                self.separated_instances[y_train[i][0]] = []
                self.separated_instances[y_train[i][0]].append(X_train_i.tolist())
            else:
                self.separated_instances[y_train[i][0]].append(X_train_i.tolist())
        for key in self.separated_instances:
            instance = np.array(self.separated_instances[key])
            avg = np.mean(instance.astype(np.float), axis=0)
            stddev = std_dev(instance.astype(np.float))
            self.memo_[key] = (avg.tolist(), stddev.tolist())
            self.memo_['proba_' + key] = np.count_nonzero(y_train==key)/y_train.shape[0]
        print(self.memo_)
        return

    def predict(self, X_test):
        predicted_y = []
        for x_sample in X_test:
            current_state_memo = {}
            for key in self.separated_instances:
                for i, xi in enumerate(x_sample):
                    gp =  gaussian_proba(
                        xi=xi,
                        avg=self.memo_[key][0][i],
                        variance=self.memo_[key][1][i]
                    )
                    if not key in current_state_memo:
                        current_state_memo[key] = self.memo_['proba_' + key]
                        current_state_memo[key] *= gp
                        continue
                    current_state_memo[key] *= gp
            print(current_state_memo)
            predicted_y.append(
                max(
                    current_state_memo,
                    key=current_state_memo.get
                    )
            )
        return predicted_y
