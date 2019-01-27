from pymla.model.base.distances import euclidean_distance
import operator as op
import numpy as np

def get_neighbors(y_train, X_train, test_sample, k):
    """Get smallest distance neighbors"""

    distances = []
    for i, train_sample in enumerate(X_train):
        distance = euclidean_distance(
            train_sample, test_sample
            )
        distances.append(
            (y_train[i], distance)
            )
    # return k smallest distances / neighbors
    distances = np.array(distances)
    neighbors = distances[distances[:, 1].argsort()[:k]]
    return neighbors

def vote_(neighbors, X_train):
    """Vote neighbors"""

    votes = {}
    for n in neighbors:
        if n[0][0] in votes:
            votes[n[0][0]] += 1
        else:
            votes[n[0][0]] = 1
    print(votes)
    return max(votes, key=votes.get)

class KNeighborsClassifier:
    def __init__(self, n_neighbors = 1):
        self.k = n_neighbors
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        """
        K-NN doesn't really have any learning algorithms (except KD-Tree etc..).
        It's just about getting the smallest distance data.
        """

        self.X_train = X_train
        self.y_train = y_train
        return

    def predict(self, X_test):
        """
        K-NN predicts by getting all closest k-neighbors
        and then vote for the 'modus'.
        """

        predicted_classes = []
        for test_sample in X_test:
            # get closest k-neighbors
            neighbors = get_neighbors(
                self.y_train,
                self.X_train,
                test_sample,
                self.k
                )
            # modus will be the predicted value
            predicted_y = vote_(neighbors, self.X_train)
            predicted_classes.append(predicted_y)
        return predicted_classes
