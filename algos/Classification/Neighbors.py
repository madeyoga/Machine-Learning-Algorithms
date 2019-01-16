from mla.base.distances import euclidean_distance
import operator as op

def get_neighbors(y_train, X_train, test_sample, k):
    # get distances
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
        self.X_train = X_train
        self.y_train = y_train
        return

    def predict(self, X_test):
        predicted_classes = []
        for test_sample in X_test:
            neighbors = get_neighbors(
                self.y_train,
                self.X_train,
                test_sample,
                self.k
                )
            predicted_y = vote_(neighbors, self.X_train)
            predicted_classes.append(predicted_y)
        return predicted_classes
