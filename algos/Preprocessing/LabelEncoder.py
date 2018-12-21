from ...base.base_preprocessing import *

class LabelEncoder(BasePreprocessing):
    def fit(self, array):
        value_id = 1
        for data in array:
            if not data in self.memo_:
                self.memo_[data] = value_id
                value_id += 1

    def transform(self, array):
        for i in range(len(array)):
            if not array[i] in self.memo_:
                return False
            array[i] = self.memo_[array[i]]
        return np.array(array).reshape(-1, 1)

    def fit_transform(self, array):
        self.fit(array)
        return self.transform(array)
