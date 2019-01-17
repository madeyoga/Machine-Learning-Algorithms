from mla.base.base_preprocessing import BasePreprocessing
from mla.base.base_preprocessing import np

class MinMaxScaler(BasePreprocessing):
    
    def transform(self, array : np.array):
        max_val = np.max(array)
        min_val = np.min(array)
        top = array - min_val
        bott = max_val - min_val
        scaled_values = top / bott
        return scaled_values
    
