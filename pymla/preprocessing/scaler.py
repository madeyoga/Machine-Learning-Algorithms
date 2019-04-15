from pymla.preprocessing.base import BasePreprocessing
from pymla.preprocessing.base import np

class MinMaxScaler(BasePreprocessing):
    """
    MinMax-Normalization.
    v' = (v - min) * (new_max - new_min) / (max - min) + new_min.
    
    currently can only scale value, 0 to 1.
    """
    def transform(self, array : np.array):
        max_val = np.max(array)
        min_val = np.min(array)
        top = array - min_val
        bott = max_val - min_val
        scaled_values = top / bott
        return scaled_values
    
class ZScoreScaler(BasePreprocessing):
    """
    Z-Score Normalization.
    v' = (v - mean) / std_dev.
    """
    
    def transform(self, array : np.array):
        return

class DecimalScaler(BasePreprocessing):
    """
    Decimal Scaler.
    v' = v / 10^j
    """

    def transform(self, array : np.array):
        return
