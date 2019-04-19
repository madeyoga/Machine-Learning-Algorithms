from pymla.preprocessing.base import BasePreprocessing
from pymla.preprocessing.base import np
from pymla.model.base.statistic import std_dev, mean

class MinMaxScaler(BasePreprocessing):
    """
    MinMax-Normalization.
    v' = (v - min) * (new_max - new_min) / (max - min) + new_min.
    
    currently can only scale value, 0 to 1.
    """
    def transform(self, array):
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
    
    def transform(self, array):
        array = np.array(array)
        scaled_values = (array - mean(array)) / std_dev(array)
        return scaled_values

class DecimalScaler(BasePreprocessing):
    """
    Decimal Scaler.
    v' = v / 10^j
    """

    def transform(self, array, j : int):
        array = np.array(array)
        scaled_values = array / 10 ** j
        return scaled_values
