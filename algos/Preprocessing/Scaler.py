from mla.base.base_preprocessing import *

class MinMaxScaler(BasePreprocessing):
    
    def transform(self, array : np.array):
        max_val = np.max(array)
        min_val = np.min(array)
        top = array - min_val
        bott = max_val - min_val
        _np_coef = top / bott
        return _np_coef
    
