import numpy as np

def root_mean_squared_error(y1, y2):
    if len(y1) != len(y2):
        raise ValueError('different row sizes, {} & {} '.format(len(y1), len(y2)))
    y1 = np.array(y1)
    y2 = np.array(y2)
    rmse = np.sqrt(((y1 - y2) ** 2).sum(0) / len(y1))
    return rmse
        
