import numpy as np

def root_mean_squared_error(y1, y2):
    if len(y1) != len(y2):
        raise ValueError('different row sizes, {} & {} '.format(len(y1), len(y2)))
    y1 = np.array(y1)
    y2 = np.array(y2)
    rmse = np.sqrt(((y1 - y2) ** 2).sum(0) / len(y1))
    return rmse

def r2_score(y1, y2):
    if len(y1) != len(y2):
        raise ValueError('different row sizes, {} & {} '.format(len(y1), len(y2)))
    y1 = np.array(y1)
    y2 = np.array(y2) ## predicted value
    mean_y = np.mean(y1)
    ss_t = ((y1 - mean_y) ** 2).sum(0)
    ss_r = ((y1 - y2) ** 2).sum(0)
    
    return 1 - (ss_r/ss_t)
