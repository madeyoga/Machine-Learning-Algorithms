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

def accuracy_score(y1, y2):
    y1 = np.array(y1)
    y2 = np.array(y2)
    if y1.shape != y2.shape:
        raise ValueError('different shape, {} & {} '.format(len(y1), len(y2)))
    count_true_pred = 0
    for i, data in enumerate(y1):
        if data == y2[i]:
            count_true_pred += 1
    return count_true_pred / y1.shape[0] * 100
        
    
                                                                
