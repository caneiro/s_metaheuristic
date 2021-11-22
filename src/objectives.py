
import numpy as np
from itertools import combinations

def cost_function(s, port):
    _, r_mean, r_std, cor_mx = port
    X, Z = s
    Z1 = np.where(Z==1)[0]
    if len(Z1) == 1:

        obj = r_mean[Z1][0]
    else:
        obj = []
        for i, j in combinations(Z1, 2):
            obj_ = cor_mx[i, j] * X[i] * X[j] #* r_std[i] * r_std[j]
            obj.append(obj_)
        obj = np.array(obj)
        obj = np.sum(obj)
    return obj

def cost_function_vectorized(S, cor_mx, r_std):
    
    return None

def augmented_cost_function(s, port, penalties, w):
    _, r_mean, r_std, cor_mx = port
    X, Z = s
    Z1 = np.where(Z==1)[0]
    if len(Z1) == 1:
        obj = r_mean[Z1][0]
    else:
        obj = []
        for i, j in combinations(Z1, 2):
            obj_ = cor_mx[i, j] * X[i] * X[j] #* r_std[i] * r_std[j]
            obj.append(obj_)
        obj = np.array(obj)
        obj = np.sum(obj)

    if penalties is not None:
        aug_cost = np.sum(w * penalties[Z1])
        obj = obj + aug_cost
    
    return obj
