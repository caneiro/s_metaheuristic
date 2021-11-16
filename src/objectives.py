
import numpy as np
from itertools import combinations

def cost_function(s, port):
    _, _, r_std, cor_mx = port
    X, Z = s
    Z1 = np.where(Z==1)[0]
    if len(Z1) == 1:
        obj = r_std[Z1]
    else:
        i, j = zip(*list(combinations(Z1, 2)))
        i = np.array(i)
        j = np.array(j)
        cor_ = cor_mx[i, j]
        obj = np.sum(cor_ * X[i] * X[j]) # * r_std[i] * r_std[j]

    return obj

def cost_function_vectorized(S, cor_mx, r_std):
    
    return None

def augmented_cost_function(s, port, penalties=None, w=0.001):
    _, _, r_std, cor_mx = port
    X, Z = s
    Z1 = np.where(Z==1)[0]
    if len(Z1) == 1:
        obj = r_std[Z1]
    else:
        i, j = zip(*list(combinations(Z1, 2)))
        i = np.array(i)
        j = np.array(j)
        cor_ = cor_mx[i, j]
        obj = np.sum(cor_ * X[i] * X[j]) # * r_std[i] * r_std[j]

    if penalties is not None:
        aug_cost = np.sum(w * penalties[Z1])
        obj_aug = obj + aug_cost

    return obj
