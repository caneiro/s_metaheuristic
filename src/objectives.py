
import numpy as np
from itertools import combinations

def cost_function(s, cov_mx, penalties=None, lambda_=0.001):
    X, Z = s
    Z1 = np.where(Z==1)[0]
    if len(Z1) == 1:
        obj = X[Z1]
    else:
        i, j = zip(*list(combinations(Z1, 2)))
        i = np.array(i)
        j = np.array(j)
        cov_ = cov_mx[i, j]

        obj = np.sum(cov_ * X[i] * X[j])

    if penalties is not None:
        aug_cost = np.sum(lambda_ * penalties[Z1])
        obj = obj + aug_cost
    return round(obj, 6)
