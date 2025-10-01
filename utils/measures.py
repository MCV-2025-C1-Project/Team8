import numpy as np


def euclidean_dist(v1, v2):
    return np.sqrt(np.sum((v2-v1)**2))

def l1_dist(v1, v2):
    return np.sum(np.abs(v2-v1))

def x2_dist(v1, v2):
    return np.sum((v2-v1)**2/(v1+v2))

def hist_intersect(v1, v2):
    return np.sum(np.minimum(v1,v2))

def hellinger_kernel(v1, v2):
    return np.sum(np.sqrt(v1*v2))