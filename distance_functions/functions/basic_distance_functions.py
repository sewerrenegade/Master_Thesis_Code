from numpy import dot
from numpy.linalg import norm
import numpy as np

def euclidean_distance_function(x, y=None):
    if y is None:
        assert len(x) == 2, "Input list must contain exactly two elements."
        out_dic = norm(x[0] - x[1])
    else:
        out_dic = norm(x - y)
    
    return out_dic

def cosine_similarity_function(x, y=None):
    if y is None:
        assert len(x) == 2, "Input list must contain exactly two elements."
        x, y = x[0], x[1]
    cos_sim = dot(x, y) / (norm(x) * norm(y))
    return cos_sim

def l1_distance_function(x, y=None):
    if y is None:
        assert len(x) == 2, "Input list must contain exactly two elements."
        x, y = x[0], x[1]
    l1_distance = np.sum(np.abs(x - y))
    return l1_distance

def linfinity_distance_function(x, y=None):
    if y is None:
        assert len(x) == 2, "Input list must contain exactly two elements."
        x, y = x[0], x[1]
    linf_distance = np.max(np.abs(x - y))
    return linf_distance