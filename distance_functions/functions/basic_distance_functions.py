from numpy import dot
from numpy.linalg import norm
import numpy as np

class EuclideanDistance:
    def __init__(self, name="Euclidean Distance"):
        self.name = name

    def __call__(self, x, y=None):
        if y is None:
            assert len(x) == 2, "Input list must contain exactly two elements."
            out_dic = norm(x[0] - x[1])
        else:
            out_dic = norm(x - y)
        return out_dic

    def get_settings(self):
        return {}

class CosineSimilarity:
    def __init__(self, name="Cosine Similarity"):
        self.name = name

    def __call__(self, x, y=None):
        if y is None:
            assert len(x) == 2, "Input list must contain exactly two elements."
            x, y = x[0], x[1]
        cos_sim = dot(x, y) / (norm(x) * norm(y))
        return cos_sim

    def get_settings(self):
        return {}

class L1Distance:
    def __init__(self, name="L1 Distance"):
        self.name = name

    def __call__(self, x, y=None):
        if y is None:
            assert len(x) == 2, "Input list must contain exactly two elements."
            x, y = x[0], x[1]
        l1_distance = np.sum(np.abs(x - y))
        return l1_distance

    def get_settings(self):
        return {}

class LinfinityDistance:
    def __init__(self, name="L-infinity Distance"):
        self.name = name

    def __call__(self, x, y=None):
        if y is None:
            assert len(x) == 2, "Input list must contain exactly two elements."
            x, y = x[0], x[1]
        linf_distance = np.max(np.abs(x - y))
        return linf_distance

    def get_settings(self):
        return {}

class LpDistance:
    def __init__(self, p, name="Lp Distance"):
        self.name = name
        self.p = p

    def __call__(self, x, y=None):
        if y is None:
            assert len(x) == 2, "Input list must contain exactly two elements."
            x, y = x[0], x[1]
        lp_distance = np.sum(np.abs(x - y) ** self.p) ** (1 / self.p)
        return lp_distance

    def get_settings(self):
        return {'p': self.p}
