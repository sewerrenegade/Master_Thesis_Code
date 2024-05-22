import numpy as np
from scipy.spatial.distance import mahalanobis

class MahalanobisDistanceCalculator:
    def __init__(self, input_distribution_data) -> None:
        self.mu = np.mean(input_distribution_data, axis=0)
        self.sigma = np.cov(input_distribution_data.T)
        self.inv_sigma = np.linalg.inv(self.sigma)

    def distance_from_distribution(self,point):
        dist = mahalanobis(point, self.mu, np.linalg.inv(self.sigma))
        return dist
    
    def distance_from_distribution(self,point):
        dist = mahalanobis(point, self.mu, np.linalg.inv(self.sigma))
        return dist