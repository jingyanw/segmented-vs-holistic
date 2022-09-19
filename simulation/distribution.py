import numpy as np
import scipy.stats

class Distribution:
    def __init__(self):
        return
    def sample(self, size=1):
        return

# Power-law distribution
class DistPowerLaw(Distribution):
    # Parameter: DELTA > 0
    def __init__(self, delta, corr=None):
        self.delta = delta
        self.corr = corr
    def sample(self, size=1): # range [1, inf)
        if self.corr is None: # iid
            return np.random.pareto(1 + self.delta, size=size) + 1

        # generate pdf from Gaussian
        assert(len(size) == 2)
        assert(np.abs(self.corr) <= 1)
        (n, d) = size
        # COV: 1 on diagonal and CORR elsewhere
        cov = np.ones((d, d)) * self.corr + np.eye(d) * (1-self.corr)
        mean = np.zeros(d)
        mtx = np.random.multivariate_normal(mean, cov, size=n) # nxd
        mtx_cdf = scipy.stats.norm.cdf(mtx)
        return np.power(1-mtx_cdf, -1/(1+self.delta))

    # compute inverse cdf of each individual value in DATA
    def inverse_cdf(self, data):
        assert(np.all(data >= 1))
        return 1 - np.power(data, -(1+self.delta))
