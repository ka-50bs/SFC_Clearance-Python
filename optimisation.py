import numpy as np
from scipy import optimize
import miepython
import matplotlib.pyplot as plt

z
class Opti(object):

    def __init(self):
        self.__angles = None
        self.__lambda = None
        self.weight = None
        self.__mu = None
        self.__bounds = None
        self.__ind_exp = None

    def mie(self, _r, _n):
        return np.array(miepython.ez_intensities(_n, _r, self.__lambda, self.__mu)[0])

    def init_consts(self, laser_lambda, angles_range, resolution):
        self.__angles = np.linspace(angles_range[0], angles_range[1], resolution)
        self.__lambda = laser_lambda
        self.__mu = np.cos(self.__angles / 180 * np.pi)

    def init_weight(self):
        func = []
        for i in (self.__angles):
            if i == 0:
                func.append(0)
            else:
                func.append(1 / i * (np.exp(-2 * np.log(i / 54) ** 2)))
        self.weight = np.array(func)

    def init_bounds(self, r_bounds, n_bounds):
        self.__r_bounds = r_bounds
        self.__n_bounds = n_bounds

    def __opti_func(self, par):
        der = self.mie(par[0], par[1]) - self.__ind_exp
        der = der * self.weight
        der = (der) ** 2
        der = sum(der)
        return der

    def fit(self, __x):
        self.__ind_exp = __x
        res = optimize.shgo(self.__opti_func, [self.__r_bounds, self.__n_bounds],
                            n=100, iters=7, sampling_method='sobol')
        return np.array(res.x)

