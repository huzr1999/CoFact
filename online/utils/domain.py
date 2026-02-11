from abc import ABC, abstractmethod
import numpy as np


class Domain(ABC):

    def __init__(self, dimension):
        self._dimension = dimension
        self._R = None
        self._r = None
        pass

    @abstractmethod
    def projection(self, x):
        pass

    @abstractmethod
    def x_init(self, seed):
        pass

    @abstractmethod
    def unit_vec(self, seed):
        '''return a unit vector of dimension self.__diemension'''
        pass

    def get_dimension(self):
        return self._dimension

    def get_R(self):
        return self._R

    def get_r(self):
        return self._r


class Ball(Domain):

    def __init__(self, dimension, radius, center=None):
        super().__init__(dimension)
        self._center = center
        if self._center is None:
            self._center = np.zeros(dimension)
        self._R = radius
        self._r = radius

    def projection(self, x, shrink=0.):
        distance = np.linalg.norm(x - self._center)
        if distance > (1 - shrink) * self._r:
            x = self._center + (1 - shrink) * (x - self._center) * self._r / distance
        return x

    def x_init(self, seed=None):
        np.random.seed(seed)
        vec = np.random.randn(self._dimension)
        vec = vec / np.linalg.norm(vec)
        np.random.seed(seed)
        x = self._center + vec * self._r * np.random.rand()
        return x

    def unit_vec(self, seed=None):
        np.random.seed(seed)
        vec = np.random.randn(self._dimension)
        vec = vec / np.linalg.norm(vec)
        return vec


class Simplex(Domain):
    def __init__(self, dimension):
        super().__init__(dimension)

    def projection(self, x):
        return x / np.linalg.norm(x, ord=1)

    def x_init(self, seed=None):
        pass

    def unit_vec(self, seed=None):
        pass

    def one_hot_vec(self, i):
        if i >= self._dimension:
            raise ValueError(i, "is out of the index of dimension!")
        vec = np.zeros(self._dimension)
        vec[i] = 1
        return vec
