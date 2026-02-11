from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):

    def __init__(self, x, y):
        self._x = x
        self._y = y

    @abstractmethod
    def func(self, x):
        pass


class SquareLoss(LossFunction):
    def __init__(self, x, y, scale=1):
        super().__init__(x, y)
        self._scale = scale

    def func(self, w):
        return self._scale * 1 / 2 * ((np.dot(w, self._x) - self._y)**2)


class LogisticLoss(LossFunction):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def func(self, z):
        if type(z) is not np.float64:
            loss = np.log(1 + np.exp(-self._y * np.dot(z, self._x)))
        else:
            loss = np.log(1 + np.exp(-self._y * z))
        return loss

    def sigmoid(self, x):
        return 1/(1 + np.e ** (-x))