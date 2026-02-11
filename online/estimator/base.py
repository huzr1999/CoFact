# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy

from scipy import optimize
from scipy.optimize import NonlinearConstraint
from online.utils.domain import Ball, Simplex
from utils.logger import get_logger
import logging
import threading

class Base(object):
    def __init__(self, cfgs, dim, seed=None, **alg_kwargs):
        self.cfgs = cfgs
        self._seed = seed
        self.alg_kwargs = alg_kwargs

        self._domain = domain = eval(cfgs['domain'])(dim, cfgs['R'])
        self._func = None
        self._grad = None
        self._x = domain.x_init(seed)

        self._feature = None
        self._label = None

    def get_model(self):
        return self._x

    def set_model(self, a):
        self._x = copy.deepcopy(a)

    def set_feature(self, feature):
        self._feature = feature.numpy()

    def set_label(self, y):
        self._label = y.numpy()

    def set_func(self, func):
        self._func = func

    def cal_logist_grad(self):
        grad = -1 * self._label * self._feature / (1 + np.exp(self._label * np.dot(self._x, self._feature)))
        return grad

    def clear(self):
        self._func = None
        self._grad = None

class ONS(Base):
    def __init__(self, cfgs=None, seed=None, **algo_kwargs):
        super(ONS, self).__init__(cfgs=cfgs, seed=seed, **algo_kwargs)
        self._r = self._domain.get_r()
        self._square_r = self._r ** 2
        self._gamma = algo_kwargs['eta_base']
        self._eps = algo_kwargs['epsilon_base']
        self._d = self._domain.get_dimension()
        self._A_t = self._eps * np.eye(self._d)
        self._y = -1
        self._feature = np.zeros(self._d)
        self.grad = np.zeros((self._d, self._d))
        self.logger = get_logger(__name__, logging.DEBUG)

    def init_model(self, a):
        self._x = copy.deepcopy(a)
        self._A_t = self._eps * np.eye(self._d)
        self._inv_A_t = np.eye(self._d) / self._eps

    def opt(self):
        x = self._x.copy()
        loss = self._func(self._x)
        gradient = self.cal_logist_grad()

        pred = np.sum(self.predict_proba(self._feature) > 0.5)

        # if "_7" or "_0" in threading.current_thread().name:
        # self.logger.debug("The loss of gradient is: %s", gradient.mean())
        self.grad = gradient
        tmp = np.outer(gradient, gradient)
        self._A_t += tmp
        self._inv_A_t -= np.dot(np.dot(self._inv_A_t, tmp), self._inv_A_t) / (1 + np.dot(np.dot(gradient, self._inv_A_t), gradient))
        # print(type(x.dtype), type(self._gamma), type(self._inv_A_t.dtype), type(gradient.dtype))
        y = x - self._gamma * self._inv_A_t.dot(gradient)
        # print(y.dtype)
        self._x = self.proj(y)
        self.clear()
        return x, loss, gradient, pred

    def proj(self, y):
        obj_fun = lambda x: 1 / 2 * (x - y).dot(self._A_t).dot(x - y)
        jac_obj_fun = lambda x: self._A_t.dot(x - y)
        x_init = self.get_model().astype(np.float64)  # use last round point as start point
        nlc = NonlinearConstraint(lambda x: np.dot(x, x), -np.inf, self._square_r, jac=lambda x: 2 * x)
        result = optimize.minimize(obj_fun, x_init, jac=jac_obj_fun, method='SLSQP', constraints=[nlc])
        if not result.success:
            print("Fail, use last round decision instead!")
            return self.get_model()
        else:
            return np.array(result.x)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, row):
        z = np.dot(row, self._x)
        return self.sigmoid(z)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = X.cpu().detach().numpy()
        self.predict_probas = []
        for i in range(len(X)):
            ypred = self.predict_proba(X[i])
            self.predict_probas.append(ypred)
        self.cutoff = 0.5

        return self.predict_probas, (np.array(self.predict_probas) >= self.cutoff) * 1.0
