# /usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class Hedge(object):
    def __init__(self, N, lr, prior):
        self._N = N
        self.lr = lr
        self.potential = torch.zeros(self._N)
        self.prob = self.set_prior(prior)
        self._t = 0

    def set_prior(self, prior):
        if prior == 'uniform':
            prob = [1 / self._N for i in range(self._N)]
        elif prior == 'nonuniform':
            prob = [(self._N + 1) / (self._N * i * (i + 1)) for i in range(1, self._N + 1)]
        else:
            prob = prior

        return torch.tensor(prob)

    @torch.no_grad()
    def opt(self, loss):
        self._t += 1

        # restart
        for idx in range(self._N):
            if self._t % (2 ** (idx + 1))==0:
                self.potential[idx] = 0
            else:
                self.potential[idx] += torch.dot(self.prob, loss) - loss[idx]

        ori_prob = self.prob.detach().clone()
        self.prob = torch.exp(self.lr * self.potential)
        self.prob /= self.prob.sum()
        if torch.any(torch.isnan(self.prob)):
            self.prob = ori_prob

    def get_prob(self):
        return self.prob

    def get_potential(self):
        return self.potential

    def update_lr(self, **kwargs):
        lr = self.lr

        self.lr = lr

class AdaMLProd(object):
    def __init__(self, N, lr, R, S, T, ignore_num):
        self._N = N
        self._T = T
        self.R = R
        self.S = S
        self.lr = lr
        self.ignore_num = ignore_num

        self._t = 0
        self.reg = torch.ones(self._N-self.ignore_num)/self._T
        self.sqr_reg = torch.zeros(self._N-self.ignore_num)
        self.potential = torch.ones(self._N-self.ignore_num)/self._T
        self.prob = torch.ones(self._N-self.ignore_num)/(self._N-self.ignore_num)


    @torch.no_grad()
    def opt(self, grad_combine, models, model_combine):
        self._t += 1
        last_prob = self.prob.detach().clone()
        last_lr = self.lr.detach().clone()

        for idx in range(self._N-self.ignore_num):
            if self._t % (2 ** (idx+self.ignore_num))==0:
                self.lr[idx] = torch.min(torch.tensor(0.5),
                                         torch.sqrt(torch.log(torch.tensor(self._N-self.ignore_num).float()) / (1 + self.sqr_reg[idx])))
                self.sqr_reg[idx] = 0.
                self.potential[idx] = 1./self._T
            else:
                self.lr[idx] = torch.min(torch.tensor(0.5),
                                         torch.sqrt(torch.log(torch.tensor(self._N-self.ignore_num).float()) / (1 + self.sqr_reg[idx])))
                self.reg[idx] = torch.dot(grad_combine.float(), (model_combine - torch.tensor(models[idx])).float()) / (2 * self.S * self.R)
                self.sqr_reg[idx] += self.reg[idx]**2
                if (1 + last_lr[idx] * self.reg[idx]) < 0:
                    self.potential[idx] = 1./self._T
                else:
                    self.potential[idx] = (self.potential[idx] * (1 + last_lr[idx] * self.reg[idx])) ** (self.lr[idx] / last_lr[idx])

        self.prob = torch.mul(self.lr, self.potential)
        self.prob /= self.prob.sum()
        if torch.any(torch.isnan(self.prob)):
            self.prob = last_prob

    def get_prob(self):
        return self.prob

    def get_reg(self):
        return self.reg

    def get_accu_reg(self):
        return self.sqr_reg

    def get_potential(self):
        return self.potential

    def get_lr(self):
        return self.lr