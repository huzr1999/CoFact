
import torch

from online.estimator.meta import AdaMLProd
from online.estimator.base import Base, ONS
from online.estimator.schedule import Schedule
from online.utils.domain import Ball, Simplex

import numpy as np


class Accous(Base):
    def __init__(self, cfgs, dim, seed=None, **alg_kwargs):
        # print(cfgs)
        # print(alg_kwargs)
        super(Accous, self).__init__(cfgs, dim=dim, seed=seed, **alg_kwargs)

        if cfgs is None:
            self.cfgs = {}
        else:
            self.cfgs = cfgs

        self.device = cfgs['device']
        self._t = 0

        self.T = cfgs['T']
        self.R = cfgs['R']
        self.S = cfgs['S']
        self.dim = dim
        self.domain = eval(cfgs['domain'])(dim, cfgs['R'])
        self.N = int(np.ceil(np.log2(self.T)))

        self.if_warm_restart = cfgs['if_warm_restart']
        self.coff_eta_base = cfgs['coff_eta_base']
        self.eta_base = self.coff_eta_base * (1 + np.exp(self.R * self.S))
        self.epsilon_base = cfgs['epsilon']
        self.eta_meta = 0.5
        self.coff_meta_prob = cfgs['coff_meta_prob']
        self.coff_output = cfgs['coff_output']
        self.weights_tranc = cfgs['weights_tranc']
        self.ignore_num = cfgs['ignore_num']

        cfgs['N'] = self.N
        alg_kwargs['dim'] = self.dim
        alg_kwargs['eta_base'] = self.eta_base
        alg_kwargs['epsilon_base'] = self.epsilon_base
        alg_kwargs['if_warm_restart'] = self.if_warm_restart

        self._schedule = self.expert_steup(**alg_kwargs)
        self._meta = self.meta_setup()

        self.loss_vector = torch.zeros(self.N-self.ignore_num)
        self.grad_vector = torch.zeros(self.N-self.ignore_num, self.dim)

        self.ori_source_data = []
        self.data_logreg = []
        self.label_logreg = []

        self.meta_weights_mean = torch.zeros(self.N-self.ignore_num)

    def expert_steup(self, **alg_kwargs):
        return Schedule(ONS, ['model'], self.N, self.cfgs, self.ignore_num, thread=self.cfgs.get('thread', 8), **alg_kwargs)

    def meta_setup(self):
        return AdaMLProd(N=self.N, lr=self.eta_meta * torch.ones(self.N-self.ignore_num), R=self.R, S=self.S, T=self.T, ignore_num=self.ignore_num)

    def get_model_grad(self, data, target, model):
        grad = 0.
        for i in range(len(target)):
            grad += -1 * target[i] * data[i] / (1 + np.exp(target[i] * np.dot(model, data[i])))

        return grad/len(target)

    def get_error(self):
        return np.dot(self._meta.get_prob(), self.loss_vector)

    def update(self):
        self._t += 1

        model_combine, _ = self.combined_model_grad()
        grad_combine = self.get_model_grad(self.data_logreg, self.label_logreg, model_combine)
        if self.if_warm_restart:
            self._schedule.set_restart_model(model_combine.numpy())
        else:
            self._schedule.set_restart_model(np.zeros(self.dim))

        self.loss_vector, _, self.acc_vector = self._schedule.opt(self.data_logreg.float(), self.label_logreg.float())
        experts = self._schedule.get_model()
        self._meta.opt(grad_combine, experts, model_combine)

        return self.loss_vector, self.acc_vector    

    def predict(self):
        prob = self._meta.get_prob()
        prob = prob ** self.coff_meta_prob
        prob /= prob.sum()

        soft_out = []
        data = self.ori_source_data.to(self.device).to(torch.float32)
        for model in self._schedule.bases:
            output, _ = model.predict(data)
            soft_out.append(output)

        soft_out = torch.tensor(soft_out).float()
        num_bases = len(self._schedule)
        combine_out = torch.mm(prob.view(1, num_bases), soft_out.view(num_bases, -1))
        combine_out = combine_out.reshape(soft_out.shape[1:])
        pred = combine_out.numpy()

        weights = torch.tensor(1 / pred - 1)
        weights = weights ** self.coff_output
        weights = torch.minimum(torch.tensor(self.weights_tranc * torch.ones(len(weights))), weights)

        return weights

    def combined_model_grad(self):
        prob = self._meta.get_prob()
        models = torch.tensor(np.array(self._schedule.get_model())).float()

        try:
            model_combine = (prob.view(1, self.N-self.ignore_num) @ models.view(self.N-self.ignore_num, -1))[0]
            grad_combine = (prob.view(1, self.N-self.ignore_num) @ torch.tensor(self.grad_vector, dtype=torch.float32).view(self.N-self.ignore_num, -1))[0]
        except RuntimeError:
            from IPython import embed
            embed()

        return model_combine, grad_combine

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def eval(self, model):
        err_cnt = 0
        pred_list = []
        soft_pred_list = []
        for i in range(len(self.label_logreg)):
            output = np.dot(model, self.data_logreg[i])
            soft_pred = 1.0 / (1.0 + np.exp(-output))
            soft_pred_list.append(soft_pred)
            pred = 1 if (soft_pred > 0.5) else -1
            pred_list.append(pred)
            err_cnt += (pred != self.label_logreg[i]).int()

        err_ = err_cnt / len(self.label_logreg)

        return 1.-err_

    def load_data(self, source_data, target_data):
        self.ori_source_data = source_data
        data_number = target_data.shape[0]

        idx = torch.randperm(source_data.shape[0])
        idx = idx[:data_number]
        source_data = source_data[idx]
        source_label = torch.ones(data_number).long()
        target_label = -1 * torch.ones(data_number).long()
        data_logreg = torch.cat((source_data, target_data), dim=0)
        label_logreg = torch.cat((source_label, target_label), dim=0)
        idx = torch.randperm(2 * data_number)
        data_logreg = data_logreg[idx]
        label_logreg = label_logreg[idx]

        self.data_logreg = data_logreg
        self.label_logreg = label_logreg

    def estimate(self, source_data, target_data):

        self.load_data(source_data, target_data)

        weights = self.predict()

        self.update()

        return weights, self.get_error()

    def predict_(self, data):
        prob = self._meta.get_prob()
        # prob = torch.zeros_like(prob)
        # prob[0] = 1
        prob = prob ** self.coff_meta_prob
        prob /= prob.sum()

        soft_out = []
        data = data.to(self.device).to(torch.float32)
        for model in self._schedule.bases:
            output, _ = model.predict(data)
            soft_out.append(output)

        soft_out = torch.tensor(soft_out).float()
        num_bases = len(self._schedule)
        combine_out = torch.mm(prob.view(1, num_bases), soft_out.view(num_bases, -1))
        combine_out = combine_out.reshape(soft_out.shape[1:])
        pred = combine_out.numpy()

        weights = torch.tensor(1 / pred - 1)
        weights = weights ** self.coff_output
        weights = torch.minimum(torch.tensor(self.weights_tranc * torch.ones(len(weights))), weights)

        return weights
    
    def predict_and_update(self, source_data, target_data, data_to_estimate):

        self.load_data(source_data, target_data)

        weights = self.predict_(data_to_estimate)
        loss_vector, acc_vector = self.update()

        return weights, loss_vector, acc_vector
