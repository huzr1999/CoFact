# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

from utils.multi_thread import MultiThreadHelper

from online.utils.loss_function import SquareLoss, LogisticLoss

import copy
from utils.logger import get_logger
import logging


np.set_printoptions(suppress=True)

class Schedule(object):
    def __init__(self, alg, cp_list, N, cfgs, ignore_num, thread=0, **alg_kwargs):
        self.bases = []
        self.dim = alg_kwargs['dim']
        self.if_warm_restart = alg_kwargs['if_warm_restart']
        self.restart_model = np.zeros(self.dim)

        for i in range(N-ignore_num):
            for k in cp_list:
                alg_kwargs[k] = alg_kwargs[k].new()
            self.bases.append(alg(cfgs=cfgs, seed=None,
                                  **alg_kwargs))
            self.bases[i].init_model(np.zeros(self.dim))

        self.ignore_num = ignore_num
        self.length = N
        self.threads = thread
        self._t = 0
        self.logger = get_logger(__name__, logging.DEBUG)

    def __len__(self):
        return self.length - self.ignore_num

    def get_model(self):
        output = []
        for i in range(self.length - self.ignore_num):
            output.append(self.bases[i].get_model())

        return output

    def set_restart_model(self, a):
        self.restart_model = copy.deepcopy(a)

    def opt(self, data, target):
        self._t += 1

        loss_vector = torch.zeros(self.length - self.ignore_num)
        grad_vector = []
        acc_vector = []

        def expert_opt(idx, expert, data, target):
            loss, grad, acc = 0., 0., 0

            for i in range(len(target)):
                func = LogisticLoss(data[i], target[i]).func
                expert.set_feature(data[i])
                expert.set_label(target[i])
                expert.set_func(func)
                _, _loss, _grad, _pred = expert.opt()
                loss += _loss
                grad += _grad
                acc += (_pred == ((target[i] + 1) / 2)).sum().item()
                # self.logger.debug(f"Expert {idx} - Sample {i} - Loss: {_loss}, - pred: {_pred} - label: {(target[i] + 1) / 2}, Acc: {_pred == (target[i] + 1) / 2}")
            loss /= len(target)
            grad /= len(target)
            acc /= len(target)
            

            if self._t % (2 ** (idx+self.ignore_num)) == 0:
                expert.set_model(self.restart_model)

            return idx, loss, grad, acc

        commands = [(expert_opt, idx, expert, data, target) for idx, expert in enumerate(self.bases)]
        loss_result = MultiThreadHelper(commands, self.threads, multi_process=False)()

        # base_learner_local_acc_list = [] 
        # if self._t % 10 == 0 and self._t > 0:
        #     local_dataset = data
        #     local_dist_labels = target

            # for model in self.bases:
            #     output, _ = model.predict(local_dataset.to("cpu").to(torch.float32))
            #     base_learner_local_acc_list.append(np.mean((np.array(output) > 0.5).astype(int) == ((local_dist_labels.numpy() + 1) / 2)))
            
            # self.logger.debug(f"The base learner local accuracy list at time {self._t} is: {base_learner_local_acc_list}")

        for idx, loss, grad, acc in loss_result:
            if loss is not None:
                loss_vector[idx] = loss
            if grad is not None:
                grad_vector.append(grad)
            if acc is not None:
                acc_vector.append(acc)

        return loss_vector, grad_vector, acc_vector