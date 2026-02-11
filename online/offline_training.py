# /usr/bin/env python
# -*- coding: utf-8 -*-


# import json as js
import os
# import time
# from os.path import join

import torch
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging              
# import random
# import numpy as np

# import pickle
import random
from datetime import datetime

import torch
# import argparse
# from wildtime import dataloader
import os
# import sys
# from torchvision import datasets, transforms
# from PIL import Image
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

# from dataset.wrapper import get_dataset
# from model.wrapper import get_model
# from online.models.wrapper import get_classifier_alg
# from online.estimator.wrapper import get_estimator_alg
# from online.utils.risk import *
# from utils.argparser import argparser
# from online.estimator.get_weights import weights_estimation

# from utils.logger import MyLogger



from utils.tools import Timer

timer = Timer()

import warnings

warnings.filterwarnings('ignore')

from online.model import Linear


def evaluation(model, data, label):
    pred_label = model.predict(data)
    correct_num = (label == pred_label).sum().item()
    acc = correct_num / len(label)
    error = 1. - acc

    return error, acc

def offline_train(X, Y, num_cls, device, saved_model_path, lr, batch_size, R, num_epochs):

    logger = logging.getLogger(__name__)

    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join(saved_model_path, timestamp_str)
    log_file_path = os.path.join(model_dir, 'training.log')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    input_dim = X.shape[1]

    X_tensor = torch.from_numpy(X).float().to(device)
    Y_tensor = torch.from_numpy(Y).long().to(device)
    train_dataset = TensorDataset(X_tensor, Y_tensor)


    model = Linear(
        input_dim=input_dim,
        output_dim=num_cls,
        R=R
    )
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  # pass the dataset to the dataloader
        batch_size=batch_size,  # a large batch size helps with the learning
        sampler=train_sampler,  # shuffling is important!
        drop_last=True,
        num_workers=0)  # apply transformations to the input images
    
    n_total = len(train_dataloader)
    for epoch in range(num_epochs):
        num_iter, correct_num, total_loss, last_loss = 0, 0, 0.0, 0.0
        for step, batch in enumerate(
            tqdm(train_dataloader, desc='Train | epoch:{} | loss:{}'.format(epoch + 1, last_loss))):
            # if distributed:
            #     dist.barrier()
            x, y = batch
            # x = x.to(device)
            # y = y.to(device)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            last_loss = torch.mean(loss.detach()).item()
            total_loss += last_loss
            loss.backward()
            optimizer.step()
            num_iter += 1
            cur_step = epoch * n_total + step


        output = model(X_tensor)
        _, pred = output.max(1)
        correct_num = (Y_tensor == pred).sum().item()
        avg_prec = correct_num / len(train_dataset)
        avg_loss = total_loss / len(train_dataloader)

        logger.info(f"Train: epoch: {epoch + 1:>02}, cur_step: {cur_step}, loss: {avg_loss:.5f}, acc: {avg_prec:.5f}")

        with open(log_file_path, "a") as file:
            file.write(f"Train: epoch: {epoch + 1:>02}, cur_step: {cur_step}, loss: {avg_loss:.5f}, acc: {avg_prec:.5f}\n")


    model_path = os.path.join(model_dir, 'offline_model_{}_{}.pth'.format(epoch + 1, timestamp_str))

    # torch.save(model.state_dict(), model_path)

    # logger.info(f"Model saved to {model_path}")

    return model
