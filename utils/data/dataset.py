import torch
import torch.utils.data as data
import logging
import numpy as np
from utils.shift_simulate import LinearShift, FixShift, NoShift, SineShift, SquareShift, BernoulliShift
import math
from utils.logger import get_logger
import umap

class TrainSet(data.Dataset):
    def __init__(self, scores, prompt_level_features, annotations):
        self.scores = scores
        self.prompt_level_features = prompt_level_features
        self.annotations = annotations

        assert len(self.scores) == len(self.prompt_level_features) == len(self.annotations)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):

        return self.scores[idx], self.prompt_level_features[idx], self.annotations[idx], idx


class TestSet(data.Dataset):
    def __init__(self, scores, annotations, prompt_level_features, responses_rep, num_samples_each_round, T, shift_cfg, rng):
        self.scores = scores
        self.prompt_level_features = prompt_level_features
        self.annotations = annotations
        self.responses_rep = responses_rep
        self.num_samples_each_round = num_samples_each_round
        self.shift_base_features = shift_cfg["shift_base_features"]

        # print(prompt_level_features.mean(axis=0))

        if self.shift_base_features == "prompt_level_features": 
            self.shift_gen = eval(shift_cfg["type"])(**shift_cfg["kwargs"], x=prompt_level_features, T=T, rng=rng)
        elif self.shift_base_features == "responses_rep":
            beta = np.ones(responses_rep.shape[1])
            coefficient = shift_cfg["kwargs"].get("coefficient", 1.0)
            self.shift_gen = eval(shift_cfg["type"])(beta=beta, coefficient=coefficient, x=responses_rep, T=T, rng=rng)
        else:
            raise ValueError("No such shift base feature: {}".format(shift_cfg["shift_base_features"]))

        assert len(self.scores) == len(self.prompt_level_features) == len(self.annotations) == len(self.responses_rep)
        
        self.logger = get_logger(__name__, logging.DEBUG)
        self.cur_weights = None

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, t):

        weights = self.shift_gen(t)

        sampled_index = torch.multinomial(weights, num_samples=self.num_samples_each_round, replacement=True).cpu().numpy()

        self.cur_weights_of_sampled = weights[sampled_index] / weights.sum()

        return (
            [self.scores[idx] for idx in sampled_index], 
            [self.annotations[idx] for idx in sampled_index],
            self.prompt_level_features[sampled_index], 
            self.responses_rep[sampled_index],
            sampled_index
        )
    
    def get_cur_weights_of_sampled(self):
        return self.cur_weights_of_sampled.cpu().numpy()

    def calculate_weights_for_samples(self, test_set_x, t):
        return self.shift_gen.get_weights(test_set_x, t)
    
    def sample_N_test_samples(self, t, N):
        if self.shift_base_features == "prompt_level_features": 
            base_features = self.prompt_level_features
        elif self.shift_base_features == "responses_rep":
            base_features = self.responses_rep
        else:
            raise ValueError("No such shift base feature: {}".format(self.shift_base_features))


        test_weights = self.calculate_weights_for_samples(base_features, t)
        sampled_inds = torch.multinomial(test_weights, num_samples=N, replacement=True).cpu().numpy()
        return sampled_inds

def normalize_features(train_features: np.ndarray, test_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalizes a set of features using min-max scaling based on the training data.
    Args:
        train_features (np.ndarray): The training feature matrix.
        test_features (np.ndarray): The testing feature matrix.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the normalized training and testing feature matrices.
    """
    feature_min = np.min(train_features, axis=0)
    feature_range = np.max(train_features, axis=0) - feature_min

    feature_range[feature_range == 0] = 1

    normalized_train = (train_features - feature_min) / feature_range
    normalized_test = (test_features - feature_min) / feature_range

    return normalized_train, normalized_test

def reduce_dimensions(train_features: np.ndarray, test_features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduces the dimensionality of features using UMAP.

    Args:
        train_features (np.ndarray): The training feature matrix.
        test_features (np.ndarray): The testing feature matrix.
        n_components (int): The number of dimensions to reduce to.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the dimension-reduced training and testing feature matrices.
    """
    reducer = umap.UMAP(n_components=n_components)
    reducer.fit(train_features)
    
    reduced_train = reducer.transform(train_features)
    reduced_test = reducer.transform(test_features)
    
    return reduced_train, reduced_test
