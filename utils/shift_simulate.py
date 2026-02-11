from abc import ABC, abstractmethod
import numpy as np
import torch
from utils.logger import get_logger
import logging
import math


class LinearShift:

    def __init__(self, T, beta_start, coefficient_start, beta_end, coefficient_end, x, rng):
        self.T = T
        self.x = torch.from_numpy(x).float() # The representations of test samples
        self.beta_start = torch.tensor(beta_start).float() * coefficient_start
        self.beta_end = torch.tensor(beta_end).float() * coefficient_end

        feature_dim = self.x.shape[1]


        assert len(self.beta_start) == feature_dim and len(self.beta_end) == feature_dim


    def __call__(self, t):

        t_norm = min(t, self.T) / self.T
        beta_t = (1 - t_norm) * self.beta_start + t_norm * self.beta_end
        
        # print(self.prompt_level_features_tensor.dtype, beta_t.dtype)
        weights = torch.exp(self.x @ beta_t)
        return weights
        
    def get_weights(self, x, t):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        t_norm = min(t, self.T) / self.T
        beta_t = (1 - t_norm) * self.beta_start + t_norm * self.beta_end
        
        # print(type(x), type(beta_t.dtype))
        
        weights = torch.exp(x @ beta_t)
        return weights

class NoShift:

    def __init__(self, T, coefficient, beta, x, rng):
        self.num_samples = x.shape[0]

    def __call__(self, t):

        weights = torch.ones(self.num_samples)
        return weights

    def get_weights(self, x, t):
        return torch.ones_like(x)

class FixShift:

    def __init__(self, T, coefficient, beta, x, rng):
        self.num_samples = x.shape[0]
        self.coefficient = coefficient
        self.beta = torch.tensor(beta).float() * self.coefficient
        self.x = torch.from_numpy(x).float()
        self.weights = self.get_weights(self.x, 0)


    def __call__(self, t):
        return self.weights

    def get_weights(self, x, t):
        return self.coefficient * torch.exp(self.beta @ x.T)



class SineShift:
    """
    Simulates a periodic covariate shift where beta coefficients change
    periodically between beta_start and beta_end using a sine wave.
    The period of oscillation is M = 2 * sqrt(T).
    """
    def __init__(self, T, beta_start, coefficient_start, beta_end, coefficient_end, x, rng):
        self.T = T
        self.x = torch.from_numpy(x).float() # The representations of test samples
        self.beta_start = torch.tensor(beta_start).float() * coefficient_start
        self.beta_end = torch.tensor(beta_end).float() * coefficient_end

        # Calculate the new period M based on T
        self.M = 2 * math.sqrt(self.T)

        feature_dim = self.x.shape[1]

        assert len(self.beta_start) == feature_dim and len(self.beta_end) == feature_dim

    def __call__(self, t):
        # Calculate the periodic factor using a cosine wave.
        # This factor oscillates between 1 and 0 based on the period M.
        oscillation_factor = (1 + math.cos((2 * math.pi * t) / self.M)) / 2

        # Interpolate beta_t based on the oscillation factor.
        beta_t = oscillation_factor * self.beta_start + (1 - oscillation_factor) * self.beta_end
        
        weights = torch.exp(self.x @ beta_t)
        return weights
        
    def get_weights(self, x, t):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Same logic as the __call__ method, using the new period M.
        oscillation_factor = (1 + math.cos((2 * math.pi * t) / self.M)) / 2
        beta_t = oscillation_factor * self.beta_start + (1 - oscillation_factor) * self.beta_end
        
        weights = torch.exp(x @ beta_t)
        return weights


class SquareShift:
    """
    Simulates a periodic covariate shift where beta coefficients change
    periodically between beta_start and beta_end using a square wave.
    The period of oscillation is M = 2 * sqrt(T).
    """
    def __init__(self, T, beta_start, coefficient_start, beta_end, coefficient_end, x, rng):
        self.T = T
        self.x = torch.from_numpy(x).float() # The representations of test samples
        self.beta_start = torch.tensor(beta_start).float() * coefficient_start
        self.beta_end = torch.tensor(beta_end).float() * coefficient_end

        # Calculate the new period M based on T
        self.M = 2 * math.sqrt(self.T)

        feature_dim = self.x.shape[1]

        assert len(self.beta_start) == feature_dim and len(self.beta_end) == feature_dim

    def __call__(self, t):
        # Calculate the periodic factor using a square wave.
        # This factor oscillates between 1 and 0 based on the period M.
        oscillation_factor = 1 if (t // (self.M / 2)) % 2 == 0 else 0
        # Interpolate beta_t based on the oscillation factor.
        beta_t = oscillation_factor * self.beta_start + (1 - oscillation_factor) * self.beta_end
        
        weights = torch.exp(self.x @ beta_t)
        return weights
        
    def get_weights(self, x, t):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Same logic as the __call__ method, using the new period M.
        oscillation_factor = 1 if (t // (self.M / 2)) % 2 == 0 else 0
        beta_t = oscillation_factor * self.beta_start + (1 - oscillation_factor) * self.beta_end
        
        weights = torch.exp(x @ beta_t)
        return weights
    
class BernoulliShift:
    """
    Simulates a covariate shift where beta coefficients switch between
    beta_start and beta_end based on a Bernoulli process with probability p = 1/sqrt(T).
    """
    def __init__(self, T, beta_start, coefficient_start, beta_end, coefficient_end, x, rng):
        self.T = T
        self.x = torch.from_numpy(x).float() # The representations of test samples
        self.beta_start = torch.tensor(beta_start).float() * coefficient_start
        self.beta_end = torch.tensor(beta_end).float() * coefficient_end

        # Calculate the switching probability p based on T
        self.p = 1 / math.sqrt(self.T)
        self.current_beta = self.beta_start

        feature_dim = self.x.shape[1]

        assert len(self.beta_start) == feature_dim and len(self.beta_end) == feature_dim

        # Initialize random number generator
        self.rng = rng

        self.dist_decision = self.rng.random(size=self.T) < self.p
        self.dist_flag = True

    def __call__(self, t):
        # Decide whether to switch beta_t based on the Bernoulli process.
        if self.dist_decision[t]:
            self.dist_flag = not self.dist_flag

        self.current_beta = self.beta_start if self.dist_flag else self.beta_end
        
        weights = torch.exp(self.x @ self.current_beta)
        return weights
        
    def get_weights(self, x, t):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Same logic as the __call__ method for switching beta_t.
        if self.dist_decision[t]:
            self.dist_flag = not self.dist_flag
            # self.current_beta = self.beta_end if torch.equal(self.current_beta, self.beta_start) else self.beta_start
        self.current_beta = self.beta_start if self.dist_flag else self.beta_end
        
        weights = torch.exp(x @ self.current_beta)
        return weights