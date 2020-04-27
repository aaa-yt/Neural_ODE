import sys
sys.path.append("../")
from config import Config
import numpy as np


class Optimizer(object):
    def __init__(self, config: Config):
        self.config = config
        self.mc = config.model
        self.tc = config.trainer


class SGD(Optimizer):
    def __init__(self, config: Config):
        super(SGD, self).__init__(config)
        self.rate = self.tc.rate

    def __call__(self, params, g_params):
        return tuple(param - self.rate * g_param for param, g_param in zip(params, g_params))


class Momentum(Optimizer):
    def __init__(self, config: Config):
        super(Momentum, self).__init__(config)
        self.rate = self.tc.rate
        self.momentum = self.tc.momentum
        alpha = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha, beta, gamma)
    
    def __call__(self, params, g_params):
        new_params = tuple(param + self.momentum * (param - v) - self.rate * g_param for param, g_param, v in zip(params, g_params, self.v))
        self.v = params
        return new_params


class AdaGrad(Optimizer):
    def __init__(self, config: Config):
        super(AdaGrad ,self).__init__(config)
        self.rate = self.tc.rate
        alpha = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha, beta, gamma)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(param - np.multiply(np.divide(self.rate, np.sqrt((v + np.square(g_param) + self.eps).astype(np.float32))), g_param), v + np.square(g_param)) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


class RMSprop(Optimizer):
    def __init__(self, config: Config):
        super(RMSprop, self).__init__(config)
        self.rate = self.tc.rate
        self.decay = self.tc.decay
        alpha = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha, beta, gamma)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(param - np.multiply(np.divide(self.rate, np.sqrt((self.decay * v + (1. - self.decay) * np.square(g_param) + self.eps).astype(np.float32))), g_param), self.decay * v + (1. - self.decay) * np.square(g_param)) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


class AdaDelta(Optimizer):
    def __init__(self, config: Config):
        super(AdaDelta, self).__init__(config)
        self.decay = self.tc.decay
        alpha = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha, beta, gamma)
        self.s = (alpha, beta, gamma)
        self.params_prev = (alpha, beta, gamma)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v, new_s = zip(*[(param - np.multiply(np.divide(np.sqrt((self.decay * s + (1. - self.decay) * np.square(param - param_prev) + self.eps).astype(np.float32)), np.sqrt((self.decay * v + (1. - self.decay) * np.square(g_param) + self.eps).astype(np.float32))), g_param), self.decay * v + (1. - self.decay) * np.square(g_param), self.decay * s + (1. - self.decay) * np.square(param - param_prev)) for param, g_param, v, s, param_prev in zip(params, g_params, self.v, self.s, self.params_prev)])
        self.param_prev = params
        self.v = new_v
        self.s = new_s
        return new_params


class Adam(Optimizer):
    def __init__(self, config: Config):
        super(Adam, self).__init__(config)
        self.rate = self.tc.rate
        self.decay1 = self.tc.decay
        self.decay2 = self.tc.decay2
        alpha = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha, beta, gamma)
        self.s = (alpha, beta, gamma)
        self.t = 1
        self.eps = 1e-8

    def __call__(self, params, g_params):
        new_params, new_v, new_s = zip(*[(param - np.multiply(np.divide(self.rate, np.sqrt((np.divide(self.decay2 * s + (1. - self.decay2) * np.square(g_param), 1. - self.decay2 ** self.t) + self.eps).astype(np.float32))), np.divide(self.decay1 * v + (1. - self.decay1) * g_param, 1. - self.decay1 ** self.t)), self.decay1 * v + (1. - self.decay1) * g_param, self.decay2 * s + (1. - self.decay2) * np.square(g_param)) for param, g_param, v, s in zip(params, g_params, self.v, self.s)])
        self.v = new_v
        self.s = new_s
        self.t += 1
        return new_params


def get(config: Config):
    tc = config.trainer
    all_optimizer = {
        "sgd": SGD(config=config),
        "momentum": Momentum(config=config),
        "adagrad": AdaGrad(config=config),
        "rmsprop": RMSprop(config=config),
        "adadelta": AdaDelta(config=config),
        "adam": Adam(config=config)
    }

    optimizer_type = tc.optimizer_type
    if optimizer_type.lower() in all_optimizer:
        optimizer_type = optimizer_type.lower()
        return all_optimizer[optimizer_type]
    else:
        return all_optimizer["sgd"]