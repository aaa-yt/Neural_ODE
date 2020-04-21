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
        new_params = []
        for param, g_param in zip(params, g_params):
            new_params.append(param - self.rate * g_param)
        return new_params

class Momentum(Optimizer):
    def __init__(self, config: Config):
        super(Momentum, self).__init__(config)
        self.rate = self.tc.rate
        self.momentum = self.tc.momentum
        alpha = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        self.v = [alpha, beta, gamma]
    
    def __call__(self, params, g_params):
        new_params = []
        new_v = []
        for param, g_param, v in zip(params, g_params, self.v):
            g = self.momentum * v - self.rate * g_param
            new_v.append(g)
            new_params.append(param + g)
        self.v = new_v
        return new_params

class AdaGrad(Optimizer):
    def __init__(self, config: Config):
        super(AdaGrad ,self).__init__(config)
        self.rate = self.tc.rate
        alpha = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        self.v = [alpha, beta, gamma]
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params = []
        new_v = []
        for param, g_param, v in zip(params, g_params, self.v):
            g = v + np.square(g_param)
            new_v.append(g)
            new_params.append(param - np.multiply(np.divide(self.rate, np.sqrt((g + self.eps).astype(np.float32))), g_param))
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
        self.v = [alpha, beta, gamma]
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params = []
        new_v = []
        for param, g_param, v in zip(params, g_params, self.v):
            g = self.decay * v + (1. - self.decay) * np.square(g_param)
            new_v.append(g)
            new_params.append(param - np.multiply(np.divide(self.rate, np.sqrt((g + self.eps).astype(np.float32))), g_param))
        self.v = new_v
        return new_params


class AdaDelta(Optimizer):
    def __init__(self, config: Config):
        super(AdaDelta, self).__init__(config)
        self.decay = self.tc.decay
        alpha = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        self.v = [alpha, beta, gamma]
        self.s = [alpha, beta, gamma]
        self.params_prev = [alpha, beta, gamma]
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params = []
        new_v, new_s = [], []
        for param, g_param, v, s, param_prev in zip(params, g_params, self.v, self.s, self.params_prev):
            gv = self.decay * v + (1. - self.decay) * np.square(g_param)
            gs = self.decay * s + (1. - self.decay) * np.square(param - param_prev)
            new_v.append(gv)
            new_s.append(gs)
            new_params.append(param - np.multiply(np.divide(np.sqrt((gs + self.eps).astype(np.float32)), np.sqrt((gv + self.eps).astype(np.float32))), g_param))
        self.params_prev = params
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
        self.v = [alpha, beta, gamma]
        self.s = [alpha, beta, gamma]
        self.t = 1
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params = []
        new_v, new_s = [], []
        for param, g_param, v, s in zip(params, g_params, self.v, self.s):
            gv = self.decay1 * v + (1. - self.decay1) * g_param
            gs = self.decay2 * s + (1. - self.decay2) * np.square(g_param)
            new_v.append(gv)
            new_s.append(gs)
            new_params.append(param - np.multiply(np.divide(self.rate, np.sqrt(np.divide(gs, 1. - self.decay2 ** self.t) + self.eps)), np.divide(gv, 1. - self.decay1 ** self.t)))
        self.v = new_v
        self.s = new_s
        self.t = self.t + 1
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