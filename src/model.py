import os
import json
from logging import getLogger
from config import Config
from lib.helper import string_to_function, euler
import numpy as np
from scipy.integrate import odeint

logger = getLogger(__name__)

class NeuralODEModel:
    def __init__(self, config: Config):
        self.config = config
        mc = config.model
        alpha = np.zeros(shape=(mc.division, mc.dim_out), dtype=np.float32)
        beta = np.zeros(shape=(mc.division, mc.dim_in, mc.dim_in), dtype=np.float32)
        gamma = np.zeros(shape=(mc.division, mc.dim_in), dtype=np.float32)
        self.params = [alpha, beta, gamma]
        self.A = np.eye(mc.dim_out, mc.dim_in, dtype=np.float32)
        self.function, self.d_function = string_to_function(mc.function_type)
        self.t = np.linspace(0., mc.max_time, mc.division)

    def __call__(self, x0):
        def func(t, x, params, A, function):
            index = int(t * (len(params[0]) - 1))
            return [np.dot(x[0], params[1][index].T) + params[2][index], params[0][index] * function(np.dot(x[0], A.T))]
        
        y0 = np.zeros(shape=(len(x0), self.config.model.dim_out), dtype=np.float32)
        #self.x = odeint(func, self.t, [x0, y0], args=(self.params, self.A, self.function))
        self.x = euler(func, self.t, [x0, y0], args=(self.params, self.A, self.function))
        return self.x[-1][1]
    
    def gradient(self, x0, y_true):
        def func(t, a, params, A, function, x):
            index = int(t * (len(params[0]) - 1))
            return [-np.dot(a[0], params[1][index]) - np.dot(a[1] * params[0][index] * function(np.dot(x[index][0], A.T)), A), np.zeros_like(x[-1][1])]
        
        n_data = len(x0)
        y_pred = self(x0)
        aT = np.zeros_like(x0)
        bT = (y_pred - y_true) / n_data 
        #a = odeint(func, self.t[::-1], [aT, bT], args=(self.params, self.A, self.d_function, self.x))
        a = euler(func, self.t[::-1], [aT, bT], args=(self.params, self.A, self.d_function, self.x))
        _a0 = np.array(list(map(lambda x: x[0], a)))[::-1].astype(np.float32)
        _a1 = np.array(list(map(lambda x: x[1], a)))[::-1].astype(np.float32)
        _x0 = np.array(list(map(lambda x: x[0], self.x)))[::-1].astype(np.float32)
        g_alpha = np.einsum("ijk,ijk->ik", _a1, self.function(np.dot(_x0, self.A.T)).astype(np.float32))
        g_beta = np.einsum("ilj,ilk->ijk", _a0, _x0)
        g_gamma = np.einsum("ijk->ik", _a0)
        return [g_alpha, g_beta, g_gamma]
    
    def load(self, model_path):
        if os.path.exists(model_path):
            logger.debug("loding model from {}".format(model_path))
            with open(model_path, "rt") as f:
                model_weights = json.load(f)
            alpha = np.array(model_weights.get("Alpha"))
            beta = np.array(model_weights.get("Beta"))
            gamma = np.array(model_weights.get("Gamma"))
            A = np.array(model_weights.get("A"))
            if self.params[0].shape == alpha.shape: self.params[0] = alpha
            if self.params[1].shape == beta.shape: self.params[1] = beta
            if self.params[2].shape == gamma.shape: self.params[2] = gamma
            if self.A.shape == A.shape: self.A = A
    
    def save(self, config_path, model_path):
        logger.debug("save model config to {}".format(config_path))
        self.config.save_parameter(config_path)
        logger.debug("save model to {}".format(model_path))
        model_data = {
            "Alpha": self.params[0].tolist(),
            "Beta": self.params[1].tolist(),
            "Gamma": self.params[2].tolist(),
            "A": self.A.tolist()
        }
        with open(model_path, "wt") as f:
            json.dump(model_data, f, indent=4)


def mean_square_error(y_pred, y_true):
    return np.mean(np.sum(np.square(y_pred - y_true), 1)) * 0.5

def cross_entropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1.), 1))

def accuracy(y_pred, y_true):
    if len(y_true[0]) == 1:
        return np.mean(np.equal(np.where(y_pred<0.5, 0, 1), y_true).astype(np.float32))
    else:
        return np.mean(np.equal(np.argmax(y_pred, 1), np.argmax(y_true, 1)).astype(np.float32))

            