import numpy as np

class Sigmoid:
    def __init__(self, gain=1.):
        self.gain = gain
        self.function = np.frompyfunc(self._sigmoid, 1, 1)
        self.d_function = np.frompyfunc(self._derivative_sigmoid, 1, 1)
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return self.d_function(x)
    
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-self.gain * x))
    
    def _derivative_sigmoid(self, x):
        return self.gain * self._sigmoid(x) * (1. - self._sigmoid(x))

class Relu:
    def __init__(self, gain=1.):
        self.gain = gain
        self.function = np.frompyfunc(self._relu, 1, 1)
        self.d_function = np.frompyfunc(self._derivative_relu, 1, 1)
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return d_function(x)
    
    def _relu(self, x):
        if x < 0.:
            return 0.
        else:
            return self.gain * x
    
    def _derivative_relu(self, x):
        if x < 0.:
            return 0.
        else:
            return self.gain

