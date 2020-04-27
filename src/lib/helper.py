import numpy as np

from .functions import Sigmoid, Relu

def string_to_function(function_type):
    all_functions = {
        "sigmoid": Sigmoid(),
        "relu": Relu()
    }
    if function_type.lower() in all_functions:
        function_type = function_type.lower()
        return all_functions[function_type], all_functions[function_type].derivative
    else:
        return all_functions["sigmoid"], all_functions["sigmoid"].derivative

def euler(func, x0, t, args=None):
    solution = [x0]
    x = x0
    for i, dt in enumerate(np.diff(t)):
        x = x + dt * func(x, t[i], *args)
        solution.append(x)
    return np.array(solution, dtype=np.float32)