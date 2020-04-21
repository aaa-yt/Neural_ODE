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

def euler(func, t, x0, args=None):
    solution = [x0]
    x = x0
    for i, dt in enumerate(np.diff(t)):
        x = list(map(lambda a, b: a+b, x, list(map(lambda a: a*dt, func(t[i], x, *args)))))
        solution.append(x)
    return solution