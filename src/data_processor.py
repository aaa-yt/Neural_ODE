import os
import json
from logging import getLogger
from config import Config
import numpy as np

logger = getLogger(__name__)

def start(config: Config):
    return DataProcessor(config).start()

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
    
    def start(self):
        data_path = self.config.resource.data_path
        if os.path.exists(data_path):
            logger.debug("Dataset already exists")
        else:
            self.dataset = self.create_dataset()
            self.save_dataset(data_path)
    
    def create_dataset(self):
        logger.info("Create a new dataset")
        n_train = 1000
        n_val = 333
        sigma = 0.
        train = self.get_data(n_train, sigma)
        validation = self.get_data(n_val, sigma)
        #train = self.get_data3(n_train)
        #validation = self.get_data3(n_val)
        dataset = {
            "Train": {
                "Input": train[0].tolist(),
                "Output": train[1].tolist()
            }, 
            "Validation": {
                "Input": validation[0].tolist(),
                "Output": validation[1].tolist()
            }
        }
        return dataset
    
    def get_data3(self, n=2):
        x, y = [], []
        while len(x) < int(n / 2.):
            xx = np.random.rand(3)
            if xx[0] < 0.5:
                x.append(xx)
                y.append([1., 0.])
        while len(x) < n:
            xx = np.random.rand(3)
            if xx[0] > 0.5:
                x.append(xx)
                y.append([0., 1.])
        return np.array(x), np.array(y)
    
    def get_data(self, n=1, sigma=0.):
        def function(x):
            return np.sin(np.pi * x)
        
        x, y = [], []
        for i in np.linspace(0., 1., n):
            x.append([i])
            y.append([function(i) + np.random.normal(0., sigma)])
        return np.array(x), np.array(y)

    def save_dataset(self, data_path):
        logger.debug("Save a new dataset")
        with open(data_path, "wt") as f:
            json.dump(self.dataset, f, indent=4)


