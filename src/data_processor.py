import os
import json
from logging import getLogger
from config import Config
import numpy as np

logger = getLogger(__name__)

def start(config: Config):
    return DataCreator(config).start()

class DataCreator:
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
        n_data = 5000
        sigma = 0.
        data = self.get_data(n_data, sigma)
        #data = self.get_data53(n_data)
        dataset = {
            "Input": data[0].tolist(),
            "Output": data[1].tolist()
        }
        return dataset
    
    def get_data(self, n_data, sigma):
        def function(x):
            return np.sin(np.pi * x)
        
        x, y = [], []
        for i in np.linspace(0., 1., n_data):
            x.append([i])
            y.append([function(i) + np.random.normal(0., sigma)])
        return np.array(x), np.array(y)

    def get_data53(self, n_data):
        x, y = [], []
        while len(x) < int(n_data / 3.):
            xx = np.random.rand(5)
            if xx[0] < 0.3:
                x.append(xx)
                y.append([1, 0, 0])
        while len(x) < int(2 * n_data / 3.):
            xx = np.random.rand(5)
            if xx[0] > 0.3 and xx[0] < 0.7:
                x.append(xx)
                y.append([0, 1, 0])
        while len(x) < n_data:
            xx = np.random.rand(5)
            if xx[0] > 0.7:
                x.append(xx)
                y.append([0, 0, 1])
        return np.array(x), np.array(y)

    def save_dataset(self, data_path):
        logger.debug("Save a new dataset")
        with open(data_path, "wt") as f:
            json.dump(self.dataset, f, indent=4)