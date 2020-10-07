import os
import json
from datetime import datetime
from logging import getLogger
import numpy as np
from sklearn.model_selection import train_test_split

from config import Config

logger = getLogger(__name__)

def start(config: Config):
    return ModelAPI(config).start()

class ModelAPI:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = None
    
    def start(self):
        self.model = self.load_model()
        self.dataset = self.load_dataset()
        y_pred_train = self.model(self.dataset[0][0])
        y_pred_val = self.model(self.dataset[1][0])
        y_pred_test = self.model(self.dataset[2][0])
        #y_pred = self.model(self.dataset[0])
        self.save_data_predict(y_pred_train, y_pred_val, y_pred_test)
    
    def load_model(self):
        from model import NeuralODEModel
        model = NeuralODEModel(self.config)
        model.load(self.config.resource.model_path)
        return model
    
    def load_dataset(self):
        data_path = self.config.resource.data_path
        if os.path.exists(data_path):
            logger.debug("loading data from {}".format(data_path))
            with open(data_path, "rt") as f:
                datasets = json.load(f)
            x_train = datasets.get("Train", {}).get("Input")
            y_train = datasets.get("Train", {}).get("Output")
            x_val = datasets.get("Validation", {}).get("Input")
            y_val = datasets.get("Validation", {}).get("Output")
            x_test = datasets.get("Test", {}).get("Input")
            y_test = datasets.get("Test", {}).get("Output")
            if x_train is None or y_train is None:
                raise TypeError("Dataset does not exists in {}".format(data_path))
            if len(x_train[0]) != self.config.model.dim_in:
                raise ValueError("Input dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_in, len(x_train[0])))
            if len(y_train[0]) != self.config.model.dim_out:
                raise ValueError("Output dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_out, len(y_train[0])))
            if x_val is None or y_val is None or x_test is None or y_test is None:
                x_train, x_test, y_train, y_test = train_test_split(np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32), test_size=self.config.trainer.test_size)
                x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.config.trainer.validation_size)
            train = (np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32))
            validation = (np.array(x_val, dtype=np.float32), np.array(y_val, dtype=np.float32))
            test = (np.array(x_test, dtype=np.float32), np.array(y_test, dtype=np.float32))

            '''
            x = datasets.get("Input")
            y = datasets.get("Output")
            if x is None or y is None:
                raise TypeError("Dataset does not exists in {}".format(data_path))
            if len(x[0]) != self.config.model.dim_in:
                raise ValueError("Input dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_in, len(x[0])))
            if len(y[0]) != self.config.model.dim_out:
                raise ValueError("Output dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_out, len(y[0])))
            return (np.array(x, dtype=np.float32), np.array(y,dtype=np.float32))
            '''
            return (train, validation, test)
        else:
            raise FileNotFoundError("Dataset file can not loaded!")
    
    def save_data_predict(self, y_pred_train, y_pred_val, y_pred_test):
        rc = self.config.resource
        result_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(rc.result_dir, "result_predict_{}".format(result_id))
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "data_predict.json")
        data_predict = {
            "Train": {
                "Input": self.dataset[0][0].tolist(),
                "Output": y_pred_train.tolist()
            },
            "Validation": {
                "Input": self.dataset[1][0].tolist(),
                "Output": y_pred_val.tolist()
            },
            "Test": {
                "Input": self.dataset[2][0].tolist(),
                "Output": y_pred_test.tolist()
            }
        }
        logger.debug("save prediction data to {}".format(result_path))
        with open(result_path, "wt") as f:
            json.dump(data_predict, f, indent=4)