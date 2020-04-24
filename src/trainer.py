import os
import json
import time
import csv
from datetime import datetime
from logging import getLogger
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config
from lib import optimizers
from model import mean_square_error, cross_entropy, accuracy
from visualize import Visualize

logger = getLogger(__name__)

def start(config: Config):
    return Trainer(config).start()


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.visualize = Visualize(config)
    
    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        tc = self.config.trainer
        self.compile_model()
        self.dataset = self.load_dataset()
        self.fit(x=self.dataset[0][0], y=self.dataset[0][1], epochs=tc.epoch, batch_size=tc.batch_size, validation_data=self.dataset[1], is_visualize=tc.is_visualize, is_accuracy=tc.is_accuracy)
        self.evaluate(self.dataset[2][0], self.dataset[2][1])
        self.save_model()
        self.save_result()

    def compile_model(self):
        self.optimizer = optimizers.get(self.config)
        self.loss = mean_square_error
        self.accuracy = accuracy
    
    def fit(self, x=None, y=None, epochs=1, batch_size=1, validation_data=None, is_shuffle=True, is_visualize=False, is_accuracy=False):
        if x is None or y is None:
            raise ValueError("There is no fitting data")
        n_train = len(x)
        self.losses = []
        if validation_data is not None: self.losses_val = []
        if is_accuracy: self.accuracies = []
        if validation_data is not None and is_accuracy: self.accuracies_val = []

        logger.info("training start")
        start_time = time.time()
        for epoch in range(epochs):
            if is_shuffle:
                x, y = shuffle(x, y)
            with tqdm(range(0, n_train, batch_size), desc="[Epoch: {}]".format(epoch+1)) as pbar:
                for i, ch in enumerate(pbar):
                    self.model.params = self.optimizer(self.model.params, self.model.gradient(x[i:i+batch_size], y[i:i+batch_size]))
                    #error = self.loss(self.model(x[0:i+1]), y[0:i+1])
                    #pbar.set_postfix({"loss": error})
            y_pred = self.model(x)
            error = self.loss(y_pred, y)
            self.losses.append(error)
            if validation_data is None:
                if not is_accuracy:
                    message = "Epoch:{}  Training loss:{:.5f}".format(epoch+1, error)
                    if is_visualize: self.visualize.plot_realtime(self.model.t, self.model.params, [self.losses])
                else:
                    accuracy = self.accuracy(y_pred, y)
                    self.accuracies.append(accuracy)
                    message = "Epoch:{}  Training loss:{:.5f}  Training accuracy:{:.5f}".format(epoch+1, error, accuracy)
                    if is_visualize: self.visualize.plot_realtime(self.model.t, self.model.params, [self.losses], [self.accuracies])
            else:
                y_val_pred = self.model(validation_data[0])
                error_val = self.loss(y_val_pred, validation_data[1])
                self.losses_val.append(error_val)
                if not is_accuracy:
                    message = "Epoch:{}  Training loss:{:.5f}  Validation loss:{:.5f}".format(epoch+1, error, error_val)
                    if is_visualize: self.visualize.plot_realtime(self.model.t, self.model.params, [self.losses, self.losses_val])
                else:
                    accuracy = self.accuracy(y_pred, y)
                    self.accuracies.append(accuracy)
                    accuracy_val = self.accuracy(y_val_pred, validation_data[1])
                    self.accuracies_val.append(accuracy_val)
                    message = "Epoch:{}  Training loss:{:.5f}  Validation loss:{:.5f}  Training accuracy:{:.5f}  Validation accuracy:{:.5f}".format(epoch+1, error, error_val, accuracy, accuracy_val)
                    if is_visualize: self.visualize.plot_realtime(self.model.t, self.model.params, [self.losses, self.losses_val], [self.accuracies, self.accuracies_val])
            logger.info(message)
        interval = time.time() - start_time
        logger.info("end of training")
        logger.info("time: {}".format(interval))
        logger.info(message)

    def evaluate(self, x, y):
        y_pred = self.model(x)
        error = self.loss(y_pred, y)
        if self.config.trainer.is_accuracy:
            accuracy = self.accuracy(y_pred, y)
            message = "Test loss:{}  Test accuracy:{}".format(error, accuracy)
        else:
            message = "Test loss:{}".format(error)
        logger.info(message)

    def load_model(self):
        from model import NeuralODEModel
        model = NeuralODEModel(self.config)
        model.load(self.config.resource.model_path)
        return model
    
    def save_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join(rc.model_dir, "model_{}".format(model_id))
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, "parameter.conf")
        model_path = os.path.join(model_dir, "model.json")
        self.model.save(config_path, model_path)

    def save_result(self):
        rc = self.config.resource
        tc = self.config.trainer
        result_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(rc.result_dir, "result_train_{}".format(result_id))
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "learning_curve.csv")
        e = [i for i in range(1, tc.epoch+1)]
        try:
            self.visualize.save_plot_loss([self.losses, self.losses_val], xlabel="Epoch", ylabel="Loss", title="Loss", save_file=os.path.join(result_dir, "loss.png"))
            if tc.is_accuracy:
                result_csv = [e, self.losses, self.losses_val, self.accuracies, self.accuracies_val]
                columns = ["epoch", "loss_train", "loss_val", "accuracy_train", "accuracy_val"]
                self.visualize.save_plot_accuracy([self.accuracies, self.accuracies_val], xlabel="Epoch", ylabel="Accuracy", title="Accuracy", save_file=os.path.join(result_dir, "accuracy.png"))
            else:
                result_csv = [e, self.losses, self.losses_val]
                columns = ["epoch", "loss_train", "loss_val"]
        except AttributeError:
            self.visualize.save_plot_loss([self.losses], xlabel="Epoch", ylabel="Loss", title="Loss", save_file=os.path.join(result_dir, "loss.png"))
            if tc.is_accuracy:
                result_csv = [e, self.losses, self.accuracies]
                columns = ["epoch", "loss_train", "accuracy_train"]
                self.visualize.save_plot_accuracy([self.accuracies], xlabel="Epoch", ylabel="Accuracy", title="Accuracy", save_file=os.path.join(result_dir, "accuracy.png"))
            else:
                result_csv = [e, self.losses]
                columns = ["epoch", "loss_train"]
        logger.debug("save result to {}".format(result_path))
        with open(result_path, "wt") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(list(zip(*result_csv)))
        save_params_path = [os.path.join(result_dir, "alpha.png"), os.path.join(result_dir, "beta.png"), os.path.join(result_dir, "gamma.png"), os.path.join(result_dir, "params.png")]
        self.visualize.save_plot_params(self.model.t, self.model.params, save_file=save_params_path)

 
    def load_dataset(self):
        data_path = self.config.resource.data_path
        if os.path.exists(data_path):
            logger.debug("loading data from {}".format(data_path))
            with open(data_path, "rt") as f:
                datasets = json.load(f)
            x = datasets.get("Input")
            y = datasets.get("Output")
            if x is None or y is None:
                raise TypeError("Dataset does not exists in {}".format(data_path))
            if len(x[0]) != self.config.model.dim_in:
                raise ValueError("Input dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_in, len(x[0])))
            if len(y[0]) != self.config.model.dim_out:
                raise ValueError("Output dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_out, len(y[0])))
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.config.trainer.test_size)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.config.trainer.validation_size)
            train = (x_train, y_train)
            validation = (x_val, y_val)
            test = (x_test, y_test)
            return (train, validation, test)
        else:
            raise FileNotFoundError("Dataset file can not loaded!")
