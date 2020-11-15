import os
import shutil
import json
import configparser
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_data_file(config, data_path):
    if os.path.exists(data_path):
        print("Data file already exists.")
        return
    
    def get_data(n_data, dim_in):
        x, y = [], []
        while len(x) < int(n_data / 2.):
            xx = np.random.rand(dim_in)
            if (xx[0]-0.5)**2 + (xx[1]-0.5)**2 < 0.3**2:
                x.append(xx.tolist())
                y.append([1.])
        while len(x) < n_data:
            xx = np.random.rand(dim_in)
            if (xx[0]-0.5)**2 + (x[1]-0.5)**2 >0.3**2:
                x.append(xx.tolist())
                y.append([0.])
        return (x, y)
    
    print("Create a dataset file (circle)")
    x, y = get_data(config["N_data"], config["Input_dimension"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config["Test_size"])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=config["Validation_size"])
    dataset = {
        "Train": {
            "Input": x_train,
            "Output": y_train
        },
        "Validation": {
            "Input": x_val,
            "Output": y_val
        },
        "Test": {
            "Input": x_test,
            "Output": y_test
        }
    }
    with open(data_path, "wt") as f:
        json.dump(dataset, f, indent=4)
    
    print("The number of data")
    print("n_train:{},  n_val:{}  n_test:{}".format(len(x_train), len(x_val), len(x_test)))
    print()

def create_config_file(config, config_path):
    if os.path.exists(config_path):
        print("Config file already exists.")
        return
    
    config_parser = configparser.ConfigParser()
    config_parser["MODEL"] = {
        "Input_dimension": config["Input_dimension"],
        "Output_dimension": config["Output_dimension"],
        "Maximum_time": config["Maximum_time"],
        "Weights_division": config["Weights_division"],
        "Function_type": config["Function_type"]
    }
    config_parser["TRAINER"] = {
        "Optimizer_type": config["Optimizer_type"],
        "Learning_rate": config["Learning_rate"],
        "Momentum": config["Momentum"],
        "Decay": config["Decay"],
        "Decay2": config["Decay2"],
        "Epoch": config["Epoch"],
        "Batch_size": config["Batch_size"],
        "Test_size": config["Test_size"],
        "Validation_size": config["Validation_size"],
        "Is_visualize": config["Is_visualize"],
        "Is_accuracy": config["Is_accuracy"]
    }
    with open(config_path, "wt") as f:
        config_parser.write(f)
    
    print("Config:")
    for k, v in config.items():
        print("{}: {}".format(k, v))
    print()

def setting_file(path):
    if not os.path.exists(path["Config_dir"]):
        os.makedirs(path["Config_dir"])
    if not os.path.exists(path["Data_dir"]):
        os.makedirs(path["Data_dir"])
    if not os.path.exists(path["Data_processed_dir"]):
        os.makedirs(path["Data_processed_dir"])
    shutil.copy(path["Ex_config_path"], path["Config_path"])
    shutil.copy(path["Ex_data_path"], path["Data_path"])

def copy_result(path):
    if not os.path.exists(path["Ex_result_dir"]):
        os.makedirs(path["Ex_result_dir"])
    for p in os.listdir(path["Result_dir"]):
        result_dir = os.path.join(path["Result_dir"], p)
        for pp in os.listdir(result_dir):
            result_path = os.path.join(result_dir, pp)
            shutil.move(result_path, path["Ex_result_dir"])
    shutil.move(path["Model_path"], path["Ex_dir"])
    shutil.move(path["Log_path"], path["Ex_dir"])

def clear(path):
    shutil.rmtree(path["Data_dir"])
    shutil.rmtree(path["Config_dir"])
    shutil.rmtree(path["Model_dir"])
    shutil.rmtree(path["Log_dir"])

def plot_training_data(data_path, fig_dir):
    with open(data_path, "rt") as f:
        dataset = json.load(f)
    x_train = np.array(dataset.get("Train").get("Input"), dtype=np.float32)
    y_train = np.array(dataset.get("Train").get("Output"), dtype=np.float32)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_train[np.where(y_train==0)[0]][:, 0], x_train[np.where(y_train==0)[0]][:, 1], s=5, c="blue", label=r'$F(\xi)=0$')
    ax.scatter(x_train[np.where(y_train==1)[0]][:, 0], x_train[np.where(y_train==1)[0]][:, 1], s=5, c="orange", label=r'$F(\xi)=1$')
    ax.set_xlabel(r'$\xi_1$')
    ax.set_ylabel(r'$\xi_2$')
    ax.set_aspect("equal")
    lgnd = ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=2)
    fig.savefig(os.path.join(fig_dir, "training_data.png"), bbox_extra_artists=(lgnd,), bbox_inches='tight')

def plot_data_predict(data_path, data_pred_path, fig_dir):
    with open(data_path, "rt") as f:
        dataset = json.load(f)
    x_val = np.array(dataset.get("Validation").get("Input"), dtype=np.float32)
    y_val = np.array(dataset.get("Validation").get("Output"), dtype=np.float32)
    with open(data_pred_path, "rt") as f:
        dataset = json.load(f)
    x_pred = np.array(dataset.get("Validation").get("Input"), dtype=np.float32)
    y_pred = np.array(dataset.get("Validation").get("Output"), dtype=np.float32)
    y_pred = np.where(y_pred<0.5, 0, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_val[np.where((y_val==0) & (y_pred==0))[0]][:, 0], x_val[np.where((y_val==0) & (y_pred==0))[0]][:, 1], s=10, c="#0000ff", label=r'$F(\xi)=0,y(T;\xi)<0.5$')
    ax.scatter(x_val[np.where((y_val==1) & (y_pred==1))[0]][:, 0], x_val[np.where((y_val==1) & (y_pred==1))[0]][:, 1], s=10, c="#ff0000", label=r'$F(\xi)=1,y(T;\xi)\geq0.5$')
    ax.scatter(x_val[np.where((y_val==0) & (y_pred==1))[0]][:, 0], x_val[np.where((y_val==0) & (y_pred==1))[0]][:, 1], s=10, c="#bbbbff", label=r'$F(\xi)=0,y(T;\xi)\geq0.5$')
    ax.scatter(x_val[np.where((y_val==1) & (y_pred==0))[0]][:, 0], x_val[np.where((y_val==1) & (y_pred==0))[0]][:, 1], s=10, c="#ffbbbb", label=r'$F(\xi)=1,y(T;\xi)<0.5$')
    ax.set_xlabel(r'$\xi_1$')
    ax.set_ylabel(r'$\xi_2$')
    ax.set_aspect("equal")
    lgnd = ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=2)
    fig.savefig(os.path.join(fig_dir, "data_predict.png"),  bbox_extra_artists=(lgnd,), bbox_inches='tight')

def plot_alpha(model_path, fig_dir):
    with open(model_path, "rt") as f:
        model = json.load(f)
    alpha = np.array(model.get("Alpha"), dtype=np.float32)
    t = np.linspace(0., 1., len(alpha))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, alpha[:, 0], label=r'$\alpha(t)$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\alpha(t)$')
    ax.legend()
    fig.savefig(os.path.join(fig_dir, "alpha.png"))

def plot_beta(model_path, fig_dir):
    with open(model_path, "rt") as f:
        model = json.load(f)
    beta = np.array(model.get("Beta"), dtype=np.float32)
    t = np.linspace(0., 1., len(beta))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, beta[:, 0, 0], label=r'$\beta_{00}(t)$')
    ax.plot(t, beta[:, 0, 1], label=r'$\beta_{01}(t)$')
    ax.plot(t, beta[:, 1, 0], label=r'$\beta_{10}(t)$')
    ax.plot(t, beta[:, 1, 1], label=r'$\beta_{11}(t)$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\beta(t)$')
    ax.legend()
    fig.savefig(os.path.join(fig_dir, "beta.png"))

def plot_gamma(model_path, fig_dir):
    with open(model_path, "rt") as f:
        model = json.load(f)
    gamma = np.array(model.get("Gamma"), dtype=np.float32)
    t = np.linspace(0., 1., len(gamma))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, gamma[:, 0], label=r'$\gamma_{0}(t)$')
    ax.plot(t, gamma[:, 1], label=r'$\gamma_{1}(t)$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\gamma(t)$')
    ax.legend()
    fig.savefig(os.path.join(fig_dir, "gamma.png"))

def plot_parameters(model_path, fig_dir):
    with open(model_path, "rt") as f:
        model = json.load(f)
    alpha = np.array(model.get("Alpha"), dtype=np.float32)
    beta = np.array(model.get("Beta"), dtype=np.float32)
    gamma = np.array(model.get("Gamma"), dtype=np.float32)
    t = np.linspace(0., 1., len(alpha))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, alpha[:, 0], label=r'$\alpha(t)$')
    ax.plot(t, beta[:, 0, 0], label=r'$\beta_{00}(t)$')
    ax.plot(t, beta[:, 0, 1], label=r'$\beta_{01}(t)$')
    ax.plot(t, beta[:, 1, 0], label=r'$\beta_{10}(t)$')
    ax.plot(t, beta[:, 1, 1], label=r'$\beta_{11}(t)$')
    ax.plot(t, gamma[:, 0], label=r'$\gamma_{0}(t)$')
    ax.plot(t, gamma[:, 1], label=r'$\gamma_{1}(t)$')
    ax.set_xlabel(r'$t$')
    ax.legend()
    fig.savefig(os.path.join(fig_dir, "parameters.png"))

    plot_alpha(model_path, fig_dir)
    plot_beta(model_path, fig_dir)
    plot_gamma(model_path, fig_dir)

def plot_loss(loss_path, fig_dir):
    df = pd.read_csv(loss_path)
    epoch = df["epoch"].values
    loss = df["loss_train"].values
    loss_val = df["loss_val"].values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epoch, loss, label='Training data')
    ax.plot(epoch, loss_val, label='Validation data')
    ax.legend()
    ax.set_xlabel(r'$\nu$')
    ax.set_ylabel(r'$E$')
    fig.savefig(os.path.join(fig_dir, "loss.png"))

def plot_acc(loss_path, fig_dir):
    df = pd.read_csv(loss_path)
    epoch = df["epoch"].values
    acc = df["accuracy_train"].values
    acc_val = df["accuracy_val"].values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epoch, acc, label='Training data')
    ax.plot(epoch, acc_val, label='Validation data')
    ax.legend()
    ax.set_xlabel(r'$\nu$')
    ax.set_ylabel(r'$\mathrm{Accuracy}$')
    fig.savefig(os.path.join(fig_dir), "accuracy.png")


def plot_result(path):
    if not os.path.exists(path["Ex_fig_dir"]):
        os.makedirs(path["Ex_fig_dir"])
    
    plot_training_data(path["Ex_data_path"], path["Ex_fig_dir"])
    plot_data_predict(path["Ex_data_path"], path["Ex_data_predict_path"], path["Ex_fig_dir"])
    plot_parameters(path["Ex_model_path"], path["Ex_fig_dir"])
    plot_loss(path["Ex_loss_path"], path["Ex_fig_dir"])
    plot_acc(path["Ex_loss_path"], path["Ex_fig_dir"])

    


def main():
    config = {
        "Input_dimension": 2,
        "Output_dimension": 1,
        "Maximum_time": 1.,
        "Weights_division": 100,
        "Function_type": "sigmoid", # sigmoid, relu
        "Optimizer_type": "SGD",
        "Learning_rate": 0.01,
        "Momentum": 0.9,
        "Decay": 0.99,
        "Decay2": 0.999,
        "Epoch": 5, #20000
        "Batch_size": 8,
        "Test_size": 2000,
        "Validation_size": 2000,
        "Is_visualize": 1,
        "Is_accuracy": 1,
        "N_data": 14000
    }

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
    data_dir = os.path.join(project_dir, "data")
    data_processed_dir = os.path.join(data_dir, "processed")
    data_path = os.path.join(data_processed_dir, "data.json")
    config_dir = os.path.join(project_dir, "config")
    config_path = os.path.join(config_dir, "parameter.conf")
    model_dir = os.path.join(project_dir, "model")
    model_path = os.path.join(model_dir, "model.json")
    log_dir = os.path.join(project_dir, "logs")
    log_path = os.path.join(log_dir, "main.log")
    result_dir = os.path.join(data_dir, "result")
    program_path =os.path.join(os.path.join(project_dir, "src"), "run.py")
    ex_binary_classification_dir = os.path.join(os.path.join(os.path.join(project_dir, "test"), "experiments"), "ex-binary-classification")
    ex_dir = os.path.join(os.path.join(ex_binary_classification_dir, "ex-activation-function"), "ex-sigmoid")
    # ex_dir = os.path.join(os.path.join(ex_binary_classification_dir, "ex-activation-function"), "ex-relu")
    ex_data_path = os.path.join(ex_dir, "data.json")
    ex_config_path = os.path.join(ex_dir, "parameter.conf")
    ex_model_path = os.path.join(ex_dir, "model_json")
    ex_result_dir = os.path.join(ex_dir, "result")
    ex_data_predict_path = os.path.join(ex_result_dir, "data_predict.json")
    ex_loss_path = os.path.join(ex_result_dir, "learning_curve.csv")
    ex_fig_dir = os.path.join(ex_dir, "figure")

    path = {
        "Project_dir": project_dir, 
        "Data_dir": data_dir,
        "Data_processed_dir": data_processed_dir,
        "Data_path": data_path,
        "Config_dir": config_dir,
        "Config_path": config_path,
        "Model_dir": model_dir,
        "Model_path": model_path,
        "Log_dir": log_dir,
        "Log_path": log_path,
        "Result_dir": result_dir,
        "Program_path": program_path,
        "Ex_binary_classification_dir": ex_binary_classification_dir,
        "Ex_dir": ex_dir,
        "Ex_data_path": ex_data_path,
        "Ex_config_path": ex_config_path,
        "Ex_model_path": ex_model_path,
        "Ex_result_dir": ex_result_dir,
        "Ex_data_predict_path": ex_data_predict_path,
        "Ex_loss_path": ex_loss_path,
        "Ex_fig_dir": ex_fig_dir
    }

    create_data_file(config, ex_data_path)
    create_config_file(config, ex_config_path)
    setting_file(path)
    subprocess.call(["python", program_path, "train"])
    shutil.copy(os.path.join(os.path.join(model_dir, os.listdir(model_dir)[0]), "model.json"), model_path)
    subprocess.call(["python", program_path, "predict"])
    copy_result(path)
    clear(path)
    plot_result(path)


if __name__ == "__main__":
    main()