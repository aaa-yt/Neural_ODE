import os
import shutil
import json
import configparser
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_config_file(config, config_path):
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

def create_data_file(config, data_path):
    if os.path.exists(data_path): return 
    def get_data(n_data):
        x, y = [], []
        while len(x) < int(n_data / 2):
            xx = np.random.rand(config["Input_dimension"])
            if ((xx[0] - 0.5)**2 + (xx[1] - 0.5)**2) < 0.3 * 0.3:
                x.append(xx.tolist())
                y.append([0.])
        while len(x) < n_data:
            xx = np.random.rand(config["Input_dimension"])
            if ((xx[0] - 0.5)**2 + (xx[1] - 0.5)**2) > 0.3 * 0.3:
                x.append(xx.tolist())
                y.append([1.])
        return (x, y)
        
    data = get_data(config["N_data"])
    x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=config["Test_size"])
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

def plot_predict(path):
    train, validation = load_dataset(path["Data_path"], path["Data_predict_path"])
    x_train = train[0]
    y_train = train[1]
    x_val = validation[0]
    y_val = validation[1]
    y_pred = validation[2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_val[:,0], x_val[:,1], s=5, c='white', edgecolor='black', label='Validation data')
    ax.scatter(x_train[np.where(y_train==1)[0]][:,0], x_train[np.where(y_train==1)[0]][:,1], s=10, c='red', label=r'$F(\xi)=1$')
    ax.scatter(x_train[np.where(y_train==0)[0]][:,0], x_train[np.where(y_train==0)[0]][:,1], s=10, c='blue', label=r'$F(\xi)=0$')
    ax.set_xlabel(r'$\xi_1$')
    ax.set_ylabel(r'$\xi_2$')
    ax.set_aspect('equal')
    lgnd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.savefig(os.path.join(path["Ex_result_dir"], "training-data.png"), bbox_extra_artists=(lgnd,), bbox_inches='tight')

    fig_result = plt.figure()
    ax_result = fig_result.add_subplot(111)
    ax_result.scatter(x_val[np.where((y_val==1) & (y_pred==1))[0]][:,0], x_val[np.where((y_val==1) & (y_pred==1))[0]][:,1], s=10, c='#ff0000', label=r'$F(\xi)=1,y(T;\xi)\geq1$')
    ax_result.scatter(x_val[np.where((y_val==0) & (y_pred==0))[0]][:,0], x_val[np.where((y_val==0) & (y_pred==0))[0]][:,1], s=10, c='#0000ff', label=r'$F(\xi)=0,y(T;\xi)\geq0$')
    ax_result.scatter(x_val[np.where((y_val==0) & (y_pred==1))[0]][:,0], x_val[np.where((y_val==0) & (y_pred==1))[0]][:,1], s=10, c='#ffbbbb', label=r'$F(\xi)=0,y(T;\xi)\geq1$')
    ax_result.scatter(x_val[np.where((y_val==1) & (y_pred==0))[0]][:,0], x_val[np.where((y_val==1) & (y_pred==0))[0]][:,1], s=10, c='#bbbbff', label=r'$F(\xi)=1,y(T;\xi)\geq0$')
    ax_result.set_xlabel(r'$\xi_1$')
    ax_result.set_ylabel(r'$\xi_2$')
    ax.set_aspect('equal')
    lgnd_result = ax_result.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig_result.savefig(os.path.join(path["Ex_result_dir"], "predict.png"), bbox_extra_artists=(lgnd_result,), bbox_inches='tight')


def plot_predict_(path):
    dataset = load_dataset(path["Data_path"], path["Data_predict_path"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig_origin = plt.figure()
    ax_origin = fig_origin.add_subplot(111)
    fig_predict = plt.figure()
    ax_predict = fig_predict.add_subplot(111)
    x = np.array(dataset[0])
    y_true = np.array(dataset[1])[:,0]
    y_pred = np.where(np.array(dataset[2])<0.5, 0, 1)[:,0]
    ax.scatter(x[np.where((y_pred==1) & (y_true==1))[0]][:, 0], x[np.where((y_pred==1) & (y_true==1))[0]][:, 1], s=10, c='#ff0000', label=r'$F(\xi)=1,y(T;\xi)\geq0.5$')
    ax.scatter(x[np.where((y_pred==0) & (y_true==0))[0]][:, 0], x[np.where((y_pred==0) & (y_true==0))[0]][:, 1], s=10, c='#0000ff', label=r'$F(\xi)=0,y(T;\xi)<0.5$')
    ax.scatter(x[np.where((y_pred==0) & (y_true==1))[0]][:, 0], x[np.where((y_pred==0) & (y_true==1))[0]][:, 1], s=10, c='#ffbbbb', label=r'$F(\xi)=1,y(T;\xi)<0.5$')
    ax.scatter(x[np.where((y_pred==1) & (y_true==0))[0]][:, 0], x[np.where((y_pred==1) & (y_true==0))[0]][:, 1], s=10, c='#bbbbff', label=r'$F(\xi)=0,y(T;\xi)\geq0.5$')
    ax_origin.scatter(x[np.where(y_true==1)[0]][:,0], x[np.where(y_true==1)[0]][:,1], s=10, c='#ff0000', label='1')
    ax_origin.scatter(x[np.where(y_true==0)[0]][:,0], x[np.where(y_true==0)[0]][:,1], s=10, c='#0000ff', label='0')
    ax_predict.scatter(x[np.where(y_pred==1)[0]][:,0], x[np.where(y_pred==1)[0]][:,1], s=10, c='#ff0000', label='1')
    ax_predict.scatter(x[np.where(y_pred==0)[0]][:,0], x[np.where(y_pred==0)[0]][:,1], s=10, c='#0000ff', label='0')
    ax.set_xlabel(r'$\xi_1$')
    ax.set_ylabel(r'$\xi_2$')
    ax.set_aspect('equal')
    ax_origin.set_xlabel(r'$\xi_1$')
    ax_origin.set_ylabel(r'$\xi_2$')
    ax_origin.set_aspect('equal')
    ax_origin.legend()
    ax_predict.set_xlabel(r'$\xi_1$')
    ax_predict.set_ylabel(r'$\xi_2$')
    ax_predict.set_aspect('equal')
    ax_predict.legend()
    lgnd = ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=2)
    fig.savefig(os.path.join(path["Ex_result_dir"], "data.png"), bbox_extra_artists=(lgnd,), bbox_inches='tight')
    fig_origin.savefig(os.path.join(path["Ex_result_dir"], "data_origin.png"))
    fig_predict.savefig(os.path.join(path["Ex_result_dir"], "data_predict.png"))



def load_dataset(data_path, data_predict_path):
    with open(data_path, "rt") as f:
        data = json.load(f)
    with open(data_predict_path, "rt") as f:
        data_predict = json.load(f)
    x_train = np.array(data.get("Train").get("Input"))
    y_train = np.array(data.get("Train").get("Output"))
    y_pred_train = np.array(data_predict.get("Train").get("Output"))
    x_val = np.array(data.get("Validation").get("Input"))
    y_val = np.array(data.get("Validation").get("Output"))
    y_pred_val = np.array(data_predict.get("Validation").get("Output"))
    train = (x_train, y_train, y_pred_train)
    validation = (x_val, y_val, y_pred_val)
    return (train, validation)

def clear(path):
    shutil.rmtree(path["Data_dir"])
    shutil.rmtree(path["Config_dir"])
    shutil.rmtree(path["Model_dir"])

if __name__ == "__main__":
    config = {
        "Input_dimension": 2,
        "Output_dimension": 1,
        "Maximum_time": 1.0,
        "Weights_division": 100,
        "Function_type": "sigmoid",
        "Optimizer_type": "SGD",
        "Learning_rate": 0.01,
        "Momentum": 0.9,
        "Decay": 0.9,
        "Decay2": 0.999,
        "Epoch": 10000,
        "Batch_size": 10,
        "Test_size": 0.2,
        "Validation_size": 0.2,
        "Is_visualize": 1,
        "Is_accuracy": 1,
        "N_data": 15625
    }

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_dir, "data")
    data_processed_dir = os.path.join(data_dir, "processed")
    data_path = os.path.join(data_processed_dir, "data.json")
    config_dir = os.path.join(project_dir, "config")
    config_path = os.path.join(config_dir, "parameter.conf")
    model_dir = os.path.join(project_dir, "model")
    model_path = os.path.join(model_dir, "model.json")
    result_dir = os.path.join(data_dir, "result")
    program_path = os.path.join(os.path.join(project_dir, "src"), "run.py")
    test_dir = os.path.join(project_dir, "test")
    ex_dir = os.path.join(os.path.join(test_dir, "experiments"), "ex-binary-classification")
    ex_data_path = os.path.join(ex_dir, "data.json")
    ex_config_path = os.path.join(ex_dir, "parameter.conf")
    ex_result_dir = os.path.join(ex_dir, "result")
    data_predict_path = os.path.join(ex_result_dir, "data_predict.json")

    path = {
        "Project_dir": project_dir,
        "Data_dir": data_dir,
        "Data_processed_dir": data_processed_dir,
        "Data_path": data_path,
        "Config_dir": config_dir,
        "Config_path": config_path,
        "Model_dir": model_dir,
        "Model_path": model_path,
        "Result_dir": result_dir,
        "Program_path": program_path,
        "Test_dir": test_dir,
        "Ex_dir": ex_dir,
        "Ex_data_path": ex_data_path,
        "Ex_config_path": ex_config_path,
        "Ex_result_dir": ex_result_dir,
        "Data_predict_path": data_predict_path
    }

    create_config_file(config, ex_config_path)
    create_data_file(config, ex_data_path)
    setting_file(path)
    subprocess.call(["python", program_path, "train"])
    shutil.copy(os.path.join(os.path.join(model_dir, os.listdir(model_dir)[0]), "model.json"), model_path)
    subprocess.call(["python", program_path, "predict"])
    copy_result(path)
    plot_predict(path)
    clear(path)