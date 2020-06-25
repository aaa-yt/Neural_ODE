import os
import shutil
import json
import configparser
import subprocess
import numpy as np
import matplotlib.pyplot as plt

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
    def function(x):
        return np.sin(2. * np.pi * x)
    
    x, y = [], []
    for i in np.linspace(0., 1., config["N_data"]):
        x.append([i])
        y.append([function(i) + np.random.normal(0., config["Data_variance"])])
    dataset = {
        "Input": x,
        "Output": y
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
    dataset = load_dataset(path["Data_path"], path["Data_predict_path"])
    plt.plot(dataset[0], dataset[1], label="Train")
    plt.plot(dataset[0], dataset[2], label="Predict")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Regression problem for $y=\sin4\pi x$')
    plt.legend()
    plt.savefig(os.path.join(path["Ex_result_dir"], "data_predict.png"))

def load_dataset(data_path, data_predict_path):
    with open(data_path, "rt") as f:
        data = json.load(f)
    with open(data_predict_path, "rt") as f:
        data_predict = json.load(f)
    x = data.get("Input")
    y = data.get("Output")
    y_pred = data_predict.get("Output")
    return (x, y, y_pred)

def clear(path):
    shutil.rmtree(path["Data_dir"])
    shutil.rmtree(path["Config_dir"])
    shutil.rmtree(path["Model_dir"])

if __name__ == "__main__":
    config = {
        "Input_dimension": 1,
        "Output_dimension": 1,
        "Maximum_time": 1.0,
        "Weights_division": 90,
        "Function_type": "sigmoid",
        "Optimizer_type": "SGD",
        "Learning_rate": 0.01,
        "Momentum": 0.9,
        "Decay": 0.99,
        "Decay2": 0.999,
        "Epoch": 10000,
        "Batch_size": 32,
        "Test_size": 0.1,
        "Validation_size": 0.2,
        "Is_visualize": 1,
        "Is_accuracy": 0,
        "N_data": 10000,
        "Data_variance": 0.
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
    ex_dir = os.path.join(os.path.join(test_dir, "experiments"), "ex-regression")
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