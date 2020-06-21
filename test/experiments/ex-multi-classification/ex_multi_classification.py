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
    def get_data(n_data, sigma):
        x, y = [], []
        while len(x) < n_data / 6:
            x1 = np.random.rand()
            x2 = 0.5 + np.sqrt(0.5*0.5-(x1-0.5)**2)
            x.append([x1, x2])
            y.append([1, 0, 0])
        while len(x) < n_data / 3:
            x1 = np.random.rand()
            x2 = 0.5 - np.sqrt(0.5*0.5-(x1-0.5)**2)
            x.append([x1, x2])
            y.append([1, 0, 0])
        while len(x) < 2 * n_data / 3:
            x1 = np.random.uniform(low=0.1, high=0.7)
            x2 = 0.4 + 1.6 * np.sqrt(0.3*0.3-(x1-0.4)**2)
            x.append([x1, x2])
            y.append([0, 1, 0])
        while len(x) < n_data:
            x1 = np.random.uniform(low=0.3, high=0.9)
            x2 = 0.6 - 1.6 * np.sqrt(0.3*0.3-(x1-0.6)**2)
            x.append([x1, x2])
            y.append([0, 0, 1])
        return (x, y)

    '''
    def get_data(n_data, sigma):
        p = 0.6
        x, y = [], []
        while len(x) < n_data / 3:
            x1 = np.random.normal(loc=0.5, scale=0.05)
            x2 = np.random.normal(loc=0.75, scale=0.05)
            x.append([x1, x2])
            y.append([1, 0, 0])
        while len(x) < 2 * n_data / 3:
            x1 = np.random.rand()
            x.append([x1, p * x1])
            y.append([0, 1, 0])
        while len(x) < n_data:
            x1 = np.random.rand()
            x.append([x1, -p * x1 + p])
            y.append([0, 0, 1])
        return (x, y)

    def get_data(n_data, sigma):
        x, y = [], []
        while len(x) < n_data / 3:
            x1 = np.random.normal(loc=0.5, scale=sigma)
            x2 = np.random.normal(loc=0.75, scale=sigma)
            x.append([x1, x2])
            y.append([1, 0, 0])
        while len(x) < 2 * n_data / 3:
            x1 = np.random.normal(loc=0.25, scale=sigma)
            x2 = np.random.normal(loc=0.25, scale=sigma)
            x.append([x1, x2])
            y.append([0, 1, 0])
        while len(x) < n_data:
            x1 = np.random.normal(loc=0.75, scale=sigma)
            x2 = np.random.normal(loc=0.25, scale=sigma)
            x.append([x1, x2])
            y.append([0, 0, 1])
        return (x, y)
    '''

    data = get_data(config["N_data"], 0.05)
    dataset = {
        "Input": data[0],
        "Output": data[1]
    }
    with open(data_path, "wt") as f:
        json.dump(dataset, f, indent=4)

def create_model_file(config, model_path):
    alpha = np.zeros(shape=(config["Weights_division"], config["Output_dimension"]), dtype=np.float32)
    beta = np.zeros(shape=(config["Weights_division"], config["Input_dimension"], config["Input_dimension"]), dtype=np.float32)
    gamma = np.zeros(shape=(config["Weights_division"], config["Input_dimension"]), dtype=np.float32)
    while True:
        A = np.random.uniform(-1., 1., size=(config["Output_dimension"], config["Input_dimension"])).astype(np.float32)
        if np.where((A<0.01) & (A>-0.01), False, True).all():
            break
    model_data = {
            "Alpha": alpha.tolist(),
            "Beta": beta.tolist(),
            "Gamma": gamma.tolist(),
            "A": A.tolist()
    }
    with open(model_path, "wt") as f:
        json.dump(model_data, f, indent=4)

def setting_file(path):
    if not os.path.exists(path["Config_dir"]):
        os.makedirs(path["Config_dir"])
    if not os.path.exists(path["Data_dir"]):
        os.makedirs(path["Data_dir"])
    if not os.path.exists(path["Data_processed_dir"]):
        os.makedirs(path["Data_processed_dir"])
    if not os.path.exists(path["Model_dir"]):
        os.makedirs(path["Model_dir"])
    shutil.copy(path["Ex_config_path"], path["Config_path"])
    shutil.copy(path["Ex_data_path"], path["Data_path"])
    shutil.move(path["Ex_model_path"], path["Model_path"])

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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig_origin = plt.figure()
    ax_origin = fig_origin.add_subplot(111)
    fig_predict = plt.figure()
    ax_predict = fig_predict.add_subplot(111)
    x = np.array(dataset[0])
    y_true = np.argmax(dataset[1], 1)
    y_pred = np.argmax(dataset[2], 1)
    ax.scatter(x[np.where((y_pred==1) & (y_true==0))[0]][:, 0], x[np.where((y_pred==1) & (y_true==0))[0]][:, 1], s=10, c='#ffbb00', label=r'$F(\xi)=1,y(T;\xi)=2$')
    ax.scatter(x[np.where((y_pred==2) & (y_true==0))[0]][:, 0], x[np.where((y_pred==2) & (y_true==0))[0]][:, 1], s=10, c='#ff00bb', label=r'$F(\xi)=1,y(T;\xi)=3$')
    ax.scatter(x[np.where((y_pred==0) & (y_true==1))[0]][:, 0], x[np.where((y_pred==0) & (y_true==1))[0]][:, 1], s=10, c='#bbff00', label=r'$F(\xi)=2,y(T;\xi)=1$')
    ax.scatter(x[np.where((y_pred==2) & (y_true==1))[0]][:, 0], x[np.where((y_pred==2) & (y_true==1))[0]][:, 1], s=10, c='#00ffbb', label=r'$F(\xi)=2,y(T;\xi)=3$')
    ax.scatter(x[np.where((y_pred==0) & (y_true==2))[0]][:, 0], x[np.where((y_pred==0) & (y_true==2))[0]][:, 1], s=10, c='#bb00ff', label=r'$F(\xi)=3,y(T;\xi)=1$')
    ax.scatter(x[np.where((y_pred==1) & (y_true==2))[0]][:, 0], x[np.where((y_pred==1) & (y_true==2))[0]][:, 1], s=10, c='#00bbff', label=r'$F(\xi)=3,y(T;\xi)=2$')
    ax.scatter(x[np.where((y_pred==0) & (y_true==0))[0]][:, 0], x[np.where((y_pred==0) & (y_true==0))[0]][:, 1], s=10, c='#ff0000', label=r'$F(\xi)=1,y(T;\xi)=1$')
    ax.scatter(x[np.where((y_pred==1) & (y_true==1))[0]][:, 0], x[np.where((y_pred==1) & (y_true==1))[0]][:, 1], s=10, c='#00ff00', label=r'$F(\xi)=2,y(T;\xi)=2$')
    ax.scatter(x[np.where((y_pred==2) & (y_true==2))[0]][:, 0], x[np.where((y_pred==2) & (y_true==2))[0]][:, 1], s=10, c='#0000ff', label=r'$F(\xi)=3,y(T;\xi)=3$')
    ax_origin.scatter(x[np.where(y_true==0)[0]][:,0], x[np.where(y_true==0)[0]][:,1], s=10, c='#ff0000', label='1')
    ax_origin.scatter(x[np.where(y_true==1)[0]][:,0], x[np.where(y_true==1)[0]][:,1], s=10, c='#00ff00', label='2')
    ax_origin.scatter(x[np.where(y_true==2)[0]][:,0], x[np.where(y_true==2)[0]][:,1], s=10, c='#0000ff', label='3')
    ax_predict.scatter(x[np.where(y_pred==0)[0]][:,0], x[np.where(y_pred==0)[0]][:,1], s=10, c='#ff0000', label='1')
    ax_predict.scatter(x[np.where(y_pred==1)[0]][:,0], x[np.where(y_pred==1)[0]][:,1], s=10, c='#00ff00', label='2')
    ax_predict.scatter(x[np.where(y_pred==2)[0]][:,0], x[np.where(y_pred==2)[0]][:,1], s=10, c='#0000ff', label='3')
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
        "Input_dimension": 2,
        "Output_dimension": 3,
        "Maximum_time": 1.0,
        "Weights_division": 100,
        "Function_type": "relu",
        "Optimizer_type": "RMSprop",
        "Learning_rate": 0.01,
        "Momentum": 0.9,
        "Decay": 0.9,
        "Decay2": 0.999,
        "Epoch": 1000,
        "Batch_size": 10,
        "Test_size": 0.1,
        "Validation_size": 0.1,
        "Is_visualize": 1,
        "Is_accuracy": 1,
        "N_data": 6000
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
    ex_dir = os.path.join(os.path.join(test_dir, "experiments"), "ex-multi-classification")
    ex_data_path = os.path.join(ex_dir, "data.json")
    ex_config_path = os.path.join(ex_dir, "parameter.conf")
    ex_model_path = os.path.join(ex_dir, "model.json")
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
        "Ex_model_path": ex_model_path,
        "Ex_result_dir": ex_result_dir,
        "Data_predict_path": data_predict_path
    }

    create_config_file(config, ex_config_path)
    create_data_file(config, ex_data_path)
    create_model_file(config, ex_model_path)
    setting_file(path)
    subprocess.call(["python", program_path, "train"])
    files = os.listdir(model_dir)
    files_dir = [f for f in files if os.path.isdir(os.path.join(model_dir, f))]
    shutil.copy(os.path.join(os.path.join(model_dir, files_dir[0]), "model.json"), model_path)
    subprocess.call(["python", program_path, "predict"])
    copy_result(path)
    plot_predict(path)
    clear(path)