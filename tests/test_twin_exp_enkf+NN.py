import os
# import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from openda.algorithms.ensemble_kalman import kalman_algorithm
from openda.algorithms.ensemble_kalman import no_filter
from openda.algorithms.GenericEnsembleKalmanFilter import GenericEnsembleKalmanFilter
from openda.algorithms.PINN import PINN
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
from openda.observers.pandas_observer import PandasObserver

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    warnings.warn('CUDA is not available, so PINN will be trained using CPU instead of GPU, which can be very slow. Please consider installing CUDA.')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data(csv):
    df = pd.read_csv(csv, delimiter=';', header=None)
    train = df.sample(frac=0.9)
    test = df.drop(train.index)

    x_train = torch.tensor(train.iloc[:,:-1].values, dtype=torch.float32)
    y_train = torch.tensor(train.iloc[:,-1].values, dtype=torch.float32).view(-1, 1)

    x_test = torch.tensor(test.iloc[:,:-1].values, dtype=torch.float32)
    y_test = torch.tensor(test.iloc[:,-1].values, dtype=torch.float32).view(-1, 1)

    return [x_train, y_train, x_test, y_test]

def setup_and_train_NN(device, data, enkf):
    layers =  [44, 10, 10, 10, 10, 1]
    model = PINN(device, layers, enkf, data)
    model.to(device)
    model.load_state_dict(torch.load('tests/PINNs/PINN.pth'))

    # optimizer = torch.optim.Adagrad(model.parameters())

    # start_time = time.time()
    # epochs, losses, val_losses = model.train_model(optimizer, n_epochs=150, batch_size=32)
    # elapsed = time.time() - start_time
    # print(f'Training time: {elapsed:.2f}')

    # plt.figure()
    # plt.plot(epochs, losses, label='Training loss')
    # plt.plot(epochs, val_losses, '--', label='Validation loss')
    # plt.title('Loss of model after training')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    plt.show(block=False)

    # torch.save(model.state_dict(), 'tests/PINNs/PINN.pth')

    return model

def get_estimated_f(model, enkf):
    f_list = []
    n = len(enkf.main_model.state)-1

    for i in range(enkf.ensemble_size):
        x = np.zeros(2*n)
        x[:n] = enkf.ensemble[i].state[:-1]
        x[n:] = enkf.ensemble[i].prev_state[:-1]
        x = torch.from_numpy(x).type(torch.float32).to(device)
        f_list.append(model.predict(x).item())
        
    return np.array(f_list).mean()

def initialize(ensemble_size, f, csv):
    alg_config = {
        '@mainModel': None,
        '@analysisTimes': None,
        '@ensembleModel': None
    }
    model_factory = SaintVenantModelFactory(f)
    obs_config = {
        'store_name': None,
        'working_dir': './observations',
        'config_file': csv,
        'labels': ['0', '6', '12', '20'],
        'std': [0.5, 0.5, 0.5, 0.5]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)

    return enkf

def run_simulation(model, enkf, f_init, f_true):
    impr = 0

    for j in range(19):
        if j%6==0: # Use Kalman Filer every .. step
            kalman_algorithm(enkf)
        else:
            no_filter(enkf)

        if j==18:
            f_est = get_estimated_f(model, enkf)
            print(f"Absolute error (init) = {abs(f_true-f_init[0])}")
            print(f'Absolute error (est) = {abs(f_true-f_est)}')
            print('')
            if abs(f_true-f_init[0]) > abs(f_true-f_est):
                impr = 1

    return f_est, impr, abs(f_true-f_est)

def test():
    ensemble_size = 50
    # Perform 7*7 = 49 twin experiments to test performance
    f_init_list = [[1e-4+i*0.00015] for i in range(7)]
    obs_list = ['obs_simulated_5min_.00010.csv', 'obs_simulated_5min_.00025.csv', 'obs_simulated_5min_.00040.csv',
                'obs_simulated_5min_.00055.csv', 'obs_simulated_5min_.00070.csv', 'obs_simulated_5min_.00085.csv', 'obs_simulated_5min_.00100.csv']

    f_est_arr = np.zeros((len(obs_list), len(f_init_list)))
    error_arr = np.zeros((len(obs_list), len(f_init_list)))
    impr_arr = np.zeros((len(obs_list), len(f_init_list)))

    data = load_data(r'./tests/training_data/training_data.csv')
    enkf = initialize(ensemble_size, None, obs_list[0])
    model = setup_and_train_NN(device, data, enkf)

    for i, csv in enumerate(obs_list):
        f_true = float(csv[-10:][:-4])
        print(f"True f = {f_true}")
        for j, f_init in enumerate(f_init_list):
            enkf = initialize(ensemble_size, f_init, csv)
            f_est_arr[i, j], impr_arr[i,j], error_arr[i, j] = run_simulation(model, enkf, f_init, f_true)

    print(f"We've had {np.sum(impr_arr)} improvements")
    print(f"This means an improvement {np.sum(impr_arr)/(len(f_init_list)*len(obs_list))*100 :.2f}% of the time")
    print(f"The Mean Absolute Error was {error_arr.mean()}")

    with np.printoptions(linewidth=200):
        print("The Error array is:")
        print(error_arr)
        print("The Improvements are at:")
        print(impr_arr)


    assert np.sum(impr_arr)/(len(f_init_list)*len(obs_list)) > 0.5 # We expect improvement at least 50% of the time


if __name__ == '__main__':
    test()