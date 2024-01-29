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
from tests.observations.generate_observations import test_gen_obs


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

    x_train = torch.tensor(train.iloc[:,:-3].values, dtype=torch.float32)
    y_train = torch.tensor(train.iloc[:,-3:].values, dtype=torch.float32).view(-1, 3)

    x_test = torch.tensor(test.iloc[:,:-3].values, dtype=torch.float32)
    y_test = torch.tensor(test.iloc[:,-3:].values, dtype=torch.float32).view(-1, 3)

    return [x_train, y_train, x_test, y_test]

def setup_and_train_NN(device, data, enkf):
    layers =  [44, 10, 10, 10, 10, 3]
    model = PINN(device, layers, enkf, data)
    model.to(device)
    model.load_state_dict(torch.load('tests/PINNs/PINN_space_dep.pth'))

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
    # plt.show(block=False)

    # plot_testing(model, data)

    # torch.save(model.state_dict(), 'tests/PINNs/PINN_space_dep.pth')
    
    return model

def plot_testing(model, data):
    _, _, x_test, y_test = data
    y_pred = model.predict(x_test.to(device)).cpu()
    lst = [y_test.min(), y_test.max()]
    labels = ['$f_1$','$f_2$','$f_3$']
    _, axs = plt.subplots(3, sharex=True, sharey=True)
    axs[2].set_xlabel('Ground Truth')
    axs[1].set_ylabel('Prediction')
    for i in range(3):
        axs[i].scatter(y_test[:,i], y_pred[:,i], marker='.', label=labels[i], alpha=0.2)
        axs[i].plot(lst, lst, '--k', label='Truth')
        axs[i].legend()
    plt.show(block=False)

def get_estimated_f(model, enkf):
    f_arr = np.zeros((enkf.ensemble_size, 3))
    n = len(enkf.main_model.state)-1

    for i in range(enkf.ensemble_size):
        x = np.zeros(2*n)
        x[:n] = enkf.ensemble[i].state[:-1]
        x[n:] = enkf.ensemble[i].prev_state[:-1]
        x = torch.from_numpy(x).type(torch.float32).to(device)
        f_arr[i] = model.predict(x).cpu().numpy()
        
    return f_arr.mean(axis=0)

def initialize(ensemble_size):
    f_true = test_gen_obs()
    plt.close('all')
    alg_config = {
        '@mainModel': None,
        '@analysisTimes': None,
        '@ensembleModel': None
    }
    model_factory = SaintVenantModelFactory()
    obs_config = {
        'store_name': None,
        'working_dir': './../observations',
        'config_file': 'obs_simulated_5min.csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.5, 0.5, 0.5, 0.5]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    return enkf, f_true

def run_simulation(model, enkf, f_true):

    for j in range(19):

        if j%6==0: # Use Kalman Filer every .. step
            kalman_algorithm(enkf)
        else:
            no_filter(enkf)

        if j==18:
            f_est = get_estimated_f(model, enkf)
            f_init = enkf.model_factory.model_attributes[0]['f']
            print(f"Initial f = {f_init}")
            print(f"True f = {f_true}")
            print(f'Estimated f = {f_est}')

            abserr = abs(f_true - f_est)
            impr = ( abs(f_true - f_est) < abs(f_true - f_init) )*1
            under_over = np.zeros(3)
            for i in range(3):
                if f_est[i] > f_true[i]:
                    under_over[i] = 1
                elif f_est[i] < f_true[i]:
                    under_over[i] = -1
                else:
                    under_over[i] = 0

            print(f"Absolute error = {abserr}")
            print(f"Improvements at {impr}")
            print(f"Under/Over estimated {under_over}")
            print("")

    return abserr, impr, under_over

def test():
    ## Initializing ##
    ensemble_size = 50
    enkf, _ = initialize(ensemble_size)
    data = load_data(r'./tests/training_data/training_data_space_dep.csv')
    model = setup_and_train_NN(device, data, enkf)

    n_runs = 10 # Perform 10 twin experiments to test performance
    abserrs = np.zeros(3)
    imprs = np.zeros(3)
    under_overs = np.zeros(3)

    for i in range(n_runs):
        print(i)
        enkf, f_true = initialize(ensemble_size)
        abserr, impr, under_over = run_simulation(model, enkf, f_true)
        abserrs += abserr
        imprs += impr
        under_overs += under_over

    print(f"We've had {imprs} improvements")
    print(f"This means a improvements {imprs/n_runs*100}% of the time")
    print(f"The Mean Absolute Errors are {abserrs/n_runs}")
    print(f"Under/Over estimated {under_overs}")


if __name__ == '__main__':
    test()
    _ = input("Press [enter] to close plots and continue.")