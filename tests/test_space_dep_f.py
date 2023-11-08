import os
import time
import warnings
import matplotlib.dates as mdates
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
    device = 'cpu'
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
    # model.load_state_dict(torch.load('PINN.pth'))

    optimizer = torch.optim.Adagrad(model.parameters())

    start_time = time.time()
    epochs, losses, val_losses = model.train_model(optimizer, n_epochs=2, batch_size=32)
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.2f}')

    plt.figure()
    plt.plot(epochs, losses, label='Training loss')
    plt.plot(epochs, val_losses, '--', label='Validation loss')
    plt.title('Loss of model after training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False)

    # torch.save(model.state_dict(), 'PINN.pth')

    return model

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

def plot_series(res, xlocs_waterlevel=None, xlocs_velocity=None):
    (t, results, no_results, results_PINN, obs, _, index) = res
    t = np.array(t)
    index = np.array(index)
    titles=[]
    ylabels=[]
    for xloc in xlocs_waterlevel:
        titles.append(f"Time Series of Waterlevel at {xloc}")
        ylabels.append("height (m)")
    for xloc in xlocs_velocity:
        titles.append(f"Time Series of Velocity at {xloc}")
        ylabels.append("velocity (m/s)")

    for i in range(results.shape[1]):
        _, ax = plt.subplots(figsize=(11,7))
        ax.plot(t, results[:,i], 'b', label="With ENKF")
        ax.plot(t, no_results[:,i], '-.r', label="Without ENKF")
        ax.plot(t, results_PINN[:,i], ':g', label="ENKF+PINN")
        ax.plot(t[index], obs[:,i][index], '--k', label="Observation")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
        ax.tick_params(axis='x', labelrotation = 70)
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)

def plot_ensemble(res, xlocs_waterlevel, xlocs_velocity):
    (t, _, _, _, obs, ensemble, index) = res
    t = np.array(t)
    index = np.array(index)
    titles=[]
    ylabels=[]
    for xloc in xlocs_waterlevel:
        titles.append(f"Time Series of Waterlevel of {ensemble.shape[0]} Ensembles at {xloc}")
        ylabels.append("height (m)")
    for xloc in xlocs_velocity:
        titles.append(f"Time Series of Velocity of {ensemble.shape[0]} Ensembles at {xloc}")
        ylabels.append("velocity (m/s)")

    for i in range(ensemble.shape[2]):
        _, ax = plt.subplots(figsize=(11,7))
        for j in range(ensemble.shape[0]):
            ax.plot(t, ensemble[j,:,i], alpha=0.4)
        ax.plot(t[index], obs[:,i][index], '--k', label="Observation")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
        ax.tick_params(axis='x', labelrotation = 70)
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)

def initialize(ensemble_size):
    f_true = test_gen_obs()
    alg_config = {
        '@mainModel': None,
        '@analysisTimes': None,
        '@ensembleModel': None
    }
    model_factory = SaintVenantModelFactory()
    obs_config = {
        'store_name': None,
        'working_dir': './observations',
        'config_file': 'obs_simulated_5min.csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.5, 0.5, 0.5, 0.5]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    enkf_PINN = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    compare_class = GenericEnsembleKalmanFilter(1, alg_config, model_factory, stoch_observer)
    return enkf, enkf_PINN, compare_class, f_true

def run_simulation(model, enkf, enkf_PINN, compare_class, f_true):
    n_obs = enkf.get_n_observations()
    n_times = enkf.get_n_times()
    results = [np.zeros((n_times, n_obs)), np.zeros((n_times, n_obs)), np.zeros((n_times, n_obs)),
               np.zeros((n_times, n_obs)), np.zeros((enkf.ensemble_size,n_times, n_obs))]
    t = []
    index = []
    next_time = enkf.model_factory.model_attributes[-1][0]
    for j in range(n_times):
        print(j)
        t.append(next_time)
        next_time = next_time + enkf.model_factory.model_attributes[-1][1]
        results[1][j, :] = no_filter(compare_class) # Compare class that never has an EnKF step

        if j%6==0: # Use Kalman Filer every .. step
            results[0][j, :] = kalman_algorithm(enkf)
            results[2][j, :] = kalman_algorithm(enkf_PINN)
            index.append(j)
        else:
            results[0][j, :] = no_filter(enkf)
            results[2][j, :] = no_filter(enkf_PINN)

        if j==18:
            f_est = get_estimated_f(model, enkf_PINN)
            print(f"Initial f = {enkf.model_factory.model_attributes[0]['f']}")
            print(f"True f = {f_true}")
            print(f'Estimated f = {f_est}')

            # Set estimated f as new bottom friction coefficient
            enkf_PINN.main_model.set_f(f_est)
            for i in range(enkf.ensemble_size):
                enkf_PINN.ensemble[i].set_f(f_est)

        for i in range(enkf.ensemble_size):
            results[4][i, j, :] = enkf.ensemble[i].get_observations(enkf.observer.labels)
            # results[4][i, j, :] = enkf_PINN.ensemble[i].get_observations(enkf_PINN.observer.labels)

    for i, idx in enumerate(enkf.observer.labels):
        results[3][:,i] = enkf.observer.all_timeseries[idx].values[:n_times]

    return (t, results[0], results[1], results[2], results[3], results[4], index)

def test():
    ## Initializing ##
    ensemble_size = 50
    enkf, enkf_PINN, compare_class, f_true = initialize(ensemble_size)
    model = setup_and_train_NN(device, load_data(r'./tests/training_data/training_data_space_dep.csv'), enkf)

    ## Running the Model ##
    res = run_simulation(model, enkf, enkf_PINN, compare_class, f_true)
    (_, results, no_results, results_PINN, obs, _, index) = res

    ## Returning results ##
    MSE_res = np.average( (results[index]-obs[index])**2, axis = 0)
    MSE_no_res = np.average( (no_results[index]-obs[index])**2, axis = 0)
    MSE_PINN = np.average( (results_PINN[index]-obs[index])**2, axis = 0)

    print(f"MSE without EnKF is {MSE_no_res}")
    print(f"MSE with EnKF is {MSE_res}")
    print(f"MSE with EnKF + PINN is {MSE_PINN}")

    plot_series(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])
    plot_ensemble(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])


if __name__ == '__main__':
    test()
    _ = input("Press [enter] to close plots and continue.")