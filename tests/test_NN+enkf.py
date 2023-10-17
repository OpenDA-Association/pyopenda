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


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    warnings.warn('CUDA is not available, so PINN will be trained using CPU instead of GPU, which can be very slow. Please consider installing CUDA.')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data(csv):
    df = pd.read_csv(csv, delimiter=';', header=None)
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    x_train = torch.tensor(train.iloc[:,:-1].values, dtype=torch.float32)
    y_train = torch.tensor(train.iloc[:,-1].values, dtype=torch.float32).view(-1, 1)

    x_test = torch.tensor(test.iloc[:,:-1].values, dtype=torch.float32)
    y_test = torch.tensor(test.iloc[:,-1].values, dtype=torch.float32).view(-1, 1)

    return [x_train, y_train, x_test, y_test]

def setup_and_train_NN(device, data, enkf):
    layers =  [44, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PINN(device, layers, enkf, data)
    model.to(device)

    optimizer = torch.optim.Adagrad(model.parameters())

    start_time = time.time()
    epochs, losses, val_losses = model.train_model(optimizer, n_epochs=1, batch_size=32)
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.2f}')

    plt.figure()
    plt.plot(epochs, losses, '--', label='Training loss')
    plt.plot(epochs, val_losses, label='Validation loss')
    plt.title('Loss of model after training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show(block=False)

    torch.save(model.state_dict(), 'PINN.pth')
    # model.load_state_dict(torch.load('PINN_timedep.pth'))

    return model

def get_estimated_f(model, enkf):
    f = []
    n = len(enkf.main_model.state)-1
    x = np.zeros(2*n)
    x[:n] = enkf.main_model.state[:-1]
    x[n:] = enkf.main_model.prev_state[:-1]
    x = torch.from_numpy(x).type(torch.float32).to(device)

    with torch.no_grad(): # To make sure we don't train when testing
        f.append(model.predict(x))

    for i in range(enkf.ensemble_size):
        x = np.zeros(2*n)
        x[:n] = enkf.ensemble[i].state[:-1]
        x[n:] = enkf.ensemble[i].prev_state[:-1]
        x = torch.from_numpy(x).type(torch.float32).to(device)

        with torch.no_grad(): # To make sure we don't train when testing
            f.append(model.forward(x).item())
    
    model.train()
    return f

def plot_series(res, xlocs_waterlevel=None, xlocs_velocity=None):
    (t, results, results_PINN, no_results, obs, ensemble, index) = res
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
        # for j in range(ensemble.shape[0]):
        #     ax.plot(t, ensemble[j,:,i], alpha=0.05)
        ax.plot(t, no_results[:,i], '-.r', label="Without ENKF")
        ax.plot(t, results[:,i], 'b', label="With ENKF")
        ax.plot(t, results_PINN[:,i], '--g', label="ENKF + PINN")
        ax.plot(t[index], obs[:,i][index], ':k', label="Observation")
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
    alg_config = {
        '@mainModel': None,
        '@analysisTimes': None,
        '@ensembleModel': None
    }
    model_factory = SaintVenantModelFactory()
    obs_config = {
        'store_name': None,
        'working_dir': './../observations',
        'config_file': 'obs (simulated2).csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.5, 0.5, 0.5, 0.5]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    enkf_PINN = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    compare_class = GenericEnsembleKalmanFilter(1, alg_config, model_factory, stoch_observer)
    return enkf, enkf_PINN, compare_class

def run_simulation(enkf, enkf_PINN, compare_class):
    span = enkf.model_factory.model_attributes[-1]
    n_obs = enkf.get_n_observations()
    n_times = enkf.get_n_times()
    results = np.zeros((n_times, n_obs))
    results_PINN = np.zeros((n_times, n_obs))
    no_results = np.zeros((n_times, n_obs))
    ensemble = np.zeros((enkf.ensemble_size, n_times, n_obs))

    data = load_data('training_data.csv')
    model = setup_and_train_NN(device, data, enkf)

    t = []
    index = []
    # knock = []
    next_time = span[0]
    for j in range(n_times):
        print(j)
        t.append(next_time)
        next_time = next_time + span[1]
        no_results[j, :] = no_filter(compare_class) # Compare class that never has an EnKF step

        # knock.append(enkf.main_model.get_state()[14])

        if j%12==0: # Use Kalman Filer every .. step
            results[j, :] = kalman_algorithm(enkf)
            results_PINN[j, :] = kalman_algorithm(enkf_PINN)
            index.append(j)
        else:
            results[j, :] = no_filter(enkf)
            results_PINN[j, :] = no_filter(enkf_PINN)

        if j==18:
            f = np.array( get_estimated_f(model, enkf) )
            f_init = enkf.model_factory.model_attributes[0]['f']
            plt.figure()
            plt.hist(f, bins=10, color='c', edgecolor='k', alpha=0.65)
            plt.axvline(np.median(f), color='c', linestyle='dashed', linewidth=1)
            plt.text(np.median(f), .5, 'Estimated $f$')
            plt.axvline(f_init, color='c', linestyle='dashed', linewidth=1)
            plt.text(f_init, .5, 'Initial $f$')
            plt.axvline(0.00018815210759974782, color='k', linestyle='dashed', linewidth=1)
            plt.text(0.00018815210759974782, .5, 'True $f$')
            plt.show(block=False)
            enkf_PINN.main_model.set_f(np.median(f))
            for i in range(enkf.ensemble_size):
                enkf_PINN.ensemble[i].set_f(np.median(f))
            print(f"Estimated f's from main & ensemble = {f}")
            print(f'Avg estimated f = {np.median(f)}')
            print(f'MSE = {(np.median(f)-0.00018815210759974782)**2}')

        for i in range(enkf.ensemble_size): # For plotting the ensemble members
            ensemble[i, j, :] = enkf.ensemble[i].get_observations(enkf.observer.labels)

    obs = np.zeros((n_times, n_obs))
    for i, idx in enumerate(enkf.observer.labels):
        obs[:,i] = enkf.observer.all_timeseries[idx].values[:n_times]

    return (t, results, results_PINN, no_results, obs, ensemble, index)#, knock)

def compare_with_knock(knock, t):
    obs_config = {'store_name': None, 'working_dir': './../observations', 'config_file': 'obs (knock).csv', 'labels': ['14'], 'std': [0]}
    stoch_knock = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))
    obs_knock = stoch_knock.all_timeseries['14'].values
    t_knock = stoch_knock.all_timeseries['14'].times

    MSE_Knock = np.average( (np.array(knock)-np.array(obs_knock[::5]))**2, axis = 0)
    print(f"MSE at Knock is {MSE_Knock}")

    _, ax = plt.subplots(figsize=(11,7))
    ax.plot(t, knock, 'b', label="With ENKF")
    ax.plot(t_knock, obs_knock, '--k', label="Observation")
    ax.set_xlabel("time")
    ax.set_ylabel('Waterlevel (m)')
    ax.set_title("Time Series of Waterlevel at Knock")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
    ax.tick_params(axis='x', labelrotation = 70)
    plt.tight_layout()
    plt.legend()
    plt.show(block=False)

def test():
    ## Initializing ##
    ensemble_size = 50
    enkf, enkf_PINN, compare_class = initialize(ensemble_size)

    ## Running the Model ##
    res = run_simulation(enkf, enkf_PINN, compare_class)
    (_, results, results_PINN, no_results, obs, _, index) = res

    ## Returning results ##
    MSE_res = np.average( (results[index]-obs[index])**2, axis = 0)
    MSE_PINN = np.average( (results_PINN[index]-obs[index])**2, axis = 0)
    MSE_no_res = np.average( (no_results[index]-obs[index])**2, axis = 0)

    print(f"MSE without EnKF is {MSE_no_res}")
    print(f"MSE with EnKF is {MSE_res}")
    print(f"MSE with EnKF + PINN is {MSE_PINN}")

    plot_series(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])
    plot_ensemble(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])


    ## Testing model with Knock ##
    # compare_with_knock(knock, t)

if __name__ == '__main__':
    test()
    _ = input("Press [enter] to close plots and continue.")
