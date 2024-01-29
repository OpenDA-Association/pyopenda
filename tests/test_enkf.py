import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from openda.algorithms.ensemble_kalman import kalman_algorithm
from openda.algorithms.ensemble_kalman import no_filter
from openda.algorithms.GenericEnsembleKalmanFilter import GenericEnsembleKalmanFilter
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
from openda.observers.pandas_observer import PandasObserver


def plot_series(res, xlocs_waterlevel=None, xlocs_velocity=None):
    (t, results, no_results, obs, ensemble, index) = res
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
        for j in range(ensemble.shape[0]):
            ax.plot(t, ensemble[j,:,i], alpha=0.05)
        ax.plot(t, results[:,i], 'b', label="With ENKF")
        ax.plot(t, no_results[:,i], '-.r', label="Without ENKF")
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
    (t, _, _, obs, ensemble, index) = res
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
        'working_dir': './observations',
        'config_file': 'obs_storm_Eunice_5min.csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.5, 0.5, 0.5, 0.5]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    compare_class = GenericEnsembleKalmanFilter(1, alg_config, model_factory, stoch_observer)
    return enkf, compare_class

def run_simulation(enkf, compare_class):
    span = enkf.model_factory.model_attributes[-1]
    n_obs = enkf.get_n_observations()
    n_times = enkf.get_n_times()
    results = np.zeros((n_times, n_obs))
    no_results = np.zeros((n_times, n_obs))
    ensemble = np.zeros((enkf.ensemble_size,n_times, n_obs))
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

        if j%2==0: # Use Kalman Filer every .. step
            results[j, :] = kalman_algorithm(enkf)
            index.append(j)
        else:
            results[j, :] = no_filter(enkf)

        for i in range(enkf.ensemble_size): # For plotting the ensemble members
            ensemble[i, j, :] = enkf.ensemble[i].get_observations(enkf.observer.labels)

    obs = np.zeros((n_times, n_obs))
    for i, idx in enumerate(enkf.observer.labels):
        obs[:,i] = enkf.observer.all_timeseries[idx].values[:n_times]

    # return (t, results, no_results, obs, ensemble, index, knock)
    return (t, results, no_results, obs, ensemble, index)

def compare_with_knock(knock, t):
    obs_config = {'store_name': None, 'working_dir': './observations', 'config_file': 'obs_knock_1min.csv', 'labels': ['14'], 'std': [0]}
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
    enkf, compare_class = initialize(ensemble_size)

    ## Running the Model ##
    res = run_simulation(enkf, compare_class)
    # (t, results, no_results, obs, _, index, knock) = res
    (_, results, no_results, obs, _, index) = res

    ## Returning results ##
    MSE_res = np.average( (results[index]-obs[index])**2, axis = 0)
    MSE_no_res = np.average( (no_results[index]-obs[index])**2, axis = 0)

    print(f"MSE without EnKF is {MSE_no_res}")
    print(f"MSE with EnKF is {MSE_res}")

    plot_series(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])
    plot_ensemble(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])

    assert all(MSE_res <= MSE_no_res) # MSE with EnKF should not be worse than MSE withouth EnKF

    ## Testing model with Knock ##
    # compare_with_knock(knock, t)


if __name__ == '__main__':
    test()
    _ = input("Press [enter] to close plots and continue.")
