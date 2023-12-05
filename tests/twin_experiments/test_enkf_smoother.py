import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from openda.algorithms.ensemble_kalman import kalman_algorithm
from openda.algorithms.ensemble_kalman import no_filter
from openda.algorithms.GenericEnsembleKalmanFilter import GenericEnsembleKalmanFilter
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
from openda.models.SaintVenantWithSmootherFactory import SaintVenantWithSmootherFactory
from openda.observers.pandas_observer import PandasObserver


def plot_series(res, xlocs_waterlevel=None, xlocs_velocity=None):
    (t, results, results_smoother, no_results, obs, _, index) = res
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
        ax.plot(t, results_smoother[:,i], ':g', label="ENKF+Smoother")
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
    model_factory = SaintVenantModelFactory(f=[0.00010])
    obs_config = {
        'store_name': None,
        'working_dir': './observations',
        'config_file': 'obs_simulated_5min_.00055.csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.5, 0.5, 0.5, 0.5]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    compare_class = GenericEnsembleKalmanFilter(1, alg_config, model_factory, stoch_observer)

    enkf_smoother = GenericEnsembleKalmanFilter(ensemble_size, alg_config, SaintVenantWithSmootherFactory(f=[0.00010]), stoch_observer)

    return enkf, enkf_smoother, compare_class

def run_simulation(enkf, enkf_smoother, compare_class):
    span = enkf.model_factory.model_attributes[-1]
    n_obs = enkf.get_n_observations()
    n_times = enkf.get_n_times()

    results = [np.zeros((n_times, n_obs)), np.zeros((n_times, n_obs)), np.zeros((n_times, n_obs)),
               np.zeros((n_times, n_obs)), np.zeros((enkf.ensemble_size,n_times, n_obs))]
    t = []
    index = []
    next_time = span[0]
    for j in range(n_times):
        print(j)
        t.append(next_time)
        next_time = next_time + span[1]
        results[2][j, :] = no_filter(compare_class) # Compare class that never has an EnKF step


        if j%6==0: # Use Kalman Filer every .. step
            results[0][j, :] = kalman_algorithm(enkf)
            results[1][j, :] = kalman_algorithm(enkf_smoother)
            print(f"EnKF main model f = {enkf.main_model.f}")
            print(f"EnKF Smoother main model f = {enkf_smoother.main_model.f}")
            index.append(j)
        else:
            results[0][j, :] = no_filter(enkf)
            results[1][j, :] = no_filter(enkf_smoother)

        for i in range(enkf.ensemble_size): # For plotting the ensemble members
            results[4][i, j, :] = enkf.ensemble[i].get_observations(enkf.observer.labels)

    for i, idx in enumerate(enkf.observer.labels):
        results[3][:,i] = enkf.observer.all_timeseries[idx].values[:n_times]

    return (t, results[0], results[1], results[2], results[3], results[4], index)

def test():
    ## Initializing ##
    ensemble_size = 50
    enkf, enkf_smoother, compare_class = initialize(ensemble_size)

    ## Running the Model ##
    res = run_simulation(enkf, enkf_smoother, compare_class)
    (_, results, results_smoother, no_results, obs, _, index) = res

    ## Returning results ##
    MSE_res = np.average( (results[index]-obs[index])**2, axis = 0)
    MSE_no_res = np.average( (no_results[index]-obs[index])**2, axis = 0)
    MSE_smoother = np.average( (results_smoother[index]-obs[index])**2, axis = 0)

    print(f"MSE without EnKF is {MSE_no_res}")
    print(f"MSE with EnKF is {MSE_res}")
    print(f"MSE with EnKF+Smoother is {MSE_smoother}")

    plot_series(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])
    plot_ensemble(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])


if __name__ == '__main__':
    test()
    _ = input("Press [enter] to close plots and continue.")
