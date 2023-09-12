import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openda.observers.pandas_observer import PandasObserver
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
from openda.algorithms.GenericEnsembleKalmanFilter import GenericEnsembleKalmanFilter
from openda.algorithms.ensemble_kalman import kalman_algorithm, no_filter

def plot_series(t, idx, results, no_results=None, obs=None, xlocs_waterlevel=None, xlocs_velocity=None):
    titles=[]
    ylabels=[]
    for xloc in xlocs_waterlevel:
        titles.append(f"Time Series of Waterlevel at x = {xloc*1e-3} km")
        ylabels.append("height (m)")
    for xloc in xlocs_velocity:
        titles.append(f"Time Series of Velocity at x = {xloc*1e-3} km")
        ylabels.append("velocity (m/s)")

    for i in range(results.shape[1]):
        fig, ax = plt.subplots(figsize=(11,7))
        ax.plot(t, results[:,i], label="With ENKF")
        ax.scatter(t[idx], results[:,i][idx], label="ENKF Step")
        ax.plot(t, no_results[:,i], label="Without ENKF")
        ax.plot(t, obs[:,i], '--k', label="Observation")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
        ax.tick_params(axis='x', labelrotation = 70)
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)

def plot_ensemble(t, ensemble, obs, xlocs_waterlevel, xlocs_velocity):
    titles=[]
    ylabels=[]
    for xloc in xlocs_waterlevel:
        titles.append(f"Time Series of Waterlevel at x = {xloc*1e-3} km")
        ylabels.append("height (m)")
    for xloc in xlocs_velocity:
        titles.append(f"Time Series of Velocity at x = {xloc*1e-3} km")
        ylabels.append("velocity (m/s)")

    for i in range(ensemble.shape[2]):
        fig, ax = plt.subplots(figsize=(11,7))
        for j in range(ensemble.shape[0]):
            ax.plot(t, ensemble[j,:,i], alpha=0.4)
        ax.plot(t, obs[:,i], '--k', label="Observation")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
        ax.tick_params(axis='x', labelrotation = 70)
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)

def test():
    ensemble_size = 25
    alg_config = {
        '@mainModel': False,
        '@analysisTimes': 288,
        '@ensembleModel': True
    }
    model_factory = SaintVenantModelFactory()
    obs_config = {
        'store_name': None,
        'working_dir': './../',
        'config_file': 'obs (simulated).csv',
        'labels': ['0', '50', '100', '150', '198'],
        'std': [0.2, 0.2, 0.2, 0.2, 0.2]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    compare_class = GenericEnsembleKalmanFilter(1, alg_config, model_factory, stoch_observer)

    span = enkf.model_factory.model_attributes[-1]
    n_obs = enkf.get_n_observations()
    n_times = enkf.get_n_times()
    
    n_steps = n_times
    results = np.zeros((n_steps, n_obs))
    no_results = np.zeros((n_steps, n_obs))
    ensemble = np.zeros((ensemble_size,n_steps, n_obs))
    t = []
    index = []
    next_time = span[0]
    for j in range(n_steps):
        print(j)
        t.append(next_time)
        next_time = next_time + span[1]

        if j%6==0:
            results[j, :] = kalman_algorithm(enkf)
            index.append(j+1)
        else:
            results[j, :] = no_filter(enkf)

        no_results[j, :] = no_filter(compare_class)

        for i in range(ensemble_size):
            ensemble[i, j, :] = enkf.ensemble[i].get_observations([0,50,100,150,198])

    obs = np.zeros((n_times, n_obs))
    for i, idx in enumerate(enkf.observer.labels):
        obs[:,i] = enkf.observer.all_timeseries[idx].values

    plot_series(np.array(t), np.array(index), results, no_results, obs[:n_steps,:], [0, 25*1e3, 50*1e3, 75*1e3, 99*1e3], [])

    plot_ensemble(np.array(t), ensemble, obs, [0, 25*1e3, 50*1e3, 75*1e3, 99*1e3], [])


if __name__ == '__main__':
    test()
    _ = input("Press [enter] to continue.")