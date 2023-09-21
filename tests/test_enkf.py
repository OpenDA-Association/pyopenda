import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from openda.observers.pandas_observer import PandasObserver
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
from openda.algorithms.GenericEnsembleKalmanFilter import GenericEnsembleKalmanFilter
from openda.algorithms.ensemble_kalman import kalman_algorithm, no_filter

def plot_series(t, idx, results, no_results=None, ensemble=None, obs=None, xlocs_waterlevel=None, xlocs_velocity=None, include_ensemble=False):
    titles=[]
    ylabels=[]
    for xloc in xlocs_waterlevel:
        titles.append(f"Time Series of Waterlevel at {xloc}")
        ylabels.append("height (m)")
    for xloc in xlocs_velocity:
        titles.append(f"Time Series of Velocity at {xloc}")
        ylabels.append("velocity (m/s)")

    for i in range(results.shape[1]):
        fig, ax = plt.subplots(figsize=(11,7))
        if include_ensemble:
            for j in range(ensemble.shape[0]):
                ax.plot(t, ensemble[j,:,i], alpha=0.1)
        ax.plot(t, results[:,i], 'b', label="With ENKF")
        ax.plot(t, no_results[:,i], '-.r', label="Without ENKF")
        ax.plot(t[idx], obs[:,i][idx], '--k', label="Observation")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
        ax.tick_params(axis='x', labelrotation = 70)
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)

def plot_ensemble(t, idx, ensemble, obs, xlocs_waterlevel, xlocs_velocity):
    titles=[]
    ylabels=[]
    for xloc in xlocs_waterlevel:
        titles.append(f"Time Series of Waterlevel of {ensemble.shape[0]} Ensembles at {xloc}")
        ylabels.append("height (m)")
    for xloc in xlocs_velocity:
        titles.append(f"Time Series of Velocity of {ensemble.shape[0]} Ensembles at {xloc}")
        ylabels.append("velocity (m/s)")

    for i in range(ensemble.shape[2]):
        fig, ax = plt.subplots(figsize=(11,7))
        for j in range(ensemble.shape[0]):
            ax.plot(t, ensemble[j,:,i], alpha=0.4)
        ax.plot(t[idx], obs[:,i][idx], '--k', label="Observation")
        ax.set_xlabel("time")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
        ax.tick_params(axis='x', labelrotation = 70)
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)

def test():
    ## Initializing ##
    ensemble_size = 50
    alg_config = {
        '@mainModel': None,
        '@analysisTimes': None,
        '@ensembleModel': None
    }
    model_factory = SaintVenantModelFactory()
    obs_config = {
        'store_name': None,
        'working_dir': './../observations',
        'config_file': 'obs (simulated).csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.6, 0.6, 0.6, 0.6]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)
    compare_class = GenericEnsembleKalmanFilter(1, alg_config, model_factory, stoch_observer)

    span = enkf.model_factory.model_attributes[-1]
    n_obs = enkf.get_n_observations()
    n_times = enkf.get_n_times()
    
    ## Running the Model ##
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
        no_results[j, :] = no_filter(compare_class) # Compare class that never has an EnKF step

        if j%1==0: # Use Kalman Filer every .. step
            results[j, :] = kalman_algorithm(enkf)
            index.append(j)
        else:
            results[j, :] = no_filter(enkf)
        
        for i in range(ensemble_size): # For plotting the ensemble members
            ensemble[i, j, :] = enkf.ensemble[i].get_observations(obs_config.get('labels'))

    obs = np.zeros((n_steps, n_obs))
    for i, idx in enumerate(enkf.observer.labels):
        obs[:,i] = enkf.observer.all_timeseries[idx].values[:n_steps]
    
    ## Returning results ##
    MSE_res = np.sum( (results[index]-obs[index])**2, axis = 0) / n_steps
    MSE_no_res = np.sum( (no_results[index]-obs[index])**2, axis = 0) / n_steps

    print(f"MSE with EnKF is {MSE_res}")
    print(f"MSE without EnKF is {MSE_no_res}")

    plot_series(np.array(t), np.array(index), results, no_results, ensemble, obs[:n_steps,:], ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [], include_ensemble=False)
    plot_ensemble(np.array(t), np.array(index), ensemble, obs, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])


if __name__ == '__main__':
    test()
    _ = input("Press [enter] to close plots and continue.")