import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from openda.algorithms.ensemble_kalman import no_filter
from openda.algorithms.GenericEnsembleKalmanFilter import GenericEnsembleKalmanFilter
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
from openda.observers.pandas_observer import PandasObserver


def plot_series(res, xlocs_waterlevel=None, xlocs_velocity=None):
    with plt.style.context('bmh'):
        (t, results1, results2) = res
        t = np.array(t)
        titles=[]
        ylabels=[]
        for xloc in xlocs_waterlevel:
            titles.append(f"Time Series of Waterlevel at {xloc}")
            ylabels.append("Waterlevel (m)")
        for xloc in xlocs_velocity:
            titles.append(f"Time Series of Velocity at {xloc}")
            ylabels.append("Velocity (m/s)")

        for i in range(results1.shape[1]):
            _, ax = plt.subplots(figsize=(11,7))
            # for j in range(ensemble.shape[0]):
            #     ax.plot(t, ensemble[j,:,i], alpha=0.05)
            ax.plot(t, results1[:,i], label="$f = 10^{-3}$",color='#A52A2A')
            ax.plot(t, results2[:,i], label="$f = 10^{-4}$",color='#008080')
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabels[i])
            # ax.set_title(titles[i])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
            ax.tick_params(axis='x', labelrotation = 70)
            plt.tight_layout()
            plt.legend()
            plt.show(block=False)


def initialize():
    alg_config = {
        '@mainModel': None,
        '@analysisTimes': None,
        '@ensembleModel': None
    }
    model_factory1 = SaintVenantModelFactory(f=[1e-3])
    model_factory2 = SaintVenantModelFactory(f=[1e-4])
    obs_config = {
        'store_name': None,
        'working_dir': './observations',
        'config_file': 'obs_storm_Eunice_5min.csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.5, 0.5, 0.5, 0.5]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    class1 = GenericEnsembleKalmanFilter(1, alg_config, model_factory1, stoch_observer)
    class2 = GenericEnsembleKalmanFilter(1, alg_config, model_factory2, stoch_observer)

    return class1, class2

def run_simulation(class1, class2):
    span = class1.model_factory.model_attributes[-1]
    n_obs = class1.get_n_observations()
    n_times = class1.get_n_times()

    results1 = np.zeros((n_times, n_obs))
    results2 = np.zeros((n_times, n_obs))

    t = []
    next_time = span[0]
    for j in range(n_times):
        print(j)
        t.append(next_time)
        next_time = next_time + span[1]
        results1[j, :] = no_filter(class1) 
        results2[j, :] = no_filter(class2)

    return (t, results1, results2)

def test():
    class1, class2 = initialize()

    res = run_simulation(class1, class2)

    plot_series(res, ['Borkum', 'Eemshaven', 'Delfzijl', 'Nieuwe Statenzijl'], [])

    (_, results1, results2) = res
    assert all(np.max(results1, axis=0) <= np.max(results2, axis=0)) # f=1e-3 should give lower waterlevels than f=1e-4


if __name__ == '__main__':
    test()
    _ = input("Press [enter] to close plots and continue.")
