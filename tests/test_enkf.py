import os
from openda.observers.pandas_observer import PandasObserver
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
import openda.algorithms.GenericEnsembleKalmanFilter as ENFK

def test():
    ensemble_size = 50
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
        'labels': ['h0', 'h25', 'h50', 'h75', 'h99'],
        'std': [0.2, 0.3, 0.3, 0.3, 0.3]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf_model = ENFK.GenericEnsembleKalmanFilter(ensemble_size, alg_config, model_factory, stoch_observer)

if __name__ == '__main__':
    test()