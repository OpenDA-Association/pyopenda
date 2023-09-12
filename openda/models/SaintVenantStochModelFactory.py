import numpy as np
from openda.models.SaintVenantStochModelInstance import SaintVenantStochModelInstance

class SaintVenantModelFactory:
    """
    Interface of Saint-Venant Model Factory

    @author Aron Schouten
    """

    def __init__(self):
        """
        Constructor
        """
        names = ["D", "f", "g", "L", "n"]
        param_values = [20.0, 20.e-5, 9.81, 100.e3, 100]
        param_uncertainty = [0, 0, 0, 0, 0]

        param = dict(zip(names, param_values))
        param_uncertainty = dict(zip(names, param_uncertainty))

        state = np.zeros(2*param['n'])
        state_uncertainty = np.zeros(2*param['n'])

        sys_mean = np.zeros(2*param['n'])
        sys_std = np.zeros(2*param['n'])

        reftime = np.datetime64('2023-11-20T00:00:00')
        span = [reftime, np.timedelta64(10,'m'), reftime+np.timedelta64(2,'D')] # [0 seconds, 10 minutes, 2 days]

        self.model_attributes = (param, param_uncertainty, state, state_uncertainty, sys_mean, sys_std, span)

    def get_instance(self, noise_config, main_or_ens):
        """
        Create an instance of the stochastic Model.

        :param noise_config: dictionary as given by EnkfAlgorithm.xml for the noise configuration.
        :param main_or_ens: determines the ouput level of the model.
        :return: the stochastic Model instance.
        """
        return SaintVenantStochModelInstance(self.model_attributes, noise_config, main_or_ens)