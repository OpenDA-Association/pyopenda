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
        param_values = [15, 2.3e-4, 9.81, 55.e3, 11]
        param_uncertainty = [0, 0, 0, 0, 0]

        param = dict(zip(names, param_values))
        param_uncertainty = dict(zip(names, param_uncertainty))

        state = np.zeros(2*param['n'] + 1)
        state_uncertainty = np.ones(2*param['n'] + 1) * 0.02
        state_uncertainty[-1] = 0

        reftime = np.datetime64('2022-02-18T08:00:00')
        span = [reftime, np.timedelta64(10,'m'), reftime+np.timedelta64(2,'D')] # [0 seconds, 10 minutes, 2 days]

        self.model_attributes = (param, param_uncertainty, state, state_uncertainty, span)

    def get_instance(self, noise_config, main_or_ens):
        """
        Create an instance of the stochastic Model.

        :param noise_config: dictionary as given by EnkfAlgorithm.xml for the noise configuration.
        :param main_or_ens: determines the ouput level of the model.
        :return: the stochastic Model instance.
        """
        return SaintVenantStochModelInstance(self.model_attributes, noise_config, main_or_ens)