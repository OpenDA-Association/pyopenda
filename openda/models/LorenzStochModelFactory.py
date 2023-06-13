#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factory for making Lorenz model instances, usable by the ensemble kalman filter algorithm.

"""

from openda.models.LorenzStochModelInstance import LorenzStochModelInstance


class LorenzStochModelFactory:
    """
    Factory for making Lorenz model instances
    """
    def __init__(self, config):
        """
        :param config: dictionary used for configuration.
        """
        if config is None:
            self.config = {'sigma': 10, 'rho': 28.0, 'beta': 8.0/3.0, 't_start': 0.0, 't_stop': 30.0, 't_step': 0.025,
                           'state': [1.508870, -1.531271, 25.46091]}
        else:
            self.config = config.copy()

    def get_instance(self, noise_config, main_or_ens):
        """
        Create an instance of the stochastic Model.

        :param noise_config: dictionary as given by EnkfAlgorithm.xml for the noise configuration.
        :param main_or_ens: determines the ouput level of the model.
        :return: the stochastic Model instance.
        """
        return LorenzStochModelInstance(self.config, noise_config, main_or_ens)
