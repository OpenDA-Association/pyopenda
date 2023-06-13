from openda.interfaces import IStochObserver, IObservationDescription
import pandas as pd

class PandasTimeseriesObserver(IStochObserver, IObservationDescription):

    def __init__(self, data, std, indices):
        self.data = data
        self.indices = indices
        self.std = std

    def __init__(self, values, times, labels, std, indices):
        self.data = pd.DataFrame(values, index=times, columns=labels)
        self.indices = indices
        self.std = std


    def create_selection(self, model_span):
        """
        Create a new observer, containing a selection of the present observer,
        based on the given time span.

        :param model_span: time span with selection.
        :return: stochastic observer containing the required selection.
        """


        raise NotImplementedError("Function not implemented.")

    def get_times(self):
        """
        Get all different times in increasing order. There is at least one observation for each time.

        :return: some type of vector containing the times
        """
        raise NotImplementedError("Function not implemented.")

    def get_count(self):
        """
        Total number of observations.

        :return: the number of observations.
        """
        raise NotImplementedError("Function not implemented.")

    def get_observation_descriptions(self):
        """
        Get the observation descriptions.

        :return: observation descriptions which are compatible with the used model instance
        """
        raise NotImplementedError("Function not implemented.")

    def get_sqrt_covariance(self):
        """
        Get the covariance matrix for the stochastic observations.

        :return: the covariance matrix as numpy array.
        """
        raise NotImplementedError("Function not implemented.")

    def get_standard_deviation(self):
        """
        Get the standard deviation for each stochastic observation
        """
        raise NotImplementedError("Function not implemented.")

    def get_realizations(self):
        """
        Get realization values for all observations, for one ensemble member.

        :return: the realizations.
        """
        raise NotImplementedError("Function not implemented.")
