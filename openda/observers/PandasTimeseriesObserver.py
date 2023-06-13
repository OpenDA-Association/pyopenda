from openda.interfaces.IStochObserver import IStochObserver
import pandas as pd

class PandasTimeseriesObserver(IStochObserver):

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
        t_start = model_span.get_start()
        t_end = model_span.get_end()
        data_sel = self.data[self.index >= t_start and self.index <= t_end]

        return PandasTimeseriesObserver(data_sel, self.std, self.indices)

    def get_times(self):
        """
        Get all different times in increasing order. There is at least one observation for each time.

        :return: some type of vector containing the times
        """
        return list(self.data.index)

    def get_count(self):
        """
        Total number of observations.

        :return: the number of observations.
        """
        return len(self.data) * len(self.data.index)

    def get_observation_descriptions(self):
        """
        Get the observation descriptions.

        :return: observation descriptions which are compatible with the used model instance
        """
        return self

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
