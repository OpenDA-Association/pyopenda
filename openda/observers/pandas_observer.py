"""
Stochastic observer for observations stored in a panda's supported format

@author: Nils van Velzen
"""

import os
import numpy as np
import pandas as pd
from openda.costFunctions.JObjects import PyTime
from openda.interfaces.IObservationDescription import IObservationDescription


class TimeSeries:

    def __init__(self, times, values, std):
        self.times = times
        self.values = values
        self.std = std

    def create_time_selection(self, span_start, span_stop):
        # Unpack span_start and span_stop if necessary
        try:
            span_start = span_start.get_start()
            span_stop = span_stop.get_start()
        except:
            pass
        eps = np.timedelta64(1, 's')
        sel_times = []
        sel_values = []
        sel_std = []
        for time, value, std in zip(self.times, self.values, self.std):
            if span_start - eps <= time <= span_stop + eps:
                sel_times.append(time)
                sel_values.append(value)
                sel_std.append(std)
        return TimeSeries(sel_times, sel_values, sel_std)

    def create_mask_selection(self, mask):
        sel_times = []
        sel_values = []
        sel_std = []
        for time, value, std, use in zip(self.times, self.values, self.std, mask):
            if use:
                sel_times.append(time)
                sel_values.append(value)
                sel_std.append(std)
        return TimeSeries(sel_times, sel_values, sel_std)

    def get_values(self):
        return self.values

    def get_std(self):
        return self.std

    def get_times(self):
        return self.times


class PandasObservationDescriptions(IObservationDescription):

    def __init__(self, times, observation_id):
        self.times = times
        self.observation_id = observation_id

    def create_time_selection(self, model_span):
        """
        Create a new observer, containing a selection of the present observer,
        based on the given time span.

        :param model_span: time span with selection.
        :return: stochastic observer containing the required selection.
        """

        # TODO use time selection of stoch observer

        raise NotImplementedError("Function not implemented.")

    def get_exchange_items(self):
        """
        ONLY HERE FOR COMPATIBILITY WITH JAVA DO NOT IMPLEMENT
         Get the exchange items describing the measures available in the stoch. observer.
         :return All exchange items in the stoch. observer.
        """
        raise NotImplementedError("Function not implemented.")

    def get_properties(self, key: str):
        """"
        Get properties (values) that correspond to a given key.

        :param key: key for which the value is asked
        :return Properties: (column of data from observation descriptions).
        """

        if key == "id":
            return self.observation_id
        elif key == "time":
            return self.times
        else:
            raise NotImplementedError("Key is not known only id and time are supported.")

    def get_property_keys(self):
        """"
        return All keys of the observation descriptions.
        """

        return ["id", "time"]

    def get_property_count(self):
        """"
        return Number of properties.
        """
        raise NotImplementedError("Function not implemented.")

    def get_observation_count(self):
        """"
        return Number of observations
        """
        raise NotImplementedError("Function not implemented.")

    def get_times(self):
        """"
        Get all different times in increasing order. There is at least one observation for each time.
        It is likely that observer.createSelection(time[i]) will be used to walk through the
        observations. The implementation of the stochobserver should garantee that al observations are
        returned in exactly one batch this way.
        :return Array with all uniquely different times.
        """
        return self.times


class PandasObserver:
    """
    A stochastic observer which uses pandas to read observations from a csv file.
    """

    def __init__(self, config=None, scriptdir=None, clone=None):
        """
        :param config: dictionary used for configuration.
        :param scriptdir: location of the main .oda file.
        :param clone: if None (default), the class will initialize from configuration,
        otherwise the class will be a copy of clone.
        """

        # TODO make a dict from clone!

        if clone is None:
            store_name = config['store_name']
            working_dir = config['working_dir']
            config_file = config['config_file']
            labels = config['labels']

            # if len(labels)!=1:
            #    raise ValueError("Work in progress... only a single label is supported")

            # Check existence of HDF5 file
            hdf5_input = os.path.join(scriptdir, working_dir, config_file)
            if not os.path.isfile(hdf5_input):
                raise FileExistsError("We cannot find the hdf5 data file " + str(hdf5_input) +
                                      "\nNote: This file is not part of the repository you " +
                                      "must provide this file yourself")

            # File can contain multiple pandas objects. Check whether proved store exists
            hdf5_data = None
            if False: #HDF5
                stores = pd.HDFStore(hdf5_input, mode='r')
                if store_name not in stores:
                    raise ValueError("selected store_name: " + str(store_name) +
                                     " does not exist in HDF5 file: " + str(hdf5_input))
                    # Read values
                    hdf5_data = pd.read_hdf(hdf5_input, key=store_name)
            else:
                hdf5_data = pd.read_csv(hdf5_input, sep=';', header=0, index_col=0, parse_dates=["time"])

            self.all_timeseries = {}
            if "std" in config:
                std_obs = config["std"]
            else:
                std_obs = [1.0] * len(labels)

            for label, std in zip(labels, std_obs):
                if label not in hdf5_data.columns:
                    raise ValueError("provided label " + str(label) + " is not present in HDF data set")

                column = pd.DataFrame(hdf5_data.loc[:, label])
                column_clean = column.dropna()  # TODO
                times = [time for time in column_clean.index.values]
                # column_clean.values is list of lists, hence make it flat
                values = [item for row in column_clean.values for item in row]
                self.all_timeseries[label] = TimeSeries(times, values, [std] * len(values))
            self.labels = labels
        else:
            time_selection = clone[0]
            parent_observer = clone[1]
            sel_label = clone[2]
            masks = clone[3]
            if isinstance(sel_label, str):
                sel_label = [sel_label]
            if isinstance(masks, str):
                masks = [masks]

            if not sel_label:
                sel_label = parent_observer.labels
            if time_selection:
                span_start = time_selection.get_start()
                span_stop = time_selection.get_end()
            else:
                times_parent = parent_observer.get_times()
                span_start = times_parent[0].get_mjd()
                span_stop = times_parent[-1].get_mjd()
            # Check masks
            if masks is None:
                masks = {}
            for mask_id in masks:
                if mask_id not in sel_label:
                    raise Warning("Mask for "+mask_id+" cannot be applied, no corresponding time series found")

            self.all_timeseries = {}
            for label in parent_observer.labels:
                if any([label == l2 for l2 in sel_label]):              # label in any sel_label:
                    sel_timeseries = parent_observer.all_timeseries[label].create_time_selection(span_start, span_stop)
                    # Apply masks
                    if any([label == l2 for l2 in masks]):  # label in masks:
                        sel_timeseries = sel_timeseries.create_mask_selection(masks[label])
                    self.all_timeseries[label] = sel_timeseries
            self.labels = self.all_timeseries.keys()
            # Check whether we cannot find some of the labels in sel_label
            if len(self.labels) != len(self.all_timeseries):
                missing = []
                for label in sel_label:
                    if label not in self.labels:
                        missing.append(label)
                str_missing = ",".join(missing)
                raise ValueError("could not match all observation id's (" + str_missing + ")")

    def __inspan(self, time_span, time):

        eps = np.timedelta64(1, 's')

        in_span = False
        span_start = time_span.get_start()
        span_stop = time_span.get_end()
        span_step = time_span.get_step_mjd()

        if span_start - eps <= time <= span_stop + eps:
            frac = abs((time - span_start) / span_step)
            delta = abs(frac - int(frac + 0.5))
            if delta < 1.0e-5:
                in_span = True
        return in_span

    def __match_times(self, time_span, times):
        indx = []
        for i_time in range(len(times)):
            time = times[i_time]
            if self.__inspan(time_span, time.get_mjd()):
                indx.append(i_time)
        return indx

    def create_selection(self, property, criterion):
        """
        Create a new observer containing a selection of the present observer
        based on the given time span.

        :param property: property to select on
        :param criterion: selection criterion corresponding to "property"

        :return: stochastic observer containing the required selection.
        """

        if property == "id":
            return PandasObserver(clone=[None, self, criterion, None])
        else:
            raise ValueError("only property id is supported not :" + property)

    def create_selection(self, time_span=None, property=None, criterion=None, masks=None):
        """
        Create a new observer containing a selection of the present observer
        based on the given time span.

        :param time_span: time span with selection.
        :param property: property to select on
        :param criterion: selection criterion corresponding to "property"
        :param masks: dictionary with masks for timeseries
        :return: stochastic observer containing the required selection.
        """
        if property and property != "id":
            raise ValueError("only propery id is suppored not :" + property)

        return PandasObserver(clone=[time_span, self, criterion, masks])

    def get_times(self):
        """
        Get all different times in increasing order. There is at least one observation
        for each time.

        :return: some type of vector containing the times
        """
        times_mjd = []

        # First contatinate all times
        all_times = []
        for label in self.labels:
            all_times += self.all_timeseries[label].get_times()

        # Sort and unique all times
        times_sorted = np.sort(all_times)
        unique_times = np.unique(times_sorted)
        for nptime in unique_times:
            # actually we need to store MJD but that is a lot of work now...
            times_mjd.append(PyTime(nptime))

        return times_mjd

    def get_standard_deviation(self):
        """
        get the observed values

        :return: values
        """
        std = []
        for label in self.labels:
            std += self.all_timeseries[label].get_std()
        return std

    def get_count(self):
        """
        Total number of observations.

        :return: the number of observations.
        """
        return len(self.get_values())

    def get_observation_descriptions(self):
        """
        Get the observation descriptions.

        :return: observation descriptions which are compatible with the used model instance
        """

        all_times = []
        all_id = []
        for label in self.labels:
            times = self.all_timeseries[label].get_times()
            observation_id = [label] * len(times)

            all_times += times
            all_id += observation_id

        return PandasObservationDescriptions(all_times, all_id)

    def get_sqrt_covariance(self):
        """
        Get the covariance matrix for the stochastic observations.

        :return: the covariance matrix as numpy array.
        """
        n = len(self.labels)
        R = np.zeros((n,n))
        for i, label in enumerate(self.labels):
            R[i,i] = self.all_timeseries[label].get_std()[0] ** 2

        return R
        # raise NotImplementedError("not implemented")

    def get_realizations(self):
        """
        Get realization values for all observations, for one ensemble member.

        :return: the realizations.
        """
        vals = self.get_values()
        stds = self.get_standard_deviation()
        noises = np.random.normal(0.0, 1.0, len(stds))
        realization = [val + std * noise for val, std, noise in zip(vals, stds, noises)]
        return realization

    def get_values(self):
        """
        get the observed values

        :return: values
        """
        values = []
        for label in self.labels:
            values += self.all_timeseries[label].get_values()
        return values

    def get_values2(self, user_labels):
        """ Gets the values for one or more labels"""
        values = []
        for label, idx in zip(user_labels, range(len(user_labels))):
            values += [self.all_timeseries[label].get_values()]

        for i in range(1, len(values)):
            if len(values[0]) != len(values[i]):
                raise ValueError("Cannot combine timeseries. Length is not equal. "+user_labels[0]+" has length " +
                                 str(len(values[0])) + " and "+user_labels[i]+" has length "+str(len(values[i])))

        numpy_array = np.array(values)
        transpose = numpy_array.T
        values = transpose.tolist()
        return values
