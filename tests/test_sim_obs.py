import pytest
assert pytest   #Just to make pyflake happy on a false positive for not using pytest
import os
import unittest as need_assertions
from openda.observers.pandas_observer import PandasObserver
from openda.costFunctions.JObjects import PyTime

assertions = need_assertions.TestCase('__init__')

def test1():
    config = {
        'store_name': None,
        'working_dir': './../',
        'config_file': 'obs (simulated).csv',
        'labels': ['h0', 'h25', 'h50', 'h75', 'h99'],
        'std': [0.2, 0.3, 0.3, 0.3, 0.3]
    }

    obs = PandasObserver(config=config, scriptdir=os.path.dirname(__file__))
    vals = obs.get_values()
    vals_noise = obs.get_realizations()
    std = obs.get_standard_deviation()
    times = obs.get_times()
    obs_descr = obs.get_observation_descriptions()
    keys = obs_descr.get_property_keys()
    ids = obs_descr.get_properties('id')

    obs_now = obs.create_selection(time_span=PyTime(times[2].get_mjd(), times[3].get_mjd()))
    vals_n = obs_now.get_values()
    vals_noise_n = obs_now.get_realizations()
    std_n = obs_now.get_standard_deviation()
    times_n = obs_now.get_times()
    obs_descr_n = obs_now.get_observation_descriptions()
    keys_n = obs_descr_n.get_property_keys()
    ids_n = obs_descr_n.get_properties('id')

    assert True

test1()