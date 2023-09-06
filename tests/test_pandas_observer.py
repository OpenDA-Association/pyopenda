import pytest
assert pytest   #Just to make pyflake happy on a false positive for not using pytest
import os
import unittest as need_assertions
from openda.observers.pandas_observer import PandasObserver

assertions = need_assertions.TestCase('__init__')

def test1():
    config = {
        'store_name': None,
        'working_dir': '.',
        'config_file': 'obs.csv',
        'labels': ['obs1', 'obs2'],
        'std': [0.3, 0.4]
    }

    obs = PandasObserver(config=config, scriptdir=os.path.dirname(__file__))
    vals = obs.get_values()
    vals_noise = obs.get_realizations()
    std = obs.get_standard_deviation()
    times = obs.get_times()
    obs_descr = obs.get_observation_descriptions()
    keys = obs_descr.get_property_keys()
    ids = obs_descr.get_properties('id')


    assert True
