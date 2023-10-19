import pytest


assert pytest   #Just to make pyflake happy on a false positive for not using pytest
import os
import unittest as need_assertions
from openda.costFunctions.JObjects import PyTime
from openda.observers.pandas_observer import PandasObserver


assertions = need_assertions.TestCase('__init__')

@pytest.fixture
def generate_obs():
    config = {
        'store_name': None,
        'working_dir': './observations',
        'config_file': 'obs_storm_Eunice_5min.csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.6, 0.6, 0.6, 0.6]
    }
    obs = PandasObserver(config=config, scriptdir=os.path.dirname(__file__))
    return obs

def test_loaded_all(generate_obs):
    vals = generate_obs.get_values()
    times = generate_obs.get_times()
    assert len(vals)//len(times) == 4

def test_std(generate_obs):
    stds = generate_obs.get_standard_deviation()
    for std in stds:
        assert std == 0.6

def test_ids(generate_obs):
    n = len(generate_obs.get_times())
    obs_descr = generate_obs.get_observation_descriptions()
    ids = obs_descr.get_properties('id')
    for i, idx in enumerate(ids):
        assert idx == generate_obs.labels[i//n]

def test_keys(generate_obs):
    obs_descr = generate_obs.get_observation_descriptions()
    keys = obs_descr.get_property_keys()
    assert keys == ['id', 'time']


@pytest.fixture
def generate_obs_now():
    config = {
        'store_name': None,
        'working_dir': './observations',
        'config_file': 'obs_storm_Eunice_5min.csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.6, 0.6, 0.6, 0.6]
    }

    obs = PandasObserver(config=config, scriptdir=os.path.dirname(__file__))
    times = obs.get_times()
    obs_now = obs.create_selection(time_span=PyTime(times[0].get_mjd(), times[1].get_mjd()))
    return obs_now

def test_loaded_all_now(generate_obs_now):
    vals_now = generate_obs_now.get_values()
    assert len(vals_now) == 4

def test_std_now(generate_obs_now):
    stds_now = generate_obs_now.get_standard_deviation()
    for std_now in stds_now:
        assert std_now == 0.6

def test_ids_now(generate_obs_now):
    obs_descr_now = generate_obs_now.get_observation_descriptions()
    ids_now = obs_descr_now.get_properties('id')
    for id_now in ids_now:
        assert id_now in generate_obs_now.labels

def test_keys_now(generate_obs_now):
    obs_descr = generate_obs_now.get_observation_descriptions()
    keys = obs_descr.get_property_keys()
    assert keys == ['id', 'time']
