import pytest
assert pytest   #Just to make pyflake happy on a false positive for not using pytest
import unittest as need_assertions
from openda.observers.PandasTimeseriesObserver import PandasTimeseriesObserver

assertions = need_assertions.TestCase('__init__')

def test_observer_1():
    values = [[1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3]]
    times =[ 0.0, 1.0, 2.0]
    labels = ['obs1', 'obs2', 'obs3']
    std = [0.1, 0.2, 0,3]
    indices = [0, 1, 2]

    obs = PandasTimeseriesObserver(values, times, labels, std, indices)
    times = obs.get_times()
    nobs = obs.get_count()
    assertions.assertAlmostEqual(times[0], 0.0)
    assertions.assertAlmostEqual(times[1], 1.0)
    assertions.assertAlmostEqual(times[2], 2.0)
    assertions.assertEqual(nobs,9)
