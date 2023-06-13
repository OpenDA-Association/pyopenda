import pytest
assert pytest   #Just to make pyflake happy on a false positive for not using pytest
import unittest as need_assertions
import numpy as np
import matplotlib.pyplot as plt

from openda.models.LorenzStochModelFactory import LorenzStochModelFactory

assertions = need_assertions.TestCase('__init__')

def test_model():
    factory = LorenzStochModelFactory(None)
    m1 = factory.get_instance(noise_config=None, main_or_ens='main')
    m2 = factory.get_instance(noise_config=None, main_or_ens='main')
    x2_0 = m2.get_state()
    x2_0[0]+=1.0
    m2.update_state(x2_0,'main')
    m3 = factory.get_instance(noise_config={'stochParameter':False, 'stochForcing':False, 'stochInit':True}, main_or_ens='main')

    time_span = m1.get_time_horizon();
    x1 = []
    x2 = []
    x3 = []
    for t in np.arange(time_span.get_start(),time_span.get_end(), 0.1):
        m1.compute(t)
        m2.compute(t)
        m3.compute(t)
        x1.append(m1.get_state())
        x2.append(m2.get_state())
        x3.append(m3.get_state())
    x1 = np.matrix(x1)
    x2 = np.matrix(x2)
    x3 = np.matrix(x3)
    plt.plot(x1[:, 1])
    plt.plot(x2[:, 1])
    plt.plot(x3[:, 1])
    plt.show()
