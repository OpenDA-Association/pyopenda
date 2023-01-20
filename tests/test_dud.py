import pytest
import unittest as need_assertions
from openda.algorithms import Dud

assertions = need_assertions.TestCase('__init__')

def f(p):
    return [p[0], p[1]]



def test_dud():
    (total_cost, p_opt, hist) = Dud.dud(f, p_start=[2, 7], p_std=[1.0, 2.0], p_pert=[0.1, 0.1], obs=[1, 33],
                                        std=[1.0, 1.0])

    assertions.assertAlmostEqual(total_cost, 0.0)
    assertions.assertAlmostEqual(p_opt[0], 1.0)
    assertions.assertAlmostEqual(p_opt[1], 33.0)

def test_dud_constr():
    (total_cost, p_opt, hist) = Dud.dud(f, p_start=[2, 7], p_std=[1.0, 2.0], p_pert=[0.1, 0.1], obs=[1, 33],
                                        std=[1.0, 1.0], l_bound=[0.0, 0.0], u_bound=[10.0, 30.0])

    assertions.assertAlmostEqual(total_cost, 4.5)
    assertions.assertAlmostEqual(p_opt[0], 1.0)
    assertions.assertAlmostEqual(p_opt[1], 30.0)
