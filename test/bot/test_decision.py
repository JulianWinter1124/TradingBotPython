import unittest

import numpy as np

from bot_ai import decision


class CollectorTest(unittest.TestCase):

    def test_calc_tanh_diff(self):
        print(decision.calc_tanh_diff(0, 1))
        print(decision.calc_tanh_diff(1, 1))
        print(decision.calc_tanh_diff(2, 2))
        print(decision.calc_tanh_diff(3, 2))

    def test_decide_action_on_prediction(self):
        print(decision.decide_action_on_prediction(np.array([[10.0,10.02,10.03,10.0,10.08,10.01]]), 0.5))
        print(decision.decide_action_on_prediction(np.array([[10, 10.001, 10.002, 10.003]]), 0.5))
        print(decision.decide_action_on_prediction(np.array([[1.0, 1.0001, 1.002111]]), 0.5))
        print(decision.decide_action_on_prediction(np.array([[10, 9, 8]]), 0.5))
        print(decision.decide_action_on_prediction(np.array([[1, 1.005, 4]]), 1))
        print(decision.decide_action_on_prediction(np.array([[2, 2.005, 1]]), 1))
        print(decision.decide_action_on_prediction(np.array([[2, 1.95]]), 1))
        print(decision.decide_action_on_prediction(np.array([[2, 1.95, 1]]), 1))
        print(decision.decide_action_on_prediction(np.array([[2, 1.95, 1, 4]]), 1))