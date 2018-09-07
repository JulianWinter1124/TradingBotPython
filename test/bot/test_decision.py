import unittest

import numpy as np

from bot.simulation import Simulation
from bot_ai import decision


class CollectorTest(unittest.TestCase):

    def test_calc_tanh_diff(self):
        self.assertEqual(decision.calc_tanh_diff(0, 1), 0.0)
        self.assertEqual(decision.calc_tanh_diff(0, 2), 0.0)
        self.assertEqual(decision.calc_tanh_diff(1, 1), np.tanh(0.5))
        self.assertEqual(decision.calc_tanh_diff(3, 3), np.tanh(0.5))
        self.assertTrue(decision.calc_tanh_diff(1, 2) < np.tanh(0.5))
        self.assertTrue(decision.calc_tanh_diff(10.08, 10.0)>np.tanh(0.5))

    def test_decide_action_on_prediction(self):
        state = Simulation(100, False)
        pred = np.array([[10.0,10.02,10.03,10.0,10.08,10.01]])
        action = decision.decide_action_on_prediction('USDT_BTC', pred, state, 1.0, True, np.tanh(0.5))
        self.assertEqual(action[1], 'buy')
        pred = np.array([[10, 10.001, 10.002, 10.003]])
        action = decision.decide_action_on_prediction('USDT_BTC', pred, state, 1.0, True, np.tanh(0.5))
        self.assertEqual(action[1], 'hold')
        pred = np.array([[10.0, 10.00, 10.00]])
        action = decision.decide_action_on_prediction('USDT_BTC', pred, state, 1.0, False, np.tanh(0.5))
        self.assertEqual(action[1], 'buy')
        pred = np.array([[10, 9, 8]])
        action = decision.decide_action_on_prediction('USDT_BTC', pred, state, 1.0, True, np.tanh(0.5))
        self.assertEqual(action[1], 'sell')
        pred = np.array([[2, 2.0005, 1]])
        action = decision.decide_action_on_prediction('USDT_BTC', pred, state, 1.0, True, np.tanh(0.5))
        self.assertEqual(action[1], 'hold')
        pred = np.array([[2, 1.9999, 4]])
        action = decision.decide_action_on_prediction('USDT_BTC', pred, state, 1.0, True, np.tanh(0.5))
        self.assertEqual(action[1], 'hold')
        pred = np.array([[2, 1.95, 1, 4]])
        action = decision.decide_action_on_prediction('USDT_BTC', pred, state, 1.0, True, np.tanh(0.5))
        self.assertEqual(action[1], 'sell')