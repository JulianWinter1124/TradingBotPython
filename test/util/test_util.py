import itertools

import pandas as pd
import numpy as np
import unittest

from util import data_enhancer


class DataEnhancerTest(unittest.TestCase):
    def test_make_binary_labels(self):
        test1 = [1, 2, 3, 4, 5, 6, 7, 8]
        test1_expect = [1, 1, 1, 1, 1, 1, 1]
        calc1 = data_enhancer.make_binary_labels(test1)
        self.assertEqual(test1_expect, calc1)

        test2 = [8, 7, 6, 5, 4, 3, 2, 1]
        test2_expect = [-1, -1, -1, -1, -1, -1, -1]
        calc2 = data_enhancer.make_binary_labels(test2)
        self.assertEqual(test2_expect, calc2)

        test3 = [8, 7, 6, 7, 4, 3, 2, 3]
        test3_expect = [-1, -1, 1, -1, -1, -1, 1]
        calc3 = data_enhancer.make_binary_labels(test3)
        self.assertEqual(test3_expect, calc3)
        print(calc3)

    def test_make_ranged_labels(self):
        test1 = [1, 2, 1, 3, 2, 2, 1]
        test1_expect = [0.5, -0.5, 1, -0.5, 0, -0.5]
        calc1 = data_enhancer.make_ranged_labels(test1)
        self.assertEqual(test1_expect, calc1)

        test3 = [1, 2, 3, 4, 5]
        test3_expect = [0.25, 0.25, 0.25, 0.25]
        calc3 = data_enhancer.make_ranged_labels(test3)
        self.assertEqual(test3_expect, calc3)


import util.data_modifier as dm
from definitions import ROOT_DIR
import os


class DataModifierTest(unittest.TestCase):

    def test_data_to_supervised(self):
        df = pd.read_csv(os.path.join(ROOT_DIR, 'data/BTCUSD300.csv'))
        for i, j in itertools.product(range(1, 3), range(3)):
            timeseries = dm.data_to_supervised(df, n_in=i, n_out=j, drop_columns_indices=[5],
                                               label_columns_indices=[0])
            print('###############################################################')
            print(timeseries.head(1))

        for i, j in itertools.product(range(1, 2), range(2)):
            timeseries = dm.data_to_supervised(df, n_in=i, n_out=j, drop_columns_indices=[5],
                                               label_columns_indices=[0, 6])
            print('###############################################################')
            print(timeseries.head(1))

    def test_add_indicators(self):
        df = pd.read_csv(os.path.join(ROOT_DIR, 'data/BTCUSD300.csv'))
        print('normal:', df.shape)
        print(dm.add_SMA_indicator_to_data(df.values).shape)
        print(dm.add_RSI_indicator_to_data(df.values).shape)
        print(dm.add_BBANDS_indicator_to_data(df.values).shape)
        print(dm.add_OBV_indicator_to_data(df.values, [0, 6]).shape)

        test = pd.DataFrame(dm.add_BBANDS_indicator_to_data(df.values))
        print(dm.data_to_supervised(test))

    def test_drop_NaN_rows(self):
        df = pd.read_csv(os.path.join(ROOT_DIR, 'data/BTCUSD300.csv'))
        sma = dm.add_SMA_indicator_to_data(df.values, timeperiod=40)
        print(len(sma) - len(dm.drop_NaN_rows(sma)))
        sma = dm.add_SMA_indicator_to_data(df.values, timeperiod=30)
        print(len(sma) - len(dm.drop_NaN_rows(sma)))
