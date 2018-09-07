import itertools

import pandas as pd
import numpy as np
import unittest

import definitions
import util.data_modifier as dm
import os


class DataModifierTest(unittest.TestCase):

    def test_data_to_supervised_timeseries(self):
        data = np.array([[1,-1],
                         [2,-2],
                         [3,-3],
                         [4,-4],
                         [5,-5],
                         [6,-6],
                         [7,-7],
                         [8,-8],
                         [9,-9],
                         [10,-10],
                         [11,-11]])
        print(pd.DataFrame(data))
        print(pd.DataFrame(dm.data_to_supervised_timeseries(data, n_in=3, n_out=2, n_out_jumps=3, label_columns_indices=[0])))
        print(pd.DataFrame(
            dm.data_to_supervised_timeseries(data, n_in=1, n_out=1, n_out_jumps=3, label_columns_indices=[0])))
        print(pd.DataFrame(
            dm.data_to_supervised_timeseries(data, n_in=1, n_out=0, n_out_jumps=3, label_columns_indices=[0])))
        print(pd.DataFrame(
            dm.data_to_supervised_timeseries(data, n_in=2, n_out=0, n_out_jumps=0, label_columns_indices=[0])))


    def test_add_indicators(self):
        df = pd.read_csv(definitions.get_absolute_path('data/BTCUSD300.csv'))
        print('normal:', df.shape)
        print(dm.add_SMA_indicator_to_data(df.values).shape)
        print(dm.add_RSI_indicator_to_data(df.values).shape)
        print(dm.add_BBANDS_indicator_to_data(df.values).shape)
        print(dm.add_OBV_indicator_to_data(df.values, [0, 6]).shape)

        test = pd.DataFrame(dm.add_BBANDS_indicator_to_data(df.values))
        print(dm.data_to_supervised_timeseries(test))

    def test_drop_NaN_rows(self):
        df = pd.read_csv(definitions.get_absolute_path('data/BTCUSD300.csv'))
        sma = dm.add_SMA_indicator_to_data(df.values, timeperiod=40)
        print(len(sma) - len(dm.drop_NaN_rows(sma)))
        sma = dm.add_SMA_indicator_to_data(df.values, timeperiod=30)
        print(len(sma) - len(dm.drop_NaN_rows(sma)))

