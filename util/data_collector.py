import time
from multiprocessing import Queue

import h5py
import numpy as np
import pandas as pd
from definitions import get_absolute_path, file_exists


class DataCollector:

    def __init__(self, BASE_FILEPATH='data', currency_pairs=['BTC_USDT'],
                 start_dates=[1405699200], end_dates=[9999999999], time_periods=[300], overwrite=False):
        self.filepath = get_absolute_path(BASE_FILEPATH + '/pair_data_unmodified.h5')
        self.BASE_URL = 'https://poloniex.com/public?command=returnChartData'
        self.time_periods = time_periods
        self.currency_pairs = currency_pairs
        self.start_dates = start_dates
        self.last_dates = start_dates
        self.end_dates = end_dates
        self.overwrite = overwrite
        self.h5py_file = self.create_or_read_h5py_file()
        self.stop = False

    def run(self):
        while not self.stop:
            self.update_h5py_file()
            print('Hallo')
            time.sleep(min(self.time_periods)/100)  # TODO: bad, change later?

    def update_h5py_file(self):
        for i in range(len(self.currency_pairs)):
            dset_name = self.currency_pairs[i]
            pair_exists_in_file = dset_name in self.h5py_file
            if pair_exists_in_file:
                dset = self.h5py_file[dset_name]
                self.last_dates[i] = dset[-1, 1] + 1  # get last line's date
                url = self.build_url(self.currency_pairs[i], self.last_dates[i], self.end_dates[i],
                                     self.time_periods[i])
                print(url)
                df = pd.read_json(url, convert_dates=False) # TODO: Catch website errors! + forward
                if len(df) == 1 & (df == 0).all(axis=1)[0]:  # No new data?
                    print('no new data')
                    continue
                values = df.values
                dset.resize((dset.shape[0] + values.shape[0]), axis=0)
                dset[-values.shape[0]:] = values
                print(dset[-1,:])
            else:
                url = self.build_url(self.currency_pairs[i], self.last_dates[i], self.end_dates[i],
                                     self.time_periods[i])
                print(url)
                df = pd.read_json(url, convert_dates=False)
                if len(df) == 1 & (df == 0).all(axis=1)[0]:  # No new data?
                    print('no new data')
                    continue
                self.last_dates[i] = df.tail(1)['date'] + 1
                dset = self.h5py_file.create_dataset(dset_name, data=df.values, maxshape=(None, df.shape[1]))
                print('new data')

    def create_or_read_h5py_file(self):
        if self.overwrite:
            return h5py.File(self.filepath, 'w')
        else:
            return h5py.File(self.filepath, 'a')

    def build_url(self, pair, start_date, end_date, time_period) -> str:
        return self.BASE_URL + '&currencyPair=' + pair + '&start=' + str(start_date) + '&end=' + str(end_date) + '&period=' + str(
            time_period)

    def stop(self):
        self.stop = True


