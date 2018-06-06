import os

import h5py

import pandas as pd

from util import data_enhancer as de


class DataSequencer():

    def __init__(self, filepath, time_period, currency_pair, start_date=1405699200, end_date=9999999999, expand=True,
                 overwrite=True, batch_size=8):
        self.filepath = filepath
        self.BASE_URL = 'https://poloniex.com/public?command=returnChartData'
        self.time_period = time_period
        self.currency_pair = currency_pair
        self.start_date = start_date
        self.last_date = start_date
        self.end_date = end_date
        self.expand = expand
        self.BATCH_SIZE = batch_size
        self.dset_name = 'data'
        if overwrite or not os.path.isfile(self.filepath):
            self.create_h5py_unmodified()

    def build_url(self, start_date, end_date):
        return self.BASE_URL + '&currencyPair=' + self.currency_pair + '&start=' + start_date + '&end=' + end_date + '&period=' + self.time_period

    def create_h5py_unmodified(self):
        df = pd.read_json(self.build_url(self.last_date, self.end_date), convert_dates=False)
        self.last_date = df.tail(1)['date'] + 1  # +1 so that last column is not saved twice
        with h5py.File(self.filepath, 'a') as f:
            dset = f.create_dataset(self.currency_pair, data=df.values)
            f.close()

    def update_dates_and_h5py(self):  # Call this after each epoch? or after x batches?
        df = pd.read_json(self.build_url(self.last_date, self.end_date), convert_dates=False)
        self.last_date = df.tail(1)['date'] + 1  # +1 so that last column is not saved twice
        values = df.values
        with h5py.File(self.filepath, 'a') as hf:
            hf[self.dset_name].resize((hf[self.dset_name].shape[0] + values.shape[0]), axis=0)
            hf[self.dset_name][-values.shape[0]:] = values

    def data_generator(self, start_index=0):
        with h5py.File(self.filepath, 'r') as hf:
            i = start_index
            while True:
                df = pd.DataFrame(hf[self.dset_name][i:i + self.BATCH_SIZE])
                i += self.BATCH_SIZE
                x, y = de.supervised_to_split_nparrays(de.series_to_supervised(df))
                yield (x, y)
