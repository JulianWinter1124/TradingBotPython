from urllib.error import URLError, HTTPError
import logging
import os
import time
from multiprocessing import Process, Queue

import h5py
import pandas as pd

from definitions import get_absolute_path


class DataCollector(): #TODO: drop columns

    def __init__(self, BASE_FILEPATH='data', currency_pairs=['BTC_USDT'], start_dates=[1405699200],
                 end_dates=[9999999999], time_periods=[300], overwrite=False, log_level=logging.INFO):
        #super(DataCollector, self).__init__()
        logging.getLogger().setLevel(log_level)
        self.filepath = get_absolute_path(BASE_FILEPATH + '/pair_data_unmodified.h5')
        self.BASE_URL = 'https://poloniex.com/public?command=returnChartData'
        self.time_periods = time_periods
        self.currency_pairs = currency_pairs
        self.start_dates = start_dates
        self.last_dates = start_dates  # The last dates that had data available
        self.end_dates = end_dates
        self.overwrite = overwrite

        #Call file creation related methods
        self.create_h5py_file()
        self.create_databases()
        self.update_latest_dates()

    def mp_worker(self, q, pair_index):
        pair, start_date, last_date, end_date, time_period = self.currency_pairs[pair_index], self.start_dates[
            pair_index], self.last_dates[pair_index], self.end_dates[pair_index], self.time_periods[pair_index]
        while True:
            url = self.build_url(pair, last_date, end_date, time_period)
            try:
                df = pd.read_json(url, convert_dates=False)  # TODO: catch errors
            except (HTTPError, URLError) as e:
                logging.error('error retrieving data. Trying again in 5 seconds.')
                time.sleep(5)
                continue
            if len(df) == 1 & (df == 0).all(axis=1)[0]:  # No new data?
                print('no new data for downloader[' + pair + ']. Latest date: ' + str(last_date))
            else:
                last_date = df['date'].tail(1).values[0] + 1 # +1 so that request does not get the same again
                self.last_dates[pair_index] = last_date
                q.put((pair, df.values))  # put data AND the currency pair
                print('New Data found for downloader[' + pair + ']. New latest date: ' + str(last_date))
            del df
            time.sleep(time_period / 10)  # TODO: find good time

    def run_unmodified_loop(self):
        n = min(len(self.currency_pairs), 6)  # 6 processes at maximum
        q = Queue()
        processes = []
        for i in range(n):
            p = Process(target=self.mp_worker, args=(q, i,))
            processes.append(p)
            p.start()
        while True:  # LUL
            pair, data = q.get() #  This is a stopping method
            file = self.read_h5py_file()
            dset = file[str(pair)]
            dset.resize((dset.shape[0] + data.shape[0]), axis=0)
            dset[-data.shape[0]:] = data
            dset.flush()
            file.flush()
            file.close()
            print('Saved data to file for pair[' + pair + ']')

    def create_databases(self):
        file = self.read_h5py_file()
        for pair in self.currency_pairs:
            if pair in file:
                print('Pair: ' + pair + 'already exists in' + str(file) + '...continuing')
            else:
                print('Pair: ' + pair + 'was not found in' + str(file) + '...creating new dataset')
                dset = file.create_dataset(pair, (0, 8), maxshape=(None, 8))
                dset.flush()
        file.swmr_mode = True

    def read_h5py_file(self):
        return h5py.File(self.filepath, 'a', libver='latest')


    def update_latest_dates(self):
        if not self.overwrite:  # Otherwise there is no need to update anything
            file = self.read_h5py_file()
            for i in range(len(self.currency_pairs)):
                dset = file[self.currency_pairs[i]]
                date = dset[-1, 1]#TODO: might replace second index with variable
                self.last_dates[i] = date + 1

    def create_h5py_file(self):
        if not os.path.exists(os.path.dirname(self.filepath)):
            try:
                os.makedirs(os.path.dirname(self.filepath))
            except OSError as exc:  # Guard against race condition
                logging.exception(exc)
        if self.overwrite:
            print("Overwriting file")
            return h5py.File(self.filepath, 'w', libver='latest')
        else:
            print("Opening or creating file")
            return h5py.File(self.filepath, 'a', libver='latest')

    def build_url(self, pair, start_date, end_date, time_period) -> str:
        return self.BASE_URL + '&currencyPair=' + pair + '&start=' + str(start_date) + '&end=' + str(
            end_date) + '&period=' + str(
            time_period)
