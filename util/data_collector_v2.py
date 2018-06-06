import time
from functools import partial
from multiprocessing import Process
from multiprocessing.pool import Pool
from queue import Queue

import h5py
import pandas as pd

from definitions import get_absolute_path


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

    def mp_worker(queue: Queue, pair_index):
        # pair, start_date, last_date, end_date, time_period = self.currency_pairs[pair_index], self.start_dates[
        #     pair_index], self.last_dates[pair_index], self.end_dates[pair_index], self.time_periods[pair_index]
        # while True:
        #     url = self.build_url(pair, start_date, end_date, time_period)
        #     df = pd.read_json(url, convert_dates=False)  # TODO: catch errors
        #     if len(df) == 1 & (df == 0).all(axis=1)[0]:  # No new data?
        #         print('no new data in worker:', pair)
        #     else:
        #         last_date = df.tail(1)['date']
        #         self.last_dates[pair_index] = last_date
        #         queue.put(df.values) #maybe
        #         print('New Data found and out into queue')
        #     time.sleep(time_period)  # Pls, dont ban me monkaS
        print('lul')

    def run_unmodified_loop(self):
        n = min(len(self.currency_pairs), 6) #6 processes at maximum
        worker_queue = Queue()
        processes = []
        for i in range(n):
            p = Process(target=self.mp_worker, args=(worker_queue, i,))
            processes.append(p)
            p.start()
        print(worker_queue.get(timeout=1))

    def run_modified_loop(self):
        print("note yet implkemeneted")

    def create_databases(self):
        for pair in self.currency_pairs:
            if pair in self.h5py_file:
                print('Pair: ', pair, 'already exists in', self.h5py_file, '...continuing')
            else:
                print('Pair: ', pair, 'was not found in', self.h5py_file, '...creating new dataset')
                dset = self.h5py_file.create_dataset(pair, (1, 8), maxshape=(None, None))
        self.h5py_file.swmr_mode = True

    def append_data_to_dataset(self, dset_name, data):
        dset = self.h5py_file[dset_name]
        dset.resize((dset.shape[0] + data.shape[0]), axis=0)
        dset[-data.shape[0]:] = data

    def create_or_read_h5py_file(self):
        if self.overwrite:
            return h5py.File(self.filepath, 'w', libver='latest')
        else:
            return h5py.File(self.filepath, 'a', libver='latest')

    def build_url(self, pair, start_date, end_date, time_period) -> str:
        return self.BASE_URL + '&currencyPair=' + pair + '&start=' + str(start_date) + '&end=' + str(
            end_date) + '&period=' + str(
            time_period)
