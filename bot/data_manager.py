

import h5py
import multiprocessing as mp

import numpy as np

from bot import API
from util import data_modifier as mod
from definitions import get_absolute_path


class DataManager(mp.Process):


    def __init__(self, input_pipe, database_output_filepath, finished_data_output_filepath, currency_pairs=['USDT_BTC'], start_dates=[1405699200],
                 end_dates=[9999999999], time_periods=[300], overwrite_download=False, use_indicators=True, use_scaling=True, drop_data_columns_indices=[], n_in=30, n_out=2):
        super(DataManager, self).__init__()
        self.database_filepath = get_absolute_path(database_output_filepath)
        self.finished_data_filepath = get_absolute_path(finished_data_output_filepath)
        self.BASE_URL = 'https://poloniex.com/public?command=returnChartData'
        self.time_periods = time_periods
        self.currency_pairs = currency_pairs
        self.start_dates = start_dates
        self.last_dates = start_dates  # The last dates that had data available
        self.end_dates = end_dates
        self.overwrite_download = overwrite_download
        self.input_pipe = input_pipe
        self.n_in = n_in
        self.n_out = n_out
        self.use_scaling = use_scaling
        self.drop_data_columns_indices = drop_data_columns_indices
        self.use_indicators = use_indicators
        self.n_completed = 0 #If you want to save progress in data processor initialize iwth something else
        self._scaler = dict()

    def run(self):
        #Create file
        if self.overwrite_download:
            database = h5py.File(self.database_filepath, 'w', libver='latest', swmr=False)
        else:
            database = h5py.File(self.database_filepath, libver='latest', swmr=False)
        #Create Datasets
        for pair in self.currency_pairs:
            if pair in database:
                print('Pair: ' + pair + 'already exists in' + str(database) + '...continuing')
            else:
                print('Pair: ' + pair + 'was not found in' + str(database) + '...creating new dataset')
                dset = database.create_dataset(pair, (0, 8), maxshape=(None, 8), dtype='float64')
                dset.flush()
        database.swmr_mode = True
        #Repeat this
        self.update_all_last_dates()
        param_list = list(zip(self.currency_pairs, self.last_dates, self.end_dates, self.time_periods))
        with mp.Pool(processes=4) as pool:
            for params in param_list:
                pool.apply_async(data_collector, args=(params,), callback=self.collector_callback)
            pool.close()
            pool.join()
        self.data_processor()


    def collector_callback(self, container):
        if not container is None:
            print(container)
            pair, data = container
            database = h5py.File(self.database_filepath, libver='latest')
            dset = database[pair]
            dset.resize((dset.shape[0] + data.shape[0]), axis=0)
            dset[-data.shape[0]:] = data
            dset.flush()
            database.flush()
            database.close()
            print('completed writing', pair, 'data to file')
            self.update_all_last_dates()

    def data_processor(self, pair, data):
        database = h5py.File(self.database_filepath, libver='latest')
        print('jkji')
        dset = database[pair]
        dset.id.refresh()
        selection_array = dset[self.n_completed:, :]  # important datas
        if len(selection_array) <= self.get_minimum_data_amount_for_timeseries():  # max(timeperiod) in out
            print('not enough data for timeseries calculation', pair)
            del selection_array
            database.close()
            return
        if self.use_indicators:
            selection_array = mod.add_indicators_to_data(selection_array)
        if self.use_scaling:
            if pair not in self._scaler:
                selection_array, self._scaler[pair] = mod.normalize_data_MinMax(selection_array)
                self.input_pipe.send(self._scaler[pair])
            else:
                selection_array = mod.normalize_data(selection_array, self._scaler)
        timeseries_data = mod.data_to_supervised_timeseries(selection_array, n_in=self.n_in, n_out=self.n_out,
                                                           drop_columns_indices=self.drop_data_columns_indices,
                                                           label_columns_indices=[0])
        self.n_completed += len(timeseries_data)
        print("Data modification finished for pair[" + pair + '] with shape', timeseries_data.shape)
        database.close()

        del selection_array
        del timeseries_data

    def update_all_last_dates(self):
        database = h5py.File(self.database_filepath, libver='latest')
        for i in range(len(self.currency_pairs)):
            dset = database[self.currency_pairs[i]]
            if not len(dset) == 0:
                date = dset[-1, 1]
                self.last_dates[i] = (date + np.float64(1.0))
        database.close()

    def get_minimum_data_amount_for_timeseries(self):
        """
        30*self.use_indicators because some indicators need 30 data points prior to t
        :return: the minimum data needed to produce at least 1 timestep
        """
        return 30*self.use_indicators + (self.n_in + self.n_out + 1) #use_indicators=0 or =1


def data_collector(params):
    pair, last_date, end_date, time_period = params
    print('requesting newest data for', last_date)
    df = API.receive_pair_data(pair, last_date, end_date, time_period)
    if len(df) == 1 & (df == 0).all(axis=1)[0]:  # No new data?
        print('no new data for downloader[' + pair + ']. Latest date: ' + str(last_date))
    else:
        last_date = df['date'].tail(1).values[0] + 1  # +1 so that request does not get the same again
        print('New Data found for downloader[' + pair + ']. New latest date: ' + str(last_date))
        return (pair, df.values)

