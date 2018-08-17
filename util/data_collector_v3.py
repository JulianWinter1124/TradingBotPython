import multiprocessing
import logging
import os
import time
from multiprocessing import Process, Queue

import h5py

from bot import API
from definitions import get_absolute_path


class DataCollector(Process): #TODO: drop columns

    def __init__(self,output_filepath, currency_pairs=['BTC_USDT'], start_dates=[1405699200],
                 end_dates=[9999999999], time_periods=[300], overwrite=False, log_level=logging.INFO):
        super(DataCollector, self).__init__()
        #logging.getLogger().setLevel(log_level)
        self.filepath = get_absolute_path(output_filepath)
        self.BASE_URL = 'https://poloniex.com/public?command=returnChartData'
        self.time_periods = time_periods
        self.currency_pairs = currency_pairs
        self.start_dates = start_dates
        self.last_dates = start_dates  # The last dates that had data available
        self.end_dates = end_dates
        self.overwrite = overwrite

        #Call file creation related methods

    def mp_worker(self, queue, pair_index):
        """
        The worker methods which collects crypto data for the given pair and puts it back to queue
        :param queue:
        :param pair_index:
        """
        pair, start_date, last_date, end_date, time_period = self.currency_pairs[pair_index], self.start_dates[
            pair_index], self.last_dates[pair_index], self.end_dates[pair_index], self.time_periods[pair_index]
        while True:
            df = API.receive_pair_data(pair, last_date, end_date, time_period)
            if len(df) == 1 & (df == 0).all(axis=1)[0]:  # No new data?
                print('no new data for downloader[' + pair + ']. Latest date: ' + str(last_date))
            else:
                last_date = df['date'].tail(1).values[0] + 1 # +1 so that request does not get the same again
                self.last_dates[pair_index] = last_date
                queue.put((pair, df.values), block=True)  # put data AND the currency pair
                print('New Data found for downloader[' + pair + ']. New latest date: ' + str(last_date)) #TODO: implement proper multiprocessing logging
            del df
            time.sleep(time_period / 10)  # TODO: find good time

    def run(self):
        """
        main loop of the datacollector. This will start all subprocess collectors for each pair and also receive their data
        via queue to put in pair_data_unmodified.h5
        """
        file = self.create_h5py_file_and_datasets()
        file.swmr_mode = True
        assert file.swmr_mode
        file.close()
        self.update_latest_dates()
        n = min(len(self.currency_pairs), 6)  # 6 processes at maximum
        q = multiprocessing.Queue()
        processes = []
        for i in range(n):
            p = Process(target=self.mp_worker, args=(q, i,)) #queue is not properly shared on windows
            processes.append(p)
            p.start()
        while True:
            print("Waiting for new data...")
            pair, data = q.get() #  This is a stopping method
            print('data received')
            with h5py.File(self.filepath, libver='latest', swmr=True) as file:
                dset = file[str(pair)]
                dset.resize((dset.shape[0] + data.shape[0]), axis=0)
                dset[-data.shape[0]:] = data
                dset.flush()
                file.flush()
                file.close()
                print('Saved data to file for pair[' + pair + ']')

    def update_latest_dates(self):
        """
        Reads the pair_data_unmodified.h5 file and checks the latest saved dates
        """
        if not self.overwrite:  # Otherwise there is no need to update anything
            with self.read_database_file() as file:
                for i in range(len(self.currency_pairs)):
                    dset = file[self.currency_pairs[i]]
                    if not len(dset) == 0:
                        date = dset[-1, 1]#TODO: might replace second index with variable
                        self.last_dates[i] = date + 1

    def read_database_file(self):
        return h5py.File(self.filepath, libver='latest')

    def create_h5py_file_and_datasets(self):
        if not os.path.exists(os.path.dirname(self.filepath)):
            try:
                os.makedirs(os.path.dirname(self.filepath))
            except OSError as exc:  # Guard against race condition
                logging.exception(exc)

        if self.overwrite:
            print("Overwriting file")
            file = h5py.File(self.filepath, 'w', libver='latest')
        else:
            print("Opening or creating file")
            file = h5py.File(self.filepath, libver='latest')
        for pair in self.currency_pairs:
            if pair in file:
                print('Pair: ' + pair + 'already exists in' + str(file) + '...continuing')
            else:
                print('Pair: ' + pair + 'was not found in' + str(file) + '...creating new dataset')
                dset = file.create_dataset(pair, (0, 8), maxshape=(None, 8))
                dset.flush()
        return file

    def build_url(self, pair, start_date, end_date, time_period) -> str:
        return self.BASE_URL + '&currencyPair=' + pair + '&start=' + str(start_date) + '&end=' + str(
            end_date) + '&period=' + str(
            time_period)
