import multiprocessing as mp
import logging
from itertools import repeat

import h5py

from bot import API
from bot import API_offline
import directory


class DataCollector(): #TODO: drop columns

    def __init__(self, relative_filepath, currency_pairs=['BTC_USDT'], start_dates=[1405699200],
                 end_dates=[9999999999], time_period=300, overwrite=False, offline=False):
        self.filepath = directory.get_absolute_path(relative_filepath)
        directory.ensure_directory(self.filepath)
        self.BASE_URL = 'https://poloniex.com/public?command=returnChartData'
        self.time_period = time_period
        self.currency_pairs = currency_pairs #All crypto pairs you want to download and predict
        self.start_dates = start_dates #Starting date list for pairs
        self.last_dates = start_dates  # The last dates that had data available
        self.end_dates = end_dates
        self.overwrite = overwrite #redownload data every time or save progress
        self.create_h5py_file_and_datasets() # make sure h5 file exists and all datasets are created if needed
        self.offline = offline

    def download_and_save(self):
        """
        Main method to call. downloads and saves all data with use of multiple processes
        """
        self._update_latest_dates()
        param_list = list(zip(self.currency_pairs, self.last_dates, self.end_dates, repeat(self.time_period), repeat(self.offline))) #make a list with all neede parameters
        with mp.Pool(processes=4) as pool:
            res = pool.starmap_async(func=data_downloader, iterable=param_list)
            pool.close() #close pool to not allow new tasks (not really needed)
            pool.join() #wait for pool
        results = res.get(timeout=None) #get all results from the pool
        with self._read_database_file() as datafile: #save all data pair wise in the file
            for result in results:
                if result is None: continue
                pair, data = result
                dset = datafile[pair]
                dset.resize((dset.shape[0] + data.shape[0]), axis=0)
                dset[-data.shape[0]:] = data
                dset.flush()
                datafile.flush()
                print('completed writing ' + pair + ' data to file')
            datafile.close()



    def _update_latest_dates(self):
        """
        Reads the pair_data_unmodified.h5 file, checks the latest saved dates and updates last_dates accordingly
        """
        with self._read_database_file() as file:
            for i in range(len(self.currency_pairs)):
                dset = file[self.currency_pairs[i]]
                if not len(dset) == 0:
                    date = dset[-1, 1]#TODO: might replace second index with variable
                    self.last_dates[i] = date + 1

    def _read_database_file(self):
        return h5py.File(self.filepath, libver='latest')

    def create_h5py_file_and_datasets(self):
        """
        Creates or reads file (overwrites if desired) and makes sure all datasets exist.
        """
        directory.ensure_directory(self.filepath)
        if self.overwrite:
            database = h5py.File(self.filepath, 'w', libver='latest', swmr=False)
        else:
            database = h5py.File(self.filepath, libver='latest', swmr=False)
        #Create Datasets
        for pair in self.currency_pairs:
            if pair in database:
                print('Pair: ' + pair + 'already exists in' + str(database) + '...continuing')
            else:
                print('Pair: ' + pair + 'was not found in' + str(database) + '...creating new dataset')
                dset = database.create_dataset(pair, (0, 8), maxshape=(None, 8), dtype='float64')
                dset.flush()
        database.swmr_mode = True #switch on swmr mode to allow multiple readers at once
        database.close()

    #utility methods for other classes ༼ つ ◕_◕ ༽つ

    def get_latest_dates(self):
        """
        :return: the last available date in a dict with pairs/dset_names as key
        """
        dates = dict()
        with self._read_database_file() as file:
            for i in range(len(self.currency_pairs)):
                dset = file[self.currency_pairs[i]]
                if not len(dset) == 0:
                    date = dset[-1, 1]#TODO: might replace second index with variable
                    dates[self.currency_pairs[i]] = date
        file.close()
        return dates

    def get_latest_date_for_pair(self, pair):
        """
        only get one last date for the specified pair
        :param pair: the ctypto pair/dset name
        :return: last date
        """
        date = 0
        with self._read_database_file() as file:
            dset = file[pair]
            if not len(dset) == 0:
                date = dset[-1, 1]
        file.close()
        return date

    def get_original_data(self, pair):
        """
        get all available data for the specified pair
        :param pair:
        :return: data as np.array
        """
        with self._read_database_file() as file:
            dset = file[pair]
            data = dset[:, :]
        file.close()
        return data

    def get_latest_closing_price(self, pair):
        """
        get only the latest closing price for the specified pair
        :param pair:
        :return:
        """
        with self._read_database_file() as file:
            dset = file[pair]
            data = dset[-1, 0]
        file.close()
        return data

def data_downloader(pair, last_date, end_date, time_period, offline):
    """
    The worker methods which collects crypto data for the given pair and puts it back to queue
    :param queue:
    :param pair_index:
    """
    print('requesting newest data for', last_date)
    if offline:
        df = API_offline.receive_pair_data(pair, last_date, end_date, time_period)
    else:
        df = API.receive_pair_data(pair, last_date, end_date, time_period)
    if df is None:
        print('no new data for downloader[' + pair + ']. Latest date: ' + str(last_date))
    elif len(df) == 0:
        print('no new data for downloader[' + pair + ']. Latest date: ' + str(last_date))
    elif len(df) == 1 and not offline and (df == 0).all(axis=1)[0]:  # No new data?
        print('no new data for downloader[' + pair + ']. Latest date: ' + str(last_date))

    else:
        last_date = df['date'].tail(1).values[0] + 1  # +1 so that request does not get the same again
        print('New Data found for downloader[' + pair + ']. New latest date: ' + str(last_date))
        return (pair, df.values)
