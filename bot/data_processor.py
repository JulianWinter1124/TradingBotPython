import logging
import multiprocessing as mp
from collections import defaultdict
from itertools import repeat

import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import bot.data_modifier as dm
import directory
logger = logging.getLogger('data_processor')

class DataProcessor():

    def __init__(self, database_filepath, output_filepath, use_indicators, use_scaling, drop_data_columns_indices, label_column_indices, n_in, n_out, n_out_jumps, overwrite_scaler):
        self.n_out_jumps = n_out_jumps
        self.n_in = n_in
        self.n_out = n_out
        self.use_scaling = use_scaling
        self.drop_data_columns_indices = drop_data_columns_indices
        self.label_column_indices = label_column_indices
        self.use_indicators = use_indicators
        self.filepath = directory.get_absolute_path(output_filepath)
        self.database_filepath = directory.get_absolute_path(database_filepath)
        self._scaler = dict()
        self.n_completed = defaultdict(lambda : 0)
        self.overwrite_scaler = overwrite_scaler
        self.scaler_base_filepath = directory.get_absolute_path('datascaler/')
        self.create_h5py_output_file_and_databases() #Create all databases
        self.update_completed_and_scaler() #Update completed counter and scalers if existend

    def process_and_save(self):
        """
        Processes all data in unmodified_data.h5 by scaling and appending all indicators to the data.
        Saves everything by pairname to self.output_filepath
        """
        database = self.read_h5py_database_file()
        selection_arrays = []
        database_keys = database.keys()
        n = len(database_keys)
        for dset_name in database_keys: #For all available pairs in unmodified data:
            dset = database[dset_name]
            selection_array = dset[self.n_completed[dset_name]:, :] #only select data that is not transformed into a timeseries yet
            if len(selection_array) <= self.get_minimum_data_amount_for_timeseries():  # if not enough data is available, skip this pair
                print(len(selection_array))
                logger.warning('not enough data for timeseries calculation %s' % dset_name)
                selection_array = None
            selection_arrays.append(selection_array)
        param_list = list(zip(database_keys, selection_arrays, self._scaler.values(), repeat(self.use_indicators), repeat(self.use_scaling),
                              repeat(self.drop_data_columns_indices), repeat(self.label_column_indices), repeat(self.n_in),
                              repeat(self.n_out), repeat(self.n_out_jumps))) #zip all params into a list for use in multiprocessing Pool
        param_list = [x for x in param_list if x[1] is not None] #filter out any entries that contain no selection array
        with mp.Pool(processes=4) as pool:
            res = pool.starmap_async(func=produce_modified_data, iterable=param_list) #Call produce_modified_data with the params
            pool.close()
            pool.join()

        results = res.get(timeout=None)
        finished_file = self.read_h5py_output_file()
        for result in results:
            dset_name, scaler, data = result
            self.n_completed[dset_name] += len(data)
            dset = finished_file[dset_name]
            dset.resize(max(dset.shape[1], data.shape[1]), axis=1)
            dset.resize((dset.shape[0] + data.shape[0]), axis=0)
            dset[-data.shape[0]:] = data
            dset.flush()
            finished_file.flush()
            if self._scaler[dset_name] is None:
                logger.info('saving new {} scaler'.format(dset_name))
                self._scaler[dset_name] = scaler
                self._save_scaler(dset_name)
            else:
                logger.info("{} scaler is old".format(dset_name))
            logging.info('completed writing ' + dset_name + ' timeseries data to file')
        finished_file.close()

    def get_number_of_features(self):
        """
        :return: The number of columns for a single point in time, including indicator column number if present
        """
        return self.use_indicators*6 + 8 - len(self.drop_data_columns_indices)  # 6=number of indicator columns

    def get_minimum_data_amount_for_timeseries(self):
        """
        calculates how many data rows are needed to produce at least one supervised data column
        29*self.use_indicators because some indicators need 29 prior data rows
        :return: the minimum data needed to produce 1 timestep
        """
        return 29 * self.use_indicators + (self.n_in + self.n_out)  # use_indicators=0 or =1

    def read_h5py_database_file(self):
        """
        Reads the database hdf5 file containing the currency pair data in read only mode
        :returns: The hdf5 file in read only mode
        """
        return h5py.File(self.database_filepath, mode='r', libver='latest', swmr=True)

    def read_h5py_output_file(self):
        return h5py.File(self.filepath, libver='latest')

    def create_h5py_output_file_and_databases(self):
        """
        Creates the .h5 datasets in the finished_data.h5 file
        overwrites file if desired
        """
        directory.ensure_directory(self.filepath)
        file = h5py.File(self.filepath, 'w',
                         libver='latest')  # Always overwrite because database might change
        database = self.read_h5py_database_file()
        for pair in database.keys():
            if pair in file:
                logger.info('Pair: {} already exists in {} ...continuing'.format(pair, str(file)))
            else:
                logger.info('Pair: {} was not found in {} ...creating new'.format(pair, str(file)))
                dset = file.create_dataset(pair, shape=(0, 1),
                                           maxshape=(None, None), dtype='float64')
                dset.flush()
        database.close()
        file.flush()
        file.close()

    def _load_scaler(self, dset_name):
        """
        loads the scaler for the pair/dset_name from /datascaler folder
        :param dset_name:
        :return:
        """
        path = self.scaler_base_filepath + dset_name + '_scaler.save'
        logger.info('Loading {} scaler from: {}'.format(dset_name, path))
        if directory.file_exists(path):
            return joblib.load(path)
        else:
            return None

    def _save_scaler(self, dset_name):
        """
        saves scaler for the dataset in the specific file in /datascaler folder
        :param dset_name:
        :return:
        """
        path = self.scaler_base_filepath + dset_name + '_scaler.save'
        logger.info('Saving {} scaler to: {}'.format(dset_name, path))
        directory.ensure_directory(path)
        joblib.dump(self._scaler[dset_name], path) #this is from sklearn and different(?) to pickle for maximum compatibility

    def get_scaler(self, dset_name):
        """
        return scaler for the pair/dset if there is one
        :param dset_name:
        :return:
        """
        if not self._scaler is None:
            return self._scaler[dset_name]

    def get_scaler_dict(self) -> dict:
        """
        :return: all scalers in a dict with pairs/dset_names as key
        """
        return self._scaler

    def update_completed_and_scaler(self):
        """
        loads all existing scalers from the datascaler folder if available.
        n_completed is not getting updated right now because unmodified data might change
        """
        database = self.read_h5py_database_file()
        #n_completed is already 0 by default TODO: maybe save progress
        for dset_name in database.keys():
            if not self.overwrite_scaler:
                self._scaler[dset_name] = self._load_scaler(dset_name)
            else:
                self._scaler[dset_name] = None
        database.close()


#this is not a class method as this needs to be pickleable by multiprocessing.
# 'self' is an instance of the class and thus bad to use in methods that are called by processes
def produce_modified_data(dset_name, selection_array, scaler, use_indicators, use_scaling, drop_data_columns_indices,
                                                       label_columns_indices, n_in, n_out, n_out_jumps):
    """
    scales the data and adds the indicators to it. transforms the data into a supervised timeseries at last
    :param dset_name: the pair or dset_name (it's the same)
    :param selection_array: the data that is transformed
    :param scaler: if there is a scaler already use this, if None make own
    :param use_indicators: #paramsn defined in config_manager.py
    :param use_scaling:
    :param drop_data_columns_indices:
    :param label_columns_indices:
    :param n_in:
    :param n_out:
    :param n_out_jumps:
    :return: the supervised timeseries data
    """
    if use_indicators:
        selection_array = dm.add_indicators_to_data(selection_array)
    if use_scaling:
        if scaler is None:
            #selection_array, scaler = dm.normalize_data_MinMax(selection_array)
            selection_array, scaler = dm.normalize_data_Standard(selection_array) #use standard scaler for now
        else:
            selection_array = dm.normalize_data(selection_array, scaler)
    timeseries_data  = dm.data_to_supervised_timeseries(selection_array, n_in=n_in, n_out=n_out, n_out_jumps=n_out_jumps, drop_columns_indices=drop_data_columns_indices, #transform to timeseries
                                                       label_columns_indices=label_columns_indices)
    return (dset_name, scaler, timeseries_data)




