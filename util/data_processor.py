import logging
import os
import time
from collections import defaultdict
from multiprocessing import Queue, Process

import h5py
import numpy as np

import util.data_modifier as dm


class DataProcessor():

    def __init__(self, callback, database_filepath, output_filepath, use_indicators=True, use_scaling=True, drop_data_columns_indices=[7], n_in=30, n_out=2): #TODO: param list of indicators with their paramas!
        self.n_in = n_in
        self.n_out = n_out
        self.use_scaling = use_scaling
        self.drop_data_columns_indices = drop_data_columns_indices
        self.use_indicators = use_indicators
        self.output_filepath = output_filepath
        self.database_filepath = database_filepath
        self.callback = callback
        self.create_h5py_output_file()
        self.min_max_scaler = {}
        self.n_completed = defaultdict(int)
        self.create_databases()

    def get_number_of_features(self):
        #data = self.read_h5py_database_file() #TODO: alles
        #dset = data[data.keys()]
        #
        if self.use_indicators:
            return 6 + 8 - len(self.drop_data_columns_indices) # +dset.shape[1]
        else:
            return 8 - len(self.drop_data_columns_indices) # +dset.shape[1]

    def read_h5py_database_file(self):
        """
        Reads the database hdf5 file containing the currency pair data in read only mode
        :returns: The hdf5 file in read only mode
        """
        return h5py.File(self.database_filepath, 'r', libver='latest')

    def read_h5py_output_file(self):
        return h5py.File(self.output_filepath, 'a', libver='latest')

    def read_database_and_produce_modified_data(self, dset_name):
        """
        Reads the databse continuesly and makes the data ready for training. All data that is ready is put into queue
        :param dset_name: the specific dataset the method should listen to
        :param queue: the queue in which finished data will be put
        :return: None
        """
        print('Processing new data')
        database = self.read_h5py_database_file()
        dset = database[dset_name]
        selection_array = dset[self.n_completed[dset_name]:, :]  # important datas
        if len(selection_array) <= 30 + 30 + 2:  # max(timeperiod) in out
            print('not enough data for timeseries calculation', dset_name, '. Waiting for next call')
            del selection_array
            database.close()
            return False
        if self.use_indicators:  # adding indicators
            selection_array = np.array(selection_array, dtype='f8')
            selection_array = dm.add_SMA_indicator_to_data(selection_array, close_index=0, timeperiod=30) #1 column
            selection_array = dm.add_BBANDS_indicator_to_data(selection_array, close_index=0) #3 columns
            selection_array = dm.add_RSI_indicator_to_data(selection_array, close_index=0) #1 column
            selection_array = dm.add_OBV_indicator_to_data(selection_array, close_index=0, volume_index=6) #1column
            selection_array = dm.drop_NaN_rows(selection_array)
        if self.use_scaling:
            if not dset_name in self.min_max_scaler:
                selection_array, self.min_max_scaler[dset_name] = dm.normalize_data_MinMax(selection_array)
            else:
                selection_array = dm.normalize_data(selection_array, self.min_max_scaler[dset_name])
        timeseries_data = dm.data_to_supervised(selection_array, n_in=self.n_in, n_out=self.n_out, drop_columns_indices=self.drop_data_columns_indices,
                                                label_columns_indices=[0])
        self.n_completed[dset_name] += len(timeseries_data)
        print("Data modification finished for pair[" + dset_name + '].')
        file = self.read_h5py_output_file()
        dset = file[dset_name]
        dset.resize(max(dset.shape[1], timeseries_data.shape[1]), axis=1)
        dset.resize((dset.shape[0] + timeseries_data.shape[0]), axis=0)
        dset[-timeseries_data.shape[0]:] = timeseries_data
        file.flush()
        file.close()
        print('Saved modified data with shape', timeseries_data.shape, 'to file for pair[' + dset_name + ']')
        del selection_array
        del timeseries_data
        database.close()
        self.callback()
        return True

    def create_h5py_output_file(self):
        if not os.path.exists(os.path.dirname(self.output_filepath)):
            try:
                os.makedirs(os.path.dirname(self.output_filepath))
            except OSError as exc:  # Guard against race condition
                logging.exception(exc)
        print("Overwriting output file")
        return h5py.File(self.output_filepath, 'w', libver='latest') #Always overwrite because databse might change

    def create_databases(self):
        file = self.read_h5py_output_file()
        database = self.read_h5py_database_file()
        for pair in database.keys():
            if pair in file:
                logging.info('Pair: ' + pair + 'already exists in' + str(file) + '...continuing')
            else:
                logging.info('Pair: ' + pair + 'was not found in' + str(file) + '...creating new dataset')
                dset = file.create_dataset(pair, shape=(0, 1), maxshape=(None, None)) #chunk param is really important
                dset.flush()
        file.swmr_mode = True

    def callback_func(self, updated_dset_name):
        print('Data processor for [' +updated_dset_name+'] was called')
        p = Process(target=self.read_database_and_produce_modified_data, args=(updated_dset_name, ))
        p.start()
