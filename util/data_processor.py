import logging
import os
import time
from multiprocessing import Queue, Process

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

import util.data_modifier as dm


class DataProcessor(Process):

    def __init__(self, queue, database_filepath, output_filepath, use_indicators=True, use_scaling=True, drop_data_columns_indices=[], n_in=30, n_out=2): #TODO: param list of indicators with their paramas!
        super(DataProcessor, self).__init__()
        self.queue = queue
        self.n_in = n_in
        self.n_out = n_out
        self.use_scaling = use_scaling
        self.drop_data_columns_indices = drop_data_columns_indices
        self.use_indicators = use_indicators
        self.output_filepath = output_filepath
        self.database_filepath = database_filepath
        self._scaler = None
        self.create_h5py_output_file()
        self.create_databases()

    def get_number_of_features(self):
        #data = self.read_h5py_database_file() #TODO: indicator calc
        #dset = data[data.keys()]
        #
        if self.use_indicators:
            return 6 + 8 - len(self.drop_data_columns_indices) # +dset.shape[1]
        else:
            return 8 - len(self.drop_data_columns_indices) # +dset.shape[1]

    def get_data_column_width(self):
        return self.n_in + 30*self.use_indicators + self.n_out + 1

    def read_h5py_database_file(self):
        """
        Reads the database hdf5 file containing the currency pair data in read only mode
        :returns: The hdf5 file in read only mode
        """
        return h5py.File(self.database_filepath, 'r', libver='latest')

    def read_h5py_output_file(self):
        return h5py.File(self.output_filepath, 'a', libver='latest')

    def run(self):
        print("Data Processor has started")
        database_file = self.read_h5py_database_file()
        q = Queue()
        processes = []
        for dset_name in database_file.keys():
            p = Process(target=self.read_database_and_produce_modified_data_loop, args=(q, dset_name,))
            processes.append(p)
            p.start()
        while True:
            dset_name, data = q.get()
            file = self.read_h5py_output_file()
            dset = file[dset_name]
            dset.resize(max(dset.shape[1], data.shape[1]), axis=1)
            dset.resize((dset.shape[0] + data.shape[0]), axis=0)
            dset[-data.shape[0]:] = data
            file.flush()
            file.close()
            print('Saved modified data with shape', data.shape, 'to file for pair[' + dset_name + ']')
            del data

    def read_database_and_produce_modified_data_loop(self, queue, dset_name):
        """
        Reads the databse continuesly and makes the data ready for training. All data that is ready is put into queue
        :param dset_name: the specific dataset the method should listen to
        :param queue: the queue in which finished data will be put
        :return: None
        """
        print("new Process started listening on dataset[" + dset_name + ']')
        n_completed = 0  # the number of finished modified rows, may self
        while True:
            # event[dset_name].wait()
            database = self.read_h5py_database_file()
            dset = database[dset_name]
            selection_array = dset[n_completed:, :]  # important datas
            if len(selection_array) <= self.get_data_column_width():  # max(timeperiod) in out
                print('not enough data for timeseries calculation', dset_name, '. Waiting for next event')
                del selection_array
                database.close()
                time.sleep(20)
                continue
            if self.use_indicators:  # adding indicators
                selection_array = np.array(selection_array, dtype='f8')
                selection_array = dm.add_SMA_indicator_to_data(selection_array, close_index=0, timeperiod=30) #1 column
                selection_array = dm.add_BBANDS_indicator_to_data(selection_array, close_index=0) #3 columns
                selection_array = dm.add_RSI_indicator_to_data(selection_array, close_index=0) #1 column
                selection_array = dm.add_OBV_indicator_to_data(selection_array, close_index=0, volume_index=6) #1column
                selection_array = dm.drop_NaN_rows(selection_array)
            if self.use_scaling:
                if self._scaler is None:
                    selection_array, self._scaler = dm.normalize_data_MinMax(selection_array) #TODO: decide over param
                    self.queue.put(self._scaler)
                else:
                    selection_array = dm.normalize_data(selection_array, self._scaler)
            timeseries_data = dm.data_to_supervised(selection_array, n_in=self.n_in, n_out=self.n_out, drop_columns_indices=self.drop_data_columns_indices,
                                                    label_columns_indices=[0])
            n_completed += len(timeseries_data)
            print("Data modification finished for pair[" + dset_name + '] with shape', timeseries_data.shape)
            queue.put((dset_name, timeseries_data))
            del selection_array
            del timeseries_data
            database.close()

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
