import logging
import os
import time
from multiprocessing import Queue, Process

import h5py
import numpy as np

import util.data_modifier as dm


class DataProcessor:

    def __init__(self, database_filepath, output_filepath, use_indicators=True, use_scaling=True, overwrite=False):
        self.use_scaling = use_scaling
        self.use_indicators = use_indicators
        self.output_filepath = output_filepath
        self.overwrite = overwrite
        self.database_filepath = database_filepath
        self.create_h5py_output_file()
        self.create_databases()

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
            print(dset_name + ' received. putting in file')
            file = self.read_h5py_output_file()
            print('hi0')
            dset = file[dset_name]
            print('hi1')
            dset.resize((dset.shape[0] + data.shape[0]), axis=0)
            print('hi2')
            dset.resize(max(dset.shape[1], data.shape[1]), axis=1)
            print('hi3')
            dset[-data.shape[0]:] = data
            print('hi4')
            file.close()
            print('hi5')
            print('Saved modified data to file for pair[' + dset_name + ']')

    def read_database_and_produce_modified_data_loop(self, queue, dset_name):
        """
        Reads the databse continuesly and makes the data ready for training. All data that is ready is put into queue
        :param dset_name: the specific dataset the method should listen to
        :param queue: the queue in which finished data will be put
        :return: None
        """
        database = self.read_h5py_database_file()
        dset = database[dset_name]
        print("new Process started listening on dataset[" + dset_name + ']')
        n_completed = 0  # the number of finished modified rows, may self
        while True:
            selection_array = dset[n_completed:, :]  # important datas
            if len(selection_array) <= 30 + 30 + 2:  # max(timeperiod) in out
                print('no new data found for', dset_name, '. looking again in 20 second')
                del selection_array
                time.sleep(20)
                continue
            if self.use_indicators:  # adding indicators
                selection_array = np.array(selection_array, dtype='f8')
                selection_array = dm.add_SMA_indicator_to_data(selection_array, close_index=0, timeperiod=30)
                selection_array = dm.add_BBANDS_indicator_to_data(selection_array, close_index=0)
                selection_array = dm.add_RSI_indicator_to_data(selection_array, close_index=0)
                selection_array = dm.add_OBV_indicator_to_data(selection_array, close_index=0, volume_index=6)
                selection_array = dm.drop_NaN_rows(selection_array)
            if self.use_scaling:
                selection_array, min_max_scaler = dm.normalize_data_MinMax(selection_array)
            timeseries_data = dm.data_to_supervised(selection_array, n_in=30, n_out=2, drop_columns_indices=[7],
                                                    label_columns_indices=[0])
            n_completed += len(timeseries_data)
            print("Data modification finished")
            queue.put((dset_name, timeseries_data))
            del selection_array
            del timeseries_data

    def load_from_config(self):
        pass

    def create_h5py_output_file(self):
        if not os.path.exists(os.path.dirname(self.output_filepath)):
            try:
                os.makedirs(os.path.dirname(self.output_filepath))
            except OSError as exc:  # Guard against race condition
                logging.exception(exc)
        if self.overwrite:
            print("Overwriting output file")
            return h5py.File(self.output_filepath, 'w', libver='latest')
        else:
            print("Opening or creating ouput file")
            return h5py.File(self.output_filepath, 'a', libver='latest')

    def create_databases(self):
        file = self.read_h5py_output_file()
        database = self.read_h5py_database_file()
        for pair in database.keys():
            if pair in file:
                logging.info('Pair: ' + pair + 'already exists in' + str(file) + '...continuing')
            else:
                logging.info('Pair: ' + pair + 'was not found in' + str(file) + '...creating new dataset')
                dset = file.create_dataset(pair, (1, 8), maxshape=(None, None))
                dset.flush()
        file.swmr_mode = True
