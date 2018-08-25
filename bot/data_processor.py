import multiprocessing as mp
from collections import defaultdict
from itertools import repeat

import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import bot.data_modifier as dm
import directory


class DataProcessor():

    def __init__(self, database_filepath, output_filepath, use_indicators=True, use_scaling=True, drop_data_columns_indices: list = [], label_column_indices=[0], n_in=30, n_out=2, overwrite_scaler=False): #TODO: param list of indicators with their paramas!
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
        self.create_h5py_output_file_and_databases()
        self.update_completed_and_scaler()
        #TODO: maybe make currencies selectable instead of just taking all

    def process_and_save(self):
        """
        The main loop of data_processor. Starts all subprocesses for each dataset and saves their modified data in the specific datset in finished_data.h5
        """
        database = self.read_h5py_database_file()
        selection_arrays = []
        database_keys = database.keys()
        n = len(database_keys)
        for dset_name in database_keys:
            dset = database[dset_name]
            selection_array = dset[self.n_completed[dset_name]:, :]
            if len(selection_array) <= self.get_minimum_data_amount_for_timeseries():  # max(timeperiod) in out
                print('not enough data for timeseries calculation', dset_name)
                selection_array = None
            selection_arrays.append(selection_array)
        param_list = list(zip(database_keys, selection_arrays, self._scaler.values(), repeat(self.use_indicators), repeat(self.use_scaling), repeat(self.drop_data_columns_indices), repeat(self.label_column_indices), repeat(self.n_in), repeat(self.n_out)))
        param_list = [x for x in param_list if x[1] is not None]
        with mp.Pool(processes=4) as pool:
            res = pool.starmap_async(func=produce_modified_data, iterable=param_list)
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
                print('saving new scaler')
                self._scaler[dset_name] = scaler
                self._save_scaler(dset_name)
            else:
                print("scaler is old")
            print('completed writing ' + dset_name + ' data to file')
        finished_file.close()

    def get_number_of_features(self):
        """
        :return: The number of columns for a single point in time, including indicator column number if present
        """
        # data = self.read_h5py_database_file() #TODO: indicator calc, unmodified column calc
        # dset = data[data.keys()]
        # 8=number of columns in pair_data_unmodified
        if self.use_indicators:
            return 6 + 8 - len(self.drop_data_columns_indices)  # 6=number of indicator columns
        else:
            return 8 - len(self.drop_data_columns_indices)

    def get_minimum_data_amount_for_timeseries(self):
        """
        30*self.use_indicators because some indicators need 30 data points prior to t
        :return: the minimum data needed to produce at least 1 timestep
        """
        return 30 * self.use_indicators + (self.n_in + self.n_out + 1)  # use_indicators=0 or =1

    def read_h5py_database_file(self):
        """
        Reads the database hdf5 file containing the currency pair data in read only mode
        :returns: The hdf5 file in read only mode
        """
        return h5py.File(self.database_filepath, mode='r', libver='latest', swmr=True)

    def read_h5py_output_file(self):
        return h5py.File(self.filepath, libver='latest')

    def create_h5py_output_file_and_databases(self):
        directory.ensure_directory(self.filepath)
        file = h5py.File(self.filepath, 'w',
                         libver='latest')  # Always overwrite because database might change TODO:change?
        database = self.read_h5py_database_file()
        for pair in database.keys():
            if pair in file:
                print('Pair: ' + pair + 'already exists in' + str(file) + '...continuing')
            else:
                print('Pair: ' + pair + 'was not found in' + str(file) + '...creating new dataset')
                dset = file.create_dataset(pair, shape=(0, 1),
                                           maxshape=(None, None))  # chunk param is really important
                dset.flush()
        database.close()
        file.flush()
        file.close()

    def _load_scaler(self, dset_name):
        path = self.scaler_base_filepath + dset_name + '_scaler.save'
        print(path)
        if directory.file_exists(path):
            return joblib.load(path)
        else:
            return None

    def _save_scaler(self, dset_name):
        path = self.scaler_base_filepath + dset_name + '_scaler.save'
        print(path)
        directory.ensure_directory(path)
        joblib.dump(self._scaler[dset_name], path)

    def get_scaler(self, dset_name) -> MinMaxScaler:
        if not self._scaler is None:
            return self._scaler[dset_name]

    def get_scaler_dict(self):
        return self._scaler

    def update_completed_and_scaler(self):
        database = self.read_h5py_database_file()
        #n_completed is already 0 by default TODO: maybe save progress
        for dset_name in database.keys():
            self._scaler[dset_name] = self._load_scaler(dset_name)


def produce_modified_data(dset_name, selection_array, scaler, use_indicators, use_scaling, drop_data_columns_indices,
                                                       label_columns_indices, n_in, n_out):
    """
    Reads the databse continuesly and makes the data ready for training. All data that is ready is put into queue
    :param dset_name: the specific dataset the method should listen to
    :param queue: the queue in which finished data will be put
    :return: None
    """
    print(dset_name, selection_array.shape)
    if use_indicators:
        selection_array = dm.add_indicators_to_data(selection_array)
    if use_scaling:
        if scaler is None:
            selection_array, scaler = dm.normalize_data_MinMax(selection_array)
        else:
            selection_array = dm.normalize_data(selection_array, scaler)
    timeseries_data  = dm.data_to_supervised_timeseries(selection_array, n_in=n_in, n_out=n_out, drop_columns_indices=drop_data_columns_indices,
                                                       label_columns_indices=label_columns_indices)
    print(timeseries_data.shape)
    return (dset_name, scaler, timeseries_data)




