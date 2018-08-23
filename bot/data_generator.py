import h5py
from multiprocessing import Process

import numpy as np


class DataGenerator:
    
    def __init__(self, finished_data_filepath='data/finished_data.hdf5'):
        self.finished_data_filepath = finished_data_filepath

    def _read_finished_data_file(self):
        return h5py.File(self.finished_data_filepath, 'r', libver='latest', swmr=True)

    def _read_file(self, filename):
        return h5py.File(filename, 'r', libver='latest', swmr=True)


    def create_data_generator_buffered(self, dset_name, batch_size, buffer_size, n_in, n_features):
        """
        Creates a Python generator for finished_data.h5 to use in keras training methods. This version preloads data into memory to reduce the times of loading from hard drive
        :param dset_name: the pair/dataset name to read from
        :param batch_size: the desired batch size
        :param buffer_size: the size of buffer. Higher = faster but more data is loaded into RAM
        :param n_in: timesteps back in past
        :param n_features: feature number of data
        """
        inner_i = 0 #The position in the buffer
        buffer_start = 0 #the index in the dataset where the buffer is starting
        file = self._read_finished_data_file()
        dset = file[dset_name]
        assert batch_size < buffer_size < len(dset)
        buffer = dset[buffer_start:buffer_start+buffer_size].copy()
        while True:
            if inner_i + batch_size > buffer_size:  # outside buffer?
                buffer_start = buffer_start+inner_i
                if buffer_start + buffer_size > len(dset):
                    buffer = np.concatenate([dset[buffer_start:, :], dset[0:buffer_start-len(dset)+buffer_size, :]], axis=0)
                else:
                    buffer = dset[buffer_start: buffer_start+buffer_size].copy()

                inner_i = 0
            selection = buffer[inner_i:inner_i + batch_size]
            inner_i += batch_size
            s_data, s_labels = selection[:, :n_in*n_features], selection[:, n_in*n_features:]
            s_data = s_data.reshape((s_data.shape[0], n_in, n_features))
            yield(s_data, s_labels)


    def create_data_generator(self, dset_name, batch_size, n_in, n_features):
        """
        Creates a Python generator for finished_data.h5 to use in keras training methods. Unbuffered version
        :param dset_name:
        :param batch_size:
        :param n_in:
        :param n_features:
        """
        i = 0
        file = self._read_finished_data_file()
        dset = file[dset_name]
        n = len(dset)
        assert batch_size < n
        while True:
            if i + batch_size > n:
                i = 0
            selection = dset[i: i+batch_size, :]
            i+=batch_size
            s_data, s_labels = selection[:, :n_in * n_features], selection[:, n_in * n_features:]
            s_data = s_data.reshape((s_data.shape[0], n_in, n_features))
            yield (s_data, s_labels)

    def read_data_and_labels_from_finished_data_file(self, dset_name, n_in, n_features):
        """
        loads ALL data from a dataset in finished_data.h5 to memory and returns it
        :param dset_name:
        :param n_in:
        :param n_features:
        :return:
        """
        file = self._read_finished_data_file()
        dset = file[dset_name]
        print(dset.shape)
        s_data, s_labels = dset[:, :n_in * n_features], dset[:, n_in * n_features:]
        s_data = s_data.reshape((s_data.shape[0], n_in, n_features))
        return s_data, s_labels

    def read_data_from_database_file(self, dset_name):
        with self._read_file('data/pair_data_unmodified.h5') as file:
            dset = file[dset_name]
            return dset[:, :]