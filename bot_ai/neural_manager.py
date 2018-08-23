import h5py
import multiprocessing as mp

from bot.data_generator import DataGenerator
from bot_ai.neural import Neural


class NeuralManager():
    def __init__(self, finished_data_filepath, overwrite_models, batch_size, output_size, n_in, n_out, n_features, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='LeakyReLU', optimizer='adam'):

        self.finished_data_filepath = finished_data_filepath
        self.overwrite_models = overwrite_models
        self.batch_size = batch_size
        self.output_size = output_size
        self.n_in = n_in
        self.n_out = n_out
        self.n_features = n_features
        self.layer_units = layer_units
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        #param end
        self.neural_instances = dict()

    def run(self):
        finished_data_file = self.read_finished_data_file()
        pairs = finished_data_file.keys()
        for pair in pairs:
            self.neural_instances[pair] = Neural(pair, overwrite=self.overwrite_models, batch_size=self.batch_size, output_size=self.output_size, n_in=self.n_in, n_out=self.n_out, n_features=self.n_features, layer_units=self.layer_units, activation_function=self.activation_function, loss_function=self.loss_function, optimizer=self.optimizer)
        #pickling horror incoming?
        with mp.Pool(processes=4) as pool:
            res = pool.starmap_async(func=data_downloader, iterable=param_list)
            pool.close()
            pool.join()
        results = res.get(timeout=None)



    def read_finished_data_file(self):
        return h5py.File(self.finished_data_filepath, libver='latest')