import h5py
import multiprocessing as mp

from sklearn.preprocessing import MinMaxScaler

from bot.data_generator import DataGenerator
from bot_ai.neural import Neural

from bot import data_modifier as dm


class NeuralManager():
    def __init__(self, unmodified_data_filepath, finished_data_filepath, overwrite_models, batch_size, output_size, n_in, n_out, n_features, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='LeakyReLU', optimizer='adam'):

        self.n_completed = dict
        self.unmodified_data_filepath = unmodified_data_filepath
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
        self.load_models()

    def load_models(self):
        """
        Loads all existing models or builds them from start. An instance of Neural is saved in neural_instances for each pair in finished_data.h5
        """
        finished_data_file = self.read_finished_data_file()
        pairs = list(finished_data_file.keys())
        finished_data_file.close()
        for pair in pairs:
            self.neural_instances[pair] = Neural(pair, overwrite=self.overwrite_models, batch_size=self.batch_size,
                                                 output_size=self.output_size, n_in=self.n_in, n_out=self.n_out,
                                                 n_features=self.n_features, layer_units=self.layer_units,
                                                 activation_function=self.activation_function,
                                                 loss_function=self.loss_function, optimizer=self.optimizer)
            self.neural_instances[pair].load_or_build_model()

    def train_models(self): #TODO: make this use multiprocessing
        """
        Trains all models with the data in finished_data.h5
        the progress is saved in data/training
        """
        finished_data_file = self.read_finished_data_file()
        pairs = list(finished_data_file.keys())
        finished_data_file.close()
        for pair in pairs:
            neur: Neural = self.neural_instances[pair]
            gen = DataGenerator('data/finished_data.hdf5')
            #generator = gen.create_data_generator(pair, batch_size=self.batch_size, n_in=self.n_in, n_features=self.n_features)
            data, labels = gen.read_data_and_labels_from_finished_data_file(pair, n_in=self.n_in,
                                                                            n_features=self.n_features, shuffled=True) #, n_completed=self.n_completed[pair]) #shuffled and n_completed is still to be tested
            split_i = int(len(data) * 0.9)
            history = neur.train_model(data[0:split_i, :], labels[0:split_i, :], data[split_i:, :], labels[split_i:, :],
                                       epochs=100, shuffle=True, save=True)  # 'normal' method
            print('finished training for', pair)

    def make_latest_predictions(self, scalers, use_scaling=True):
        """
        make predictions for all existing models, with the newest data available in unmodified_data.h5
        :param use_scaling: boolean if scaling should be used
        :param scalers: the scalers that have been used in data_processor
        :return: dictionary with all the predictions and pairs as key
        """
        predictions = dict()
        unmodified_data_file = self.read_unmodified_data_file()
        for pair in unmodified_data_file.keys(): #TODO: Change this to finished data, if processor changes behavior
            dset = unmodified_data_file[pair]
            data = dset[:, :]
            nolabels = dm.data_to_timeseries_without_labels(data, self.n_in, scalers[pair], [], True, True)
            nolabels = nolabels.reshape((nolabels.shape[0], self.n_in, self.n_features))
            neur : Neural = self.neural_instances[pair]
            predictions[pair] = neur.predict(nolabels) #TODO: only okay as long as 'pairs processor = pairs collector'
            if use_scaling:
                scaler : MinMaxScaler = scalers[pair]
                predictions[pair] = dm.reverse_normalize_prediction(predictions[pair], 0, self.n_features, scaler)
            print('predictions for', pair, predictions[pair])
        return predictions



    def read_finished_data_file(self):
        return h5py.File(self.finished_data_filepath, libver='latest')

    def read_unmodified_data_file(self):
        return h5py.File(self.unmodified_data_filepath, libver='latest')