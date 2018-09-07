import h5py
import multiprocessing as mp
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from bot.data_generator import DataGenerator
from bot_ai.neural import Neural

from bot import data_modifier as dm


class NeuralManager():
    def __init__(self, unmodified_data_filepath, finished_data_filepath, overwrite_models, batch_size, epochs, output_size, n_in, n_out, n_features, use_scaling, use_indicators, label_index_in_original_data, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='mse', optimizer='adam'):

        self.n_completed = dict
        self.unmodified_data_filepath = unmodified_data_filepath
        self.finished_data_filepath = finished_data_filepath
        self.overwrite_models = overwrite_models
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_size = output_size
        self.n_in = n_in
        self.n_out = n_out
        self.n_features = n_features
        self.use_scaling = use_scaling
        self.use_indicators = use_indicators
        self.label_index_in_original_data = label_index_in_original_data
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

    def train_models(self, plot_history=False): #TODO: make this use multiprocessing
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
                                                                            n_features=self.n_features)
            split_i = int(len(data) * 0.9)
            history = neur.train_model(data[0:split_i, :], labels[0:split_i, :], data[split_i:, :], labels[split_i:, :],
                                       epochs=self.epochs, shuffle=True, save=True)  # 'normal' method
            print('finished training for', pair)
            if plot_history:
                plt.plot(history.history['loss'], label='train'+pair)
                plt.plot(history.history['val_loss'], label='test'+pair)
        if plot_history:
            plt.legend()
            plt.show()

    def predict_latest_date(self, scalers, look_back=0):
        """
        make predictions for all existing models, with the newest data available in unmodified_data.h5
        :param use_scaling: boolean if scaling should be used
        :param scalers: the scalers that have been used in data_processor
        :param look_back; offset from date you look back. 1 = look back 1 date
        :return: the predictions dates as dictionary; dictionary with all the predictions and pairs as key
        """
        predictions = dict()
        dates = dict()
        unmodified_data_file = self.read_unmodified_data_file()
        for pair in unmodified_data_file.keys(): #TODO: Change this to finished data, if processor changes behavior
            dset = unmodified_data_file[pair]
            data = dset[:, :]
            dates[pair] = data[-1, 1]
            nolabels = dm.data_to_single_column_timeseries_without_labels(data, self.n_in, scalers[pair], [], self.use_scaling, self.use_indicators)
            nolabels = nolabels.reshape((nolabels.shape[0], self.n_in, self.n_features))
            neur : Neural = self.neural_instances[pair]
            predictions[pair] = neur.predict(nolabels) #TODO: only okay as long as 'pairs processor = pairs collector'
            if self.use_scaling:
                scaler : MinMaxScaler = scalers[pair]
                predictions[pair] = dm.reverse_normalize_prediction(predictions[pair], self.label_index_in_original_data, self.n_features, scaler)
            print('predictions for', dates[pair], pair, predictions[pair])
        return dates, predictions

    def predict_all_data(self, scalers):
        predictions = dict()
        dates = dict()
        unmodified_data_file = self.read_unmodified_data_file()
        for pair in unmodified_data_file.keys():  # TODO: Change this to finished data, if processor changes behavior
            dset = unmodified_data_file[pair]
            data = dset[:, :]
            dates[pair] = data[:, 1]
            nolabels = dm.data_to_timeseries_without_labels(data, self.n_in, scalers[pair], [], self.use_scaling, self.use_indicators)
            nolabels = nolabels.reshape((nolabels.shape[0], self.n_in, self.n_features))
            neur: Neural = self.neural_instances[pair]
            predictions[pair] = neur.predict(nolabels)  # TODO: only okay as long as 'pairs processor = pairs collector'
            if self.use_scaling:
                scaler: MinMaxScaler = scalers[pair]
                predictions[pair] = dm.reverse_normalize_prediction(predictions[pair], self.label_index_in_original_data, self.n_features, scaler)
            print('predictions for', dates[pair], pair, predictions[pair])
        return dates, predictions



    def read_finished_data_file(self):
        return h5py.File(self.finished_data_filepath, libver='latest')

    def read_unmodified_data_file(self):
        return h5py.File(self.unmodified_data_filepath, libver='latest')