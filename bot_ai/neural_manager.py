import logging

import h5py
import multiprocessing as mp
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from bot.data_generator import DataGenerator
from bot_ai.neural import Neural

from bot import data_modifier as dm
logger = logging.getLogger('neural_manager')

class NeuralManager(): #this class manages multiple neural networks for each pair
    def __init__(self, unmodified_data_filepath, finished_data_filepath, overwrite_models, batch_size, epochs, output_size, n_in, n_out, n_features, use_scaling, use_indicators, label_index_in_original_data, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='mse', optimizer='adam'):

        self.n_completed = dict #not used yet. idea: dont retrain all data but only new... how though? does it work? is it useful?
        self.unmodified_data_filepath = unmodified_data_filepath #The original data filepath
        self.finished_data_filepath = finished_data_filepath #the timeseries converted data filepath
        self.overwrite_models = overwrite_models #overwrite models at startup
        self.batch_size = batch_size #already described in config_manager.py
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
        #end of already described params
        self.neural_instances = dict() # A python dict with pair name as key vor every neural network
        self.load_models() #load models at init

    def load_models(self):
        """
        Loads all existing models or builds them from start. An instance of Neural is saved in neural_instances for each pair in finished_data.h5 with pairname as key
        """
        finished_data_file = self.read_finished_data_file()
        pairs = list(finished_data_file.keys()) #get all pairs
        finished_data_file.close()
        for pair in pairs:
            self.neural_instances[pair] = Neural(pair, overwrite=self.overwrite_models, batch_size=self.batch_size,
                                                 output_size=self.output_size, n_in=self.n_in, n_out=self.n_out,
                                                 n_features=self.n_features, layer_units=self.layer_units,
                                                 activation_function=self.activation_function,
                                                 loss_function=self.loss_function, optimizer=self.optimizer)
            self.neural_instances[pair].load_or_build_model()

    def train_models(self, plot_history=False):
        """
        Trains all models with the data in finished_data.h5
        the progress is saved in data/training
        """
        finished_data_file = self.read_finished_data_file()
        pairs = list(finished_data_file.keys()) #get all pairs
        finished_data_file.close()
        for pair in pairs:
            neur: Neural = self.neural_instances[pair] # select neural instance
            gen = DataGenerator('data/finished_data.hdf5')
            #generator = gen.create_data_generator(pair, batch_size=self.batch_size, n_in=self.n_in, n_features=self.n_features)
            data, labels = gen.read_data_and_labels_from_finished_data_file(pair, n_in=self.n_in, #read labels and data from finished file
                                                                            n_features=self.n_features)
            split_i = int(len(data) * 0.8) #use 20% as test
            history = neur.train_model(data[0:split_i, :], labels[0:split_i, :], data[split_i:, :], labels[split_i:, :],
                                       epochs=self.epochs, shuffle=True, save=True)  # train model
            logger.info('finished training for {}'.format(pair))
            metrics = neur.model.evaluate(data[split_i:, :], labels[split_i:, :], batch_size=self.batch_size)
            if plot_history: #plot the training history (loss and val_loss)
                plt.plot(history.history['loss'], label='train loss'+pair)
                plt.plot(history.history['val_loss'], label='val loss'+pair)
                plt.plot(metrics, label='test loss'+pair)
        if plot_history:
            plt.legend()
            plt.show()

    def predict_latest_date(self, scalers):
        """
        make predictions for all existing models, with the newest data available in unmodified_data.h5
        This is done by transforming the needed data into a minimal sized timeseries without labels and predicting it.
        The prediction will be of shape (1, n_out+1)
        :param use_scaling: boolean if scaling should be used
        :param scalers: the scalers that have been used in data_processor.py
        :param look_back; offset from date you look back. 1 = look back 1 date (not used right now, could be useful when simulating faster progression.
        :return: the predictions dates as dictionary with pair as key; dictionary with all the predictions and pairs as key
        """
        logger.info('predicting all pairs...')
        predictions = dict()
        dates = dict()
        unmodified_data_file = self.read_unmodified_data_file()
        for pair in unmodified_data_file.keys():
            dset = unmodified_data_file[pair]
            data = dset[:, :] #get all data in pair dset
            dates[pair] = data[-1, 1] #get last date from unmodified
            nolabels = dm.data_to_single_column_timeseries_without_labels(data, self.n_in, scalers[pair], [], self.use_scaling, self.use_indicators) #make a single column timeseries
            nolabels = nolabels.reshape((nolabels.shape[0], self.n_in, self.n_features)) #Shape data in the right form for the neural network
            neur : Neural = self.neural_instances[pair]
            predictions[pair] = neur.predict(nolabels) #predict the single row timeseries
            if self.use_scaling: #reverse scaleing if data was scaled
                scaler : MinMaxScaler = scalers[pair]
                predictions[pair] = dm.reverse_normalize_prediction(predictions[pair], self.label_index_in_original_data, self.n_features, scaler)
        return dates, predictions #return prediction-date and prediction dicts


    def predict_all_data(self, scalers):
        """
        Predict all the data in the unmodified file.
        No real use right now, more for showing purposes
        :param scalers: the used scalers
        :return: the predictions with shape (n, n_out + 1) or (n, output_size)
        """
        predictions = dict()
        dates = dict()
        original = dict()
        unmodified_data_file = self.read_unmodified_data_file()
        for pair in unmodified_data_file.keys():
            dset = unmodified_data_file[pair]
            data = dset[:, :]
            nolabels = dm.data_to_timeseries_without_labels(data, self.n_in, scalers[pair], [], self.use_scaling, self.use_indicators)
            print(nolabels.shape)
            nolabels = nolabels.reshape((nolabels.shape[0], self.n_in, self.n_features))
            neur: Neural = self.neural_instances[pair]
            predictions[pair] = neur.predict(nolabels)
            length = len(predictions[pair])
            dates[pair] = data[-length:, 1]
            original[pair] = data[-length:, self.label_index_in_original_data]
            if self.use_scaling:
                scaler: MinMaxScaler = scalers[pair]
                predictions[pair] = dm.reverse_normalize_prediction(predictions[pair], self.label_index_in_original_data, self.n_features, scaler)
        return dates, predictions, original


    def read_finished_data_file(self):
        """
        get Handle for file
        :return: h5py file
        """
        return h5py.File(self.finished_data_filepath, libver='latest')

    def read_unmodified_data_file(self):
        return h5py.File(self.unmodified_data_filepath, libver='latest')