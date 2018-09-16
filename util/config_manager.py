import logging

import directory
import pickle

from bot import data_modifier


class BotConfigManager():

    def __init__(self):
        self.filepath = directory.get_absolute_path('config.pickle')
        self.load_config()

    def load_collector_settings(self): #TODO: placeholder implementation
        #relative_filepath, currency_pairs, start_dates=[1405699200], end_dates, time_periods, drop_data_column_indices, overwrite, offline
        return self.unmodified_data_filepath, self.pairs, self.start_dates, self.end_dates, self.timesteps, self.drop_data_column_indices, self.redownload_data, self.offline

    def load_processor_settings(self): #TODO: placeholder implementation
        #database_filepath, output_filepath, label_transform_function, use_indicators=True, use_scaling=True, label_column_indices=[0], n_in=30, n_out=2, n_out_jumps=1, overwrite_scaler=False)
        return self.unmodified_data_filepath, self.finished_data_filepath, self.label_transform_function, self.use_indicators, self.use_scaling, self.data_label_column_index, self.n_in, self.n_out, self.n_out_jumps, self.overwrite_scalers

    def load_neural_manager_settings(self): #TODO: placeholder implementation
        #return finished_data_filepath, overwrite_models, batch_size, epochs, output_size, n_in, n_out, n_features, label_transform_function, use_scaling, use_indicators, label_index_in_original_data, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='LeakyReLU', optimizer='adam'
        return self.unmodified_data_filepath, self.finished_data_filepath, self.overwrite_models, self.batch_size, self.epochs, self.n_out + 1, self.n_in, self.n_out, self.n_features, self.label_transform_function, self.use_scaling, self.use_indicators, self.data_label_column_index, self.layer_sizes_list, self.activation_function, self.loss_function, self.optimizer


    def load_latest_training_run(self): #TODO: placeholder implementation
        return self.latest_training_run


    def load_prediction_history_settings(self):
        #filepath, timesteps, date_column, close_column, n_out_jumps, overwrite_history
        return self.prediction_history_filepath, self.timesteps, self.data_date_column_index, self.data_label_column_index, self.n_out_jumps, self.overwrite_history

    def load_training_prediction_history_settings(self):
        # filepath, timesteps, date_column, close_column, n_out_jumps, overwrite_history
        return 'data/training_prediction_history.pickle', self.timesteps, self.data_date_column_index, self.data_label_column_index, self.n_out_jumps, self.overwrite_history

    def set_offline_mode(self, bool):
        self.offline = bool

    def init_variables(self):
        self.unmodified_data_filepath = 'data/unmodified_data.h5'
        self.finished_data_filepath = 'data/finished_data.hdf5'
        self.prediction_history_filepath = 'data/prediction_history.pickle'
        self.pairs = ['USDT_BTC']
        self.start_dates = [1483225200] #dates in unix timestamp format
        self.end_dates = [9999999999]
        self.timesteps = 7200 # interval for new close data. this is important for poloniex api. 300secs=5minutes, valid values: 300, 900, 1800, 7200, 14400, and 86400
        self.drop_data_column_indices = [] # drop useless data columns. Please dont use negative number as this need to be DataFrame compatible
        self.data_date_column_index = 1 # specifies where the data column is
        self.data_label_column_index = 0 #where are the labels? only use one right now, more is experimental and not tested
        self.n_in = 14 # number of input data before the current data
        self.n_out = 2 # number of additional predicted labels
        self.n_out_jumps = 1 # every n_out_jumps data point is beeing predicted. e.g 2 = every second future datapoint is beeing predicted
        self.redownload_data = True #wether all data should be redownloaded. If you alter start, end or timesteps this has to be set to true, also should be true in offline mode
        self.label_transform_function = data_modifier.difference_label_transformation_function #Any function that is build like that one worls
        self.use_scaling = True #Use scaling in data_processor and anywhere else?
        self.overwrite_scalers = True #Use old scalers or overwrite at startup? this should be False normally and True if overwriting models.
        self.use_indicators = False #should indicators be added? my tests have shown that the neural net gets confused by these
        self.overwrite_models = True #overwrite models at startup. default=False
        self.overwrite_history = True #overwrites prediction history at startup. This has no big impact on bot functions
        self.batch_size = 64 #The number of data rows in one batch
        self.epochs = 100 #The maximum number of epochs (early stopping might trigger)
        self.n_features = 8 + (7 * self.use_indicators) + 1 * (not self.label_transform_function is None) - len(self.drop_data_column_indices) # There are 8 columns normally, add 7 more if indicators are used and substract all dropped columns
        self.layer_sizes_list = [100] #The number of neurons in each layer. The List should be at minimum 1 and can be as large as wanted
        self.activation_function = 'LeakyReLU' #The used activation function in all besides the last layer (last layer is softmax default)
        self.loss_function = 'mse' #The loss function to determine loss. there might be better stuff than mse
        self.optimizer = 'adam' #All praise adam our favourite optimizer
        self.latest_training_run = 0 #this is a timestamp when the latest training run was executed high number=never retrain
        self.train_every_n_seconds = 24 * 60 * 60 # retrain every n-seconds
        self.offline = False #This is the default param for offline mode. gets overwritten by --offline command (True)
        self.lag = 400 #This is the amount of data hold back in offline mode. This is essentially the number of actions that are simulated
        self.display_plot_interval = 50 #The loop interval at  which plots are plotted. 50=plots are drawn every 50 tradingbot.run() completions. when not using offline mode this should be equals 1

    def setup(self):
        """
        Code that should be run before anything starts.
        :return:
        """
        from bot import API_offline
        if self.offline:
            API_offline.load_data_and_download_if_not_existent(self.pairs, self.timesteps) #download data if offline mode is true
            API_offline.init_global_lag(self.lag)


    def save_config(self):
        """
        Save the config with all stored variables as a pickle
        :return:
        """
        logging.info("saving config")
        directory.ensure_directory(self.filepath)
        f = open(self.filepath, 'wb')
        pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_config(self):
        """
        Load the config with stored vars from file. (if existent)
        if not initialize all variables
        :return: if the load was successful
        """
        logging.info("loading config")
        if directory.file_exists(self.filepath):
            f = open(self.filepath, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict)
            return True
        else:
            self.init_variables()
            return False







