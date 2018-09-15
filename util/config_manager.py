import logging

import directory
import pickle



class BotConfigManager():

    def __init__(self):
        self.filepath = directory.get_absolute_path('config.pickle')
        self.load_config()

    def load_collector_settings(self): #TODO: placeholder implementation
        #relative_filepath, currency_pairs, start_dates=[1405699200], end_dates, time_periods, overwrite, offline
        return self.unmodified_data_filepath, self.pairs, self.start_dates, self.end_dates, self.timesteps, self.redownload_data, self.offline

    def load_processor_settings(self): #TODO: placeholder implementation
        #database_filepath, output_filepath, use_indicators=True, use_scaling=True, drop_data_columns_indices: list = [], label_column_indices=[0], n_in=30, n_out=2, n_out_jumps=1, overwrite_scaler=False)
        return self.unmodified_data_filepath, self.finished_data_filepath, self.use_indicators, self.use_scaling, self.drop_data_column_indices, self.data_label_column_indices, self.n_in, self.n_out, self.n_out_jumps, self.overwrite_scalers

    def load_neural_manager_settings(self): #TODO: placeholder implementation
        #return finished_data_filepath, overwrite_models, batch_size, epochs, output_size, n_in, n_out, n_features, use_scaling, use_indicators, label_index_in_original_data, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='LeakyReLU', optimizer='adam'
        return self.unmodified_data_filepath, self.finished_data_filepath, self.overwrite_models, self.batch_size, self.epochs, self.n_out + 1, self.n_in, self.n_out, self.n_features, self.use_scaling, self.use_indicators, self.data_label_column_indices[0], self.layer_sizes_list, self.activation_function, self.loss_function, self.optimizer


    def load_latest_training_run(self): #TODO: placeholder implementation
        return self.latest_training_run


    def load_prediction_history_settings(self):
        #filepath, timesteps, date_column, close_column, n_out_jumps, overwrite_history
        return self.prediction_history_filepath, self.timesteps, self.data_date_column_indice, self.data_label_column_indices[0], self.n_out_jumps, self.overwrite_history

    def load_training_prediction_history_settings(self):
        # filepath, timesteps, date_column, close_column, n_out_jumps, overwrite_history
        return 'data/training_prediction_history.pickle', self.timesteps, self.data_date_column_indice, self.data_label_column_indices[0], self.n_out_jumps, True

    def set_offline_mode(self, bool):
        self.offline = bool

    def init_variables(self):
        self.unmodified_data_filepath = 'data/unmodified_data.h5'
        self.finished_data_filepath = 'data/finished_data.hdf5'
        self.prediction_history_filepath = 'data/prediction_history.pickle'
        self.pairs = ['USDT_BTC', 'USDT_ETH']
        self.start_dates = [1483225200, 1483225200]
        self.end_dates = [9999999999, 9999999999]
        self.timesteps = 300 # interval for new close data. this is important for poloniex api. 300secs=5minutes
        self.drop_data_column_indices = [] # drop useless data columns. All data might be useful so this is empty
        self.data_date_column_indice = 1 # specifies where the data column is
        self.data_label_column_indices = [0] #where are the labels? only use one right now, more is experimental and not tested
        self.n_in = 10 # number of input data before the current data
        self.n_out = 2 # number of additional predicted labels
        self.n_out_jumps = 1 # every n_out_jumps data point is beeing predicted. e.g 2 = every second future datapoint is beeing predicted
        self.redownload_data = True #wether all data should be redownloaded. If you alter start, end or timesteps this has to be set to true
        self.use_scaling = True #Use scaling in data_processor and anywhere else?
        self.overwrite_scalers = False #Use old scalers or overwrite at startup? this should be False normally and True if overwriting models.
        self.use_indicators = True #should indicators be added?
        self.overwrite_models = False #overwrite models at startup. default=False
        self.overwrite_history = True #overwrites prediction history at startup. This has no big impact on bot functions
        self.batch_size = 64 #The number of datarows in one batch
        self.epochs = 50 #The maximum number of epochs (early stopping might trigger)
        self.n_features = 6 + 9 * self.use_indicators - len(self.drop_data_column_indices) # There are 6 columns normally, add 9 if indicators are used and substract all dropped columns
        self.layer_sizes_list = [100] #The number of neurons in each layer. The List should be at minimum 1 and can be as large as wanted
        self.activation_function = 'LeakyReLU' #The used activation function in all besides the last layer (last layer is softmax default)
        self.loss_function = 'mse' #The loss function to determine loss. there might be better stuff than mse
        self.optimizer = 'adam' #All praise adam our favourite optimizer
        self.latest_training_run = 99999999999 #this is a timestamp when the latest training run was executed high number=never retrain
        self.train_every_n_seconds = 24 * 60 * 60 # retrain every n-seconds
        self.offline = False #This is the default param for offline mode. gets overwritten by --offline command (True)
        self.lag = 200 #This is the amount of data hold back in offline mode. This is essentially the number of actions that are simulated
        self.display_plot_interval = 50 #The loop interval at  which plots are plotted. 50=plots are drawn every 50 tradingbot.run() completions. when not using offline mode this should be equals 1

    def setup(self):
        from bot import API_offline
        if self.offline:
            API_offline.load_data_and_download_if_not_existent(self.pairs, self.timesteps) #download data if offline mode is true
            API_offline.init_global_lag(self.lag)

    def save_config(self):
        logging.info("saving config")
        directory.ensure_directory(self.filepath)
        f = open(self.filepath, 'wb')
        pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_config(self):
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







