import directory
import pickle

class BotConfigManager():

    def __init__(self):
        self.filepath = directory.get_absolute_path('config.pickle')
        self.load_config()

    def load_collector_settings(self): #TODO: placeholder implementation
        #relative_filepath, currency_pairs, start_dates=[1405699200], end_dates, time_periods, overwrite
        return self.unmodified_data_filepath, self.pairs, self.start_dates, self.end_dates, self.timesteps, self.redownload_data

    def load_processor_settings(self): #TODO: placeholder implementation
        #database_filepath, output_filepath, use_indicators=True, use_scaling=True, drop_data_columns_indices: list = [], label_column_indices=[0], n_in=30, n_out=2, n_out_jumps=1, overwrite_scaler=False)
        return self.unmodified_data_filepath, self.finished_data_filepath, self.use_indicators, self.use_scaling, self.drop_data_column_indices, self.data_label_column_indices, self.n_in, self.n_out, self.n_out_jumps, self.overwrite_scalers

    def load_neural_manager_settings(self): #TODO: placeholder implementation
        #return finished_data_filepath, overwrite_models, batch_size, epochs, output_size, n_in, n_out, n_features, use_scaling, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='LeakyReLU', optimizer='adam'
        return self.unmodified_data_filepath, self.finished_data_filepath, self.overwrite_models, self.batch_size, self.epochs, self.n_out + 1, self.n_in, self.n_out, self.n_features, self.use_scaling, self.layer_sizes_list, self.activation_function, self.loss_function, self.optimizer


    def load_latest_training_run(self): #TODO: placeholder implementation
        return self.latest_training_run


    def load_prediction_history_settings(self):
        #filepath, timesteps, date_column, close_column, n_out_jumps
        return self.prediction_history_filepath, self.timesteps, self.data_date_column_indice, self.data_label_column_indices[0], self.n_out_jumps

    def init_variables(self):
        self.unmodified_data_filepath = 'data/unmodified_data.h5'
        self.finished_data_filepath = 'data/finished_data.hdf5'
        self.prediction_history_filepath = 'data/prediction_history.pickle'
        self.pairs = ['USDT_BTC', 'USDT_ETH']
        self.start_dates = [1503446400, 1503446400]
        self.end_dates = [9999999999, 9999999999]
        self.timesteps = 300
        self.drop_data_column_indices = []
        self.data_date_column_indice = 1
        self.data_label_column_indices = [0]
        self.n_in = 20
        self.n_out = 4
        self.n_out_jumps = 1
        self.redownload_data = True
        self.use_scaling = True
        self.overwrite_scalers = False
        self.use_indicators = True
        self.overwrite_models = True
        self.batch_size = 128
        self.epochs = 50
        self.n_features = 6 + 8 * self.use_indicators - len(self.drop_data_column_indices)
        self.layer_sizes_list = [30, 20]
        self.activation_function = 'LeakyReLU'
        self.loss_function = 'mse'
        self.optimizer = 'adam'
        self.latest_training_run = 0

    def save_config(self):
        directory.ensure_directory(self.filepath)
        f = open(self.filepath, 'wb')
        pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_config(self):
        if directory.file_exists(self.filepath):
            f = open(self.filepath, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict)
            return True
        else:
            self.init_variables()
            return False






