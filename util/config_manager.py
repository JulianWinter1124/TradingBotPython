import configparser
import abc


class ConfigManager(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def save_config(self):
        pass

    @abc.abstractmethod
    def load_config(self) -> bool:
        pass

    @abc.abstractmethod
    def create_empty_config(self):
        pass


class BotConfigManager(ConfigManager):

    def __init__(self, overwrite=False):
        super(BotConfigManager, self).__init__()
        self.filename = 'config.ini'
        self.config = configparser.ConfigParser(allow_no_value = True)
        if overwrite:
            self.create_empty_config()

    def save_config(self):
        with open(self.filename, 'w') as cfg:
            self.config.write(cfg)

    def load_config(self) -> bool:
        self.config.read(self.filename)
        if self.config.sections() is []:
            return False
        else:
            return True

    def create_empty_config(self):
        self.config.add_section('currency_pairs')
        self.config.set('currency_pairs', '; The currency pairs listed on poloniex.com')
        self.config.add_section('filenames')
        self.config.set('filenames', 'pair_data_filepath', 'data/finished_data_unmodified.h5')
        self.config.set('filenames', 'finished_data_filepath', 'data/finished_data.h5')

    def load_collector_settings(self): #TODO: placeholder implementation
        #return 'data/unmodified_data.h5', ['USDT_BTC', 'USDT_ETH', 'BTC_XRP'], [1503446400, 1503446400, 1503446400], [9999999999, 9999999999, 1506446400], [300, 300, 300], False
        return 'data/unmodified_data.h5', ['USDT_BTC', 'USDT_ETH'], [1503446400, 1503446400], [9999999999, 9999999999], [300,300], False

    def load_processor_settings(self): #TODO: placeholder implementation
        #database_filepath, output_filepath, use_indicators=True, use_scaling=True, drop_data_columns_indices: list = [], label_column_indices=[0], n_in=30, n_out=2, n_out_jumps=1, overwrite_scaler=False)
        return 'data/unmodified_data.h5', 'data/finished_data.hdf5', True, True, [], [0], 30, 4, 1, False

    def load_neural_manager_settings(self): #TODO: placeholder implementation
        #return finished_data_filepath, overwrite_models, batch_size, epochs, output_size, n_in, n_out, n_features, use_scaling, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='LeakyReLU', optimizer='adam'
        return 'data/unmodified_data.h5', 'data/finished_data.hdf5', False, 128, 35, 4+1, 30, 4, 6 + 8 - 0, True, [30, 20, 20], 'LeakyReLU', 'mse', 'adam'

    def load_latest_training_run(self): #TODO: placeholder implementation
        return 99999999999

    def load_prediction_history_settings(self):
        return 'data/prediction_history.pickle'



