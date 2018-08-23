import time

from bot.data_collector import DataCollector
from bot.data_processor import DataProcessor
from bot_ai.neural_manager import NeuralManager
from util import botstate
from util.config_manager import BotConfigManager


class TradingBot():

    def __init__(self, config: BotConfigManager):
        self.config_manager = config
        self.data_collector = DataCollector(*self.config_manager.load_collector_settings())
        self.data_processor = DataProcessor(*self.config_manager.load_processor_settings())
        self.neural_manager = NeuralManager(*self.config_manager.load_neural_manager_settings())


    #execute all task within here
    def run(self, state):
        start = time.time()
        if state is 'RUN':
            self.state = "RUN"
            print('downloader startet')
            self.data_collector.download_and_save()
            print('downloader finished')

            print('processor started')
            self.data_processor.process_and_save()
            print('processor finished')

            print('nm started')
            self.neural_manager.run()
            print('nm finished')

        if state is "PAUSE":
            self.state = "PAUSE"

        return time.time()-start

    def perform_shutdown(self):
        #do stuff
        print("shutting down")
        raise SystemExit

    def parse_settings(self, config):
        self.currency_pairs


