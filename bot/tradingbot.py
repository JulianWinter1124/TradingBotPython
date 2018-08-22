import time

from bot.data_collector import DataCollector
from bot.data_processor import DataProcessor
from util import botstate
from util.config_manager import BotConfigManager


class TradingBot():

    def __init__(self, config: BotConfigManager):
        self.config_manager = config
        self.data_collector = DataCollector(*self.config_manager.load_collector_settings())
        self.data_processor = DataProcessor(*self.config_manager.load_processor_settings())



    #execute all task within here
    def run(self, state):
        start = time.time()
        if state is 'RUN':
            self.state = "RUN"
            self.data_collector.download_and_save()

            self.data_processor.process_and_save()

        if state is "PAUSE":
            self.state = "PAUSE"

        return time.time()-start

    def perform_shutdown(self):
        #do stuff
        print("shutting down")
        raise SystemExit

    def parse_settings(self, config):
        self.currency_pairs


