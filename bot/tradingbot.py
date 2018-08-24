import time

from bot.data_collector import DataCollector
from bot.data_processor import DataProcessor
from bot_ai.neural_manager import NeuralManager
from util import botstate
from util.config_manager import BotConfigManager
from bot_ai import decision


class TradingBot():

    def __init__(self, config: BotConfigManager):
        self.config_manager = config
        self.data_collector = DataCollector(*self.config_manager.load_collector_settings())
        self.data_processor = DataProcessor(*self.config_manager.load_processor_settings())
        self.neural_manager = NeuralManager(*self.config_manager.load_neural_manager_settings())
        self.latest_training_run = self.config_manager.load_latest_training_run()


    #execute all task within here
    def run(self, state):
        start = time.time()
        if state is 'RUN':
            self.state = "RUN"
            self.data_collector.download_and_save()

            self.data_processor.process_and_save()

            if time.time() - self.latest_training_run > 24 * 60 * 60: #This is a naive day approach

                self.latest_training_run = self.data_collector.get_maximum_latest_date()

                self.neural_manager.train_models()

                self.data_collector.download_and_save()

                self.data_processor.process_and_save()

            else:
                print('skipping training because no day has passed since')

            scalers = self.data_processor.get_scaler_dict()

            predicitions = self.neural_manager.make_latest_predictions(scalers)

            for key, value in predicitions.items():

                action = decision.decide_action_on_prediction(value, 0.8)

                print(action)






        if state is "PAUSE":
            self.state = "PAUSE"

        return time.time()-start

    def perform_shutdown(self):
        #do stuff
        print("shutting down")
        raise SystemExit

    def parse_settings(self, config):
        pass


