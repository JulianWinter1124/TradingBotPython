import time
from collections import defaultdict

from bot.data_collector import DataCollector
from bot.data_processor import DataProcessor
from bot.prediction_history import PredictionHistory
from bot_ai.neural_manager import NeuralManager
from util.config_manager import BotConfigManager
from bot_ai import decision


class TradingBot():

    def __init__(self, config: BotConfigManager):
        self.config_manager = config
        self.data_collector = DataCollector(*self.config_manager.load_collector_settings())
        self.data_processor = DataProcessor(*self.config_manager.load_processor_settings())
        self.neural_manager = NeuralManager(*self.config_manager.load_neural_manager_settings())
        self.prediction_history = PredictionHistory('data/history.pickle')
        self.latest_training_run = self.config_manager.load_latest_training_run()


    #execute all task within here
    def run(self, state):
        start = time.time()
        if state is 'RUN':
            self.state = "RUN"
            self.data_collector.download_and_save()

            self.data_processor.process_and_save()

            if time.time() - self.latest_training_run > 6 * 60 * 60: #Train new all 6 hours

                self.latest_training_run = max(self.data_collector.get_latest_dates().values())

                self.neural_manager.train_models()

                self.data_collector.download_and_save()

                self.data_processor.process_and_save()

            else:
                print('skipping training because not enough time has passed since')

            scalers = self.data_processor.get_scaler_dict()

            dates, predictions = self.neural_manager.make_latest_predictions(scalers)

            for pair, values in predictions.items():

                self.prediction_history.add_prediction(pair, dates[pair], values)

                self.prediction_history.plot_prediction_history(pair, self.data_collector.get_original_data(pair), n_out_jumps=1)

                action = decision.decide_action_on_prediction(pair, values, None, 0.8)

                print(action)







        if state is "PAUSE":
            self.state = "PAUSE"
            #there is nothing to stop right now, but if there was put it here

        return time.time()-start

    def perform_shutdown(self):
        #do stuff
        print("shutting down")
        raise SystemExit

    def parse_settings(self, config):
        pass


