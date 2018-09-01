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
        self.prediction_history = PredictionHistory(*self.config_manager.load_prediction_history_settings())


    #execute all task within here
    def run(self, state):
        start = time.time()
        if state is 'RUN':
            self.state = "RUN"
            self.data_collector.download_and_save() #collect latest data for all pairs

            self.data_processor.process_and_save() #process data into train-ready data

            if time.time() - self.config_manager.latest_training_run > 6 * 60 * 60: #Train new all 6 hours

                self.config_manager.latest_training_run = time.time() #save when latest training run was executed

                self.neural_manager.train_models(plot_history=True) #Train all models (data in train-ready file)

                self.data_collector.download_and_save() #update data (training took some time)

                self.data_processor.process_and_save()

            else:
                print('skipping training because not enough time has passed since')

            scalers = self.data_processor.get_scaler_dict() #loads the scaler from the data processor

            dates, predictions = self.neural_manager.make_latest_predictions(scalers, look_back=0) #make latest predictions for latest data column (unmodified_data)

            for pair, values in predictions.items():

                self.prediction_history.add_prediction(pair, dates[pair], values) #add prediction to the history

                self.prediction_history.plot_prediction_history(pair, self.data_collector.get_original_data(pair)) #plot all predictions from history

                action = decision.decide_action_on_prediction(pair, values, None, 0.8) #Decide which action to take base on prediction

                print(action)

                #state.perform_action(action=action) #Perform the given action.




        if state is "PAUSE":
            self.state = "PAUSE"
            #there is nothing to stop right now, but if there was put it here

        return time.time()-start

    def perform_shutdown(self):
        self.config_manager.save_config()
        print("shutting down")
        raise SystemExit

    def parse_settings(self, config):
        pass


