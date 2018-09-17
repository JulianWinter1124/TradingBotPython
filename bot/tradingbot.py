import logging
import time
from collections import defaultdict

from bot.data_collector import DataCollector
from bot.data_processor import DataProcessor
from bot.prediction_history import PredictionHistory
from bot.simulation import Simulation
from bot_ai.neural_manager import NeuralManager
from util.config_manager import BotConfigManager
from bot_ai import decision

from matplotlib import pyplot as plt


class TradingBot():

    def __init__(self, config: BotConfigManager):
        self.config_manager = config
        self.data_collector = DataCollector(*self.config_manager.load_collector_settings())
        self.data_processor = DataProcessor(*self.config_manager.load_processor_settings())
        self.neural_manager = NeuralManager(*self.config_manager.load_neural_manager_settings())
        self.prediction_history = PredictionHistory(*self.config_manager.load_prediction_history_settings())
        self.training_prediction_history = PredictionHistory(*self.config_manager.load_training_prediction_history_settings())
        self.simulation = Simulation(*self.config_manager.load_simulation_settings(), 'Normal')  # The normal bot simulation
        self.simulation2 = Simulation(*self.config_manager.load_simulation_settings(), 'Random')  # simulation for the random bot
        self.count = 0


    #execute all task within here
    def run(self):

        start = time.time() #Save start for execution time

        self.data_collector.download_and_save() #collect latest data for all pairs

        self.data_processor.process_and_save() #process data into train-ready data

        scalers = self.data_processor.get_scaler_dict() #loads the scaler from the data processor

        if time.time() - self.config_manager.latest_training_run > self.config_manager.train_every_n_seconds: #Train new all n seconds

            print("Training model because {} seconds have passed...".format(time.time() - self.config_manager.latest_training_run))

            self.neural_manager.train_models(plot_history=True) #Train all models (data in train-ready file)

            self.config_manager.latest_training_run = time.time()  # save when latest training run was executed

            self.config_manager.overwrite_models = False #Reset this param

            self.config_manager.overwrite_scalers = False #This one too

            self.config_manager.save_config() #Save now in case it crashes

            dates, predictions, original = self.neural_manager.predict_all_data(scalers)
            for pair, values in predictions.items(): #Plotting the complete data with their predictions
                plt.figure(figsize=(16, 8))
                plt.plot(dates[pair], values[:, 0], label='prediction:' + pair, linewidth=0.5)
                plt.plot(dates[pair], original[pair], label='original:' + pair, linewidth=0.5)
                plt.legend()
                plt.show()

            self.data_collector.download_and_save() #update data (training took some time)

            self.data_processor.process_and_save()

        else:
            print('skipping training because not enough time has passed since')


        dates, predictions = self.neural_manager.predict_latest_date(scalers) #make latest predictions for latest data column (unmodified_data)

        for pair, values in predictions.items():

            self.prediction_history.add_prediction(pair, dates[pair], values) #add prediction to the history

            if self.count % self.config_manager.display_plot_interval == 0: #only plot at given interval
                self.prediction_history.plot_prediction_history(pair, self.data_collector.get_original_data(pair)) #plot all predictions from history

            closing_price = self.data_collector.get_latest_closing_price(pair)

            action = decision.decide_action_on_prediction(pair, values, self.simulation,  closing_price, False, 0.8) #Decide which action to take base on prediction

            action_random = decision.make_random_action(pair, self.simulation2, closing_price) #make a random action

            print(decision.stringify_action(action)) #Print the action the NORMAL bot is going to take

            self.simulation.perform_action(dates[pair], action=action) #Perform the given action.

            self.simulation2.perform_action(dates[pair], action=action_random)

        min_date = min(dates.values())
        self.simulation.update_account_standing_history(min_date) #Update the account_history data after actions have been performed
        self.simulation2.update_account_standing_history(min_date)

        if self.count % self.config_manager.display_plot_interval == 0: #only plot at given interval
            self.simulation.plot_account_history('actual bot')
            self.simulation2.plot_account_history('random bot')

        self.count += 1 #count run iterations

        return time.time()-start

    def perform_shutdown(self):
        """
        This method is called when a shutdown is performed
        """
        self.config_manager.save_config()
        logging.warning("shutting down")
        raise SystemExit
