import logging
import pickle

import numpy as np

from matplotlib import pyplot as plt

import directory

logger = logging.getLogger('prediction_history')

#a class to store all predictions and visualize them
class PredictionHistory():

    def __init__(self, filepath, timesteps, date_column, close_column, n_out_jumps, overwrite_history):
        self.timesteps = timesteps #All params explained in config_manager.py
        self.n_out_jumps = n_out_jumps
        self.close_column = close_column
        self.date_column = date_column
        self.history : dict = dict()
        self.filepath = directory.get_absolute_path(filepath)
        if not overwrite_history:
            self.load_from_file() # load the history from file if available

    def add_prediction(self, pair, prediction_date, prediction):
        """
        adds a single prediction to the history
        :param pair: the crypto pair this prediction belings to
        :param prediction_date: the date at which the prediction is made at (latest data is available at this date)
        :param prediction: the single row prediction data
        :return: None
        """
        if pair in self.history:
            if not prediction_date in self.history[pair]:
                self.history[pair][prediction_date] = prediction #add prediction, if no prediction for that date is available,
                logger.info('prediction for {} at {} put into history.'.format(pair, prediction_date))
            else:
                logger.warning('prediction for {} already in history. Skipping.'.format(prediction_date))
                return
        else:
            self.history[pair] = dict() #if no prediction for that pair has been saved yet, make a new dict
            self.history[pair][prediction_date] = prediction
        logger.info('prediction for {} at {} put into history.'.format(pair, prediction_date))
        self.save_to_file() #save history after each addition


    def plot_prediction_history(self, pair, original_data):
        """
        Plots the saved prediction history, together with original data
        :param pair: still the crypto pair
        :param original_data: original unaltered data from data_collector. this includes all columns
        """
        prediction_data = self.history[pair]
        n = len(prediction_data.keys())
        plt.figure(figsize=(16,12))
        plt.plot(original_data[-(n+100):, self.date_column], original_data[-(n+100):, self.close_column], label='Original') #Print only the last 100+number colse values of predictions data, for visibility reasons
        for date in sorted(prediction_data.keys()):
            starting_date = date + self.timesteps*self.n_out_jumps #the prediction starts at current date + offset
            dates = np.arange(starting_date, starting_date + self.timesteps*self.n_out_jumps*len(prediction_data[date][0]), self.timesteps*self.n_out_jumps) #calculate the dates for which the prediction points have been made
            predictions = prediction_data[date][0]
            plt.plot(dates, predictions) #both are 2D arrays
        plt.legend(loc='best')
        plt.show() #block=False does not work appearantly


    def load_from_file(self):
        """
        load the pickle file
        """
        if directory.file_exists(self.filepath):
            with open(self.filepath, 'rb') as handle:
                self.history = pickle.load(handle)
            logger.info('loaded history from {}', self.filepath)

    def save_to_file(self):
        """
        save the pickle file
        """
        with open(self.filepath, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('prediction history saved to {}'.format(self.filepath))

    def clear_history(self):
        """
        clears the history
        not used right now as overwrite exists, and at what point in runtime should this be deleted?
        """
        del self.history
        self.history = dict()
        logger.info('History cleared')