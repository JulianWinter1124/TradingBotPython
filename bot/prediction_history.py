import datetime
import pickle

import numpy as np

from matplotlib import pyplot as plt

import directory


class PredictionHistory():

    def __init__(self, filepath, timesteps, date_column, close_column, n_out_jumps, overwrite_history):
        self.timesteps = timesteps
        self.n_out_jumps = n_out_jumps
        self.close_column = close_column
        self.date_column = date_column
        self.history : dict = dict()
        self.filepath = directory.get_absolute_path(filepath)
        if not overwrite_history:
            self.load_from_file()

    def add_prediction(self, pair, prediction_date, prediction):
        if pair in self.history:
            if not prediction_date in self.history[pair]:
                self.history[pair][prediction_date] = prediction
            else:
                print('prediction for', prediction_date, 'already in history. Skipping')
                return
        else:
            self.history[pair] = dict()
            self.history[pair][prediction_date] = prediction
        self.save_to_file()

    def add_multiple_predictions(self, pair, dates, predictions):
        for i in range(len(predictions)):
            prediction = predictions[i, :]
            prediction_date = dates[i]
            if pair in self.history:
                if not prediction_date in self.history[pair]:
                    self.history[pair][prediction_date] = prediction
                    print(prediction)
                else:
                    print('prediction for', prediction_date, 'already in history. Skipping')
                    return
            else:
                self.history[pair] = dict()
                self.history[pair][prediction_date] = prediction
            self.save_to_file()

    def plot_prediction_history(self, pair, original_data):
        prediction_data = self.history[pair]
        n = len(prediction_data.keys())
        plt.figure(figsize=(16,12))
        plt.plot(original_data[-(n+100):, self.date_column], original_data[-(n+100):, self.close_column], label='Original')
        for date in sorted(prediction_data.keys()):
            starting_date = date + self.timesteps*self.n_out_jumps
            dates = np.arange(starting_date, starting_date + self.timesteps*self.n_out_jumps*len(prediction_data[date][0]), self.timesteps*self.n_out_jumps)
            predictions = prediction_data[date][0]
            plt.plot(dates, predictions) #both are 2D arrays
        plt.legend(loc='best')
        plt.show()


    def load_from_file(self):
        if directory.file_exists(self.filepath):
            with open(self.filepath, 'rb') as handle:
                self.history = pickle.load(handle)

    def save_to_file(self):
        with open(self.filepath, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def clear_history(self):
        del self.history
        self.history = dict()