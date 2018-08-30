import datetime
import pickle

import numpy as np

from matplotlib import pyplot as plt

import directory


class PredictionHistory():

    def __init__(self, filepath):
        self.history : dict = dict()
        self.filepath = directory.get_absolute_path(filepath)
        self.load_from_file()
        pass


    def add_prediction(self, pair, prediction_date, prediction):
        if pair in  self.history:
            if not prediction_date in self.history[pair]:
                self.history[pair][prediction_date] = prediction
            else:
                print('prediction for', prediction_date, 'already in history. Skipping')
                return
        else:
            self.history[pair] = dict()
            self.history[pair][prediction_date] = prediction
        self.save_to_file()

    def plot_prediction_history(self, pair, original_data, date_column=1, close_column=0, timesteps=1800, n_out_jumps=1):
        prediction_data = self.history[pair]
        n = len(prediction_data.keys())
        plt.figure(figsize=(16,12))
        plt.plot(original_data[-(n+100):, date_column], original_data[-(n+100):, close_column], label='Original')
        for date in sorted(prediction_data.keys()):
            dates = np.arange(date, date + timesteps*n_out_jumps*len(prediction_data[date][0]), timesteps*n_out_jumps)
            plt.plot(dates, prediction_data[date][0]) #both are 2D arrays
        plt.legend(loc='best')
        plt.show()


    def load_from_file(self):
        if directory.file_exists(self.filepath):
            with open(self.filepath, 'rb') as handle:
                self.history = pickle.load(handle)

    def save_to_file(self):
        with open(self.filepath, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)