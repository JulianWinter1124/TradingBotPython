import keras
import pandas as pd
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

from util import data_enhancer


class Neural:

    def build_model(self, train_X) -> Sequential:  # https://keras.io/getting-started/sequential-model-guide/
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mae')
        return model

    def train_model(self, model, train_X, train_Y, test_X, test_y) -> History:
        return model.fit(train_X, train_Y, epochs=100, batch_size=72, verbose=1, validation_data=(test_X, test_y), shuffle=False)