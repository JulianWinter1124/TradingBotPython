from keras.callbacks import History, CSVLogger, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU

from util import data_enhancer


class Neural:

    def __init__(self):
        self.epochs = 10 #100
        self.output_size = 1
        self.units = 50
        self.second_units = 30
        self.batch_size = 8

    # TODO: make these non static
    def load_model(self, filepath):
        return load_model(filepath)

    def save_model(self, filepath, model):
        model.save(filepath)

    def build_model(self, train_X, train_y) -> Sequential:  # https://keras.io/getting-started/sequential-model-guide/
        model = Sequential()
        model.add(LSTM(units=self.units, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
        model.add(Dropout(0.8))
        model.add(Dense(self.output_size))
        model.add(LeakyReLU())
        model.compile(loss='mse', optimizer='adam')
        return model

    def build_model_2(self, train_X):
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        return model

    def train_model_2(self, model, train_X, train_y, test_X, test_y) -> History:
        return model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                            shuffle=False)

    def train_model(self, model, train_X, train_y, test_X, test_y) -> History:
        # return model.fit(train_X, train_y, epochs=100, batch_size=8, verbose=2, validation_data=(test_X, test_y), shuffle=False)

        return model.fit(train_X, train_y, batch_size=self.batch_size,
              validation_data=(test_X, test_y), epochs=self.epochs)
