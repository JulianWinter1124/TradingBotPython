import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

from util import data_enhancer


class Neural:

    def build_binary_model(self, training_data: pd.DataFrame, labels):  # https://keras.io/getting-started/sequential-model-guide/
        model = Sequential()
        model.add(Dense(32, input_shape=training_data.shape))
        model.add(Activation('relu'))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  # binary classification; might want to tune optimizer params with hyperopts
        one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)
        model.fit(training_data, one_hot_labels, epochs=10, batch_size=32)