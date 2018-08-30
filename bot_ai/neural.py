import os



class Neural():

    def __init__(self, model_name, overwrite, batch_size, output_size, n_in, n_out, n_features, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='mse', optimizer='adam'):
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.n_out = n_out
        self.activation_function = activation_function
        self.layer_units = layer_units
        self.n_in = n_in
        self.n_features = n_features
        from keras.models import Sequential
        from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        self.overwrite = overwrite
        self.batch_size = batch_size  # how much data is processed at once
        self.filepath = 'data/training/LSTM_' + model_name + '.hdf5'
        self.output_size = output_size
        self.model: Sequential = None
        # From https://www.kaggle.com/cbryant/keras-cnn-with-pseudolabeling-0-1514-lb/ might need tuning
        self.earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        self.mcp_save = ModelCheckpoint(self.filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        self.reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4,
                                                mode='min')

    def load_or_build_model_v0(self):
        """
        Loads the model from file or creates one if not existent
        """
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, Flatten
        if self.overwrite or not os.path.isfile(self.filepath):  # Is there no existing model?
            self.model = Sequential()
            self.model.add(LSTM(units=self.layer_units[0], input_shape=(self.n_in, self.n_features), return_sequences=True))
            if self.activation_function == 'LeakyReLU':
                self.model.add(LeakyReLU(alpha=.001))
            else:
                self.model.add(Activation(self.activation_function))
            self.model.add(Dropout(0.2))
            for i in range(1, len(self.layer_units)):
                self.model.add(LSTM(units=self.layer_units[i], input_shape=(self.n_in, self.n_features), return_sequences=True))
                self.model.add(Dropout(0.2))
            self.model.add(Flatten())
            self.model.add(Dense(self.output_size))
            self.model.compile(loss=self.loss_function, optimizer=self.optimizer)
            self.model.save(self.filepath, True)
        else:
            self.model = load_model(self.filepath)
        self.model.summary()

    def load_or_build_model(self):
        """
        Loads the model from file or creates one if not existent
        """
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, Flatten
        if self.overwrite or not os.path.isfile(self.filepath):  # Is there no existing model?
            self.model = Sequential()
            for i in range(0, len(self.layer_units)-1):
                self.model.add(LSTM(units=self.layer_units[i], input_shape=(self.n_in, self.n_features), return_sequences=True))
                if self.activation_function == 'LeakyReLU':
                    self.model.add(LeakyReLU(alpha=.001))
                else:
                    self.model.add(Activation(self.activation_function))
                self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=self.layer_units[-1], input_shape=(self.n_in, self.n_features), return_sequences=True, activation='softmax'))
            self.model.add(Dropout(0.2))
            self.model.add(Flatten())
            self.model.add(Dense(self.output_size))
            self.model.compile(loss=self.loss_function, optimizer=self.optimizer)
            self.model.save(self.filepath, True)
        else:
            self.model = load_model(self.filepath)
        self.model.summary()

    def train_model_v0(self, train_X, train_Y, test_X, test_Y, epochs, shuffle=False, save=True):
        """
        Trains the model with the given data
        :param train_X:
        :param train_Y:
        :param test_X:
        :param test_Y:
        :param epochs:
        :param shuffle:
        :param save:
        :return:
        """
        if save:
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_data=(test_X, test_Y),
                                     epochs=epochs, callbacks=[self.mcp_save, self.earlyStopping],
                                     shuffle=shuffle)
        else:
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_data=(test_X, test_Y),
                                     epochs=epochs, callbacks=[self.earlyStopping],
                                     shuffle=shuffle)
        return history

    def train_model(self, train_X, train_Y, test_X, test_Y, epochs, shuffle=False, save=True):
        """
        Trains the model with the given data
        :param train_X:
        :param train_Y:
        :param test_X:
        :param test_Y:
        :param epochs:
        :param shuffle:
        :param save:
        :return:
        """
        if save:
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_split=0.1,
                                     epochs=epochs, callbacks=[self.mcp_save, self.earlyStopping],
                                     shuffle=shuffle)
        else:
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_split=0.1,
                                     epochs=epochs, callbacks=[self.earlyStopping],
                                     shuffle=shuffle)
            self.model.evaluate(test_X, test_Y, batch_size=self.batch_size)
        return history


    def train_model_generator(self, generator, steps_per_epoch, epochs, use_multiprocessing=True, workers=4): #mp not on windows LUL
        history = self.model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch, epochs=epochs, use_multiprocessing=use_multiprocessing, workers=workers)
        return history

    def plot_model_to_file(self, filename, show_shapes, show_layer_names):
        from keras.utils import plot_model
        print('Plotting model to', filename)
        plot_model(self.model, filename, show_shapes, show_layer_names)

    def predict(self, data):
        return self.model.predict(data)

    def predict_on_batch(self, data):
        return self.model.predict_on_batch(data)
