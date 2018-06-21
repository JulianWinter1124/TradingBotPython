import os



class Neural():

    def __init__(self, model_name, overwrite = False, units = 10, batch_size = 10, output_size = 1, regularizer = 1.0,
                 activation='tanh'):
        from keras.models import Sequential
        from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        self.overwrite = overwrite
        self.units = units  # output dimension
        self.batch_size = batch_size  # how much data is processed at once
        self.regularizer = regularizer # no clue
        self.filepath = 'data/training/LSTM_' + model_name + '.h5'
        self.output_size = output_size
        self.model: Sequential = None
        self.activation = activation
        # From https://www.kaggle.com/cbryant/keras-cnn-with-pseudolabeling-0-1514-lb/ might need tuning
        self.earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        self.mcp_save = ModelCheckpoint(self.filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        self.reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4,
                                                mode='min')

    def load_or_build_model(self, n_in, n_features):  # Model structure altered from https://github.com/khuangaf/CryptocurrencyPrediction/blob/master/LSTM.py
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, regularizers, Flatten
        if self.overwrite or not os.path.isfile(self.filepath):  # Is there no existing model?
            self.model = Sequential()
            self.model.add(LSTM(units=self.units, activity_regularizer=regularizers.l1(self.regularizer),
                                input_shape=(n_in, n_features), return_sequences=False))
            self.model.add(Activation(self.activation))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.output_size))
            self.model.add(LeakyReLU())
            self.model.compile(loss='mse', optimizer='adam')
            self.model.save(self.filepath)
        else:
            self.model = load_model(self.filepath)

        return self.model

    def load_or_build_model_2(self, n_in, n_out, n_features, layer_units=[10, 10, 10], activation_function='linear', loss_function='tanh', optimizer='adam'):
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, regularizers, Flatten
        if self.overwrite or not os.path.isfile(self.filepath):  # Is there no existing model?
            self.model = Sequential()
            self.model.add(LSTM(units=layer_units[0], input_shape=(n_in, n_features), return_sequences=True))
            self.model.add(Activation(activation_function))
            self.model.add(Dropout(0.2))
            for i in range(1, len(layer_units)):
                self.model.add(LSTM(units=layer_units[i], input_shape=(n_in, n_features), return_sequences=True))
                self.model.add(Dropout(0.2))
            self.model.add(Flatten())
            self.model.add(Dense(n_out+1))
            self.model.compile(loss=loss_function, optimizer=optimizer)
            self.model.save(self.filepath)
        else:
            self.model = load_model(self.filepath)

    def train_model(self, train_X, train_Y, test_X, test_Y, epochs, shuffle=False):
        history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_data=(test_X, test_Y),
                            epochs=epochs, callbacks=[self.mcp_save, self.earlyStopping], shuffle=shuffle)#, self.reduce_lr_loss])  # test
        # model.save(self.filepath) # should be done by ModelCheckpoint
        return history

    def train_model_generator(self, generator, steps_per_epoch, epochs, use_multiprocessing=False, workers=2):
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
