import os



class Neural():

    def __init__(self, model_name, overwrite = False, batch_size = 10, output_size = 1):
        from keras.models import Sequential
        from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        self.overwrite = overwrite
        self.batch_size = batch_size  # how much data is processed at once
        self.filepath = 'data/training/LSTM_' + model_name + '.h5'
        self.output_size = output_size
        self.model: Sequential = None
        # From https://www.kaggle.com/cbryant/keras-cnn-with-pseudolabeling-0-1514-lb/ might need tuning
        self.earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        self.mcp_save = ModelCheckpoint(self.filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        self.reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4,
                                                mode='min')

    def load_or_build_model(self, n_in, n_out, n_features, layer_units=[10, 10, 10], activation_function='linear', loss_function='tanh', optimizer='adam'):
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

    def train_model(self, train_X, train_Y, test_X, test_Y, epochs, shuffle=False, save=True):
        if save:
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_data=(test_X, test_Y),
                                     epochs=epochs, callbacks=[self.mcp_save, self.earlyStopping],
                                     shuffle=shuffle)
        else:
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_data=(test_X, test_Y),
                                     epochs=epochs, callbacks=[self.earlyStopping],
                                     shuffle=shuffle)
        return history

    def train_model_generator(self, generator, steps_per_epoch, epochs, use_multiprocessing=False, workers=2): #mp not on windows LUL
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
