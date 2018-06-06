import os


class Neural():

    def __init__(self, model_name, overwrite = False, units = 10, batch_size = 10, output_size = 1, regularizer = 1.0):
        from keras.models import Sequential
        from keras.callbacks import History, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        self.overwrite = overwrite
        self.units = units  # output dimension
        self.batch_size = batch_size  # how much data is processed at once
        self.regularizer = regularizer # no clue
        self.filepath = 'data/training/LSTM_' + model_name + '.h5'
        self.output_size = output_size
        self.model: Sequential = None
        # From https://www.kaggle.com/cbryant/keras-cnn-with-pseudolabeling-0-1514-lb/ might need tuning
        self.earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        self.mcp_save = ModelCheckpoint(self.filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        self.reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4,
                                                mode='min')

    def load_or_build_model(self, timesteps, nb_features):  # Model structure altered from https://github.com/khuangaf/CryptocurrencyPrediction/blob/master/LSTM.py
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Activation, LSTM, Dropout, LeakyReLU, regularizers
        if self.overwrite or not os.path.isfile(self.filepath):  # Is there no existing model?
            self.model = Sequential()
            self.model.add(LSTM(units=self.units, activity_regularizer=regularizers.l1(self.regularizer),
                           input_shape=(timesteps, nb_features), return_sequences=False))
            self.model.add(Activation('tanh'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.output_size))
            self.model.add(LeakyReLU())
            self.model.compile(loss='mse', optimizer='RMSprop')
            self.model.save(self.filepath)
        else:
            self.model = load_model(self.filepath)

    def train_model(self, train_X, train_Y, test_X, test_Y, epochs):
        history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, validation_data=(test_X, test_Y),
                            epochs=epochs, callbacks=[self.mcp_save])#, self.reduce_lr_loss])  # test
        # model.save(self.filepath) # should be done by ModelCheckpoint
        return history

    def predict(self, data):
        return self.model.predict(data)
